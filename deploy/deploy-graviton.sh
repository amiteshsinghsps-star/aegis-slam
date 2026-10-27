#!/usr/bin/env bash
# deploy/deploy-graviton.sh — provision a single Graviton (c8g) EC2 instance and
# run AEGIS-SLAM in Docker. Idempotent-ish and conservative: it does NOT open
# SSH; admin access is via SSM Session Manager. Review the variables first.
#
# Requires: aws cli v2 configured with credentials that can create EC2/IAM/S3.
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_REGION
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$AWS_REGION}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c8g.large}"          # Graviton4 (ARM64)
AMI_PARAM="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
KEY_NAME="${KEY_NAME:-}"                            # optional; SSM is preferred
SG_NAME="aegis-slam-sg"
ROLE_NAME="aegis-slam-role"
BUCKET="aegis-slam-uploads-$(aws sts get-caller-identity --query Account --output text)"
VPC_ID="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"

echo "Region=$AWS_REGION  Instance=$INSTANCE_TYPE  Bucket=$BUCKET"

# 1) S3 bucket + retention lifecycle (auto-delete user video after 1 day)
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$AWS_REGION"
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$AWS_REGION" \
      --create-bucket-configuration LocationConstraint="$AWS_REGION"
  fi
fi
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" \
  --lifecycle-configuration "file://$(dirname "$0")/s3/lifecycle.json"
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# 2) IAM instance role (least privilege) + profile
if ! aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document "file://$(dirname "$0")/iam/trust-policy.json"
  aws iam create-instance-profile --instance-profile-name "$ROLE_NAME" 2>/dev/null || true
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE_NAME" --role-name "$ROLE_NAME"
fi
POLICY_ARN="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$ROLE_NAME-policy"
if ! aws iam get-policy --policy-arn "$POLICY_ARN" 2>/dev/null; then
  sed "s/REPLACE_ACCOUNT/$(aws sts get-caller-identity --query Account --output text)/g" \
    "$(dirname "$0")/iam/instance-policy.json" > /tmp/aegis-policy.json
  aws iam create-policy --policy-name "$ROLE_NAME-policy" \
    --policy-document file:///tmp/aegis-policy.json
fi
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$POLICY_ARN"
# Session Manager needs the AWS managed policy — without it "no SSH" would
# simply mean "no access". Attached, not inlined (it is AWS-maintained).
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# 3) Security group: only 8000 from your IP (or put an ALB in front for prod).
# FAIL CLOSED: if we cannot determine your IP we do NOT open the port at all —
# add the rule manually later. Never fall back to 0.0.0.0/0.
MY_IP_RAW="$(curl -s --max-time 10 https://checkip.amazonaws.com || true)"
MY_IP_RAW="$(printf '%s' "$MY_IP_RAW" | tr -d '[:space:]')"
if [[ ! "$MY_IP_RAW" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "WARNING: could not detect your public IP; port 8000 stays CLOSED."
  echo "Open it later with: aws ec2 authorize-security-group-ingress --group-id <sg> --proto tcp --port 8000 --cidr <your-ip>/32"
  MY_IP=""
else
  MY_IP="${MY_IP_RAW}/32"
fi
SG_ID="$(aws ec2 create-security-group --group-name "$SG_NAME" \
  --description "AEGIS-SLAM" --vpc-id "$VPC_ID" --query GroupId --output text 2>/dev/null \
  || aws ec2 describe-security-groups --group-names "$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)"
if [ -n "$MY_IP" ]; then
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --protocol tcp --port 8000 --cidr "$MY_IP" 2>/dev/null || true
fi

# 4) Launch Graviton instance (SSM-only, no key required, IMDSv2 enforced)
# cloud-init does not inherit this shell's environment, so bake REPO_URL into a rendered copy.
: "${REPO_URL:?set REPO_URL to your public git URL before deploying (see deploy/README.md)}"
USER_DATA="$(mktemp /tmp/aegis-user-data.XXXXXX.sh)"
sed "s#__REPO_URL__#${REPO_URL}#g" "$(dirname "$0")/user-data.sh" > "$USER_DATA"
echo "Waiting 15s for the new instance profile to propagate..."; sleep 15
INSTANCE_ID="$(aws ec2 run-instances \
  --image-id "resolve:ssm:$AMI_PARAM" \
  --instance-type "$INSTANCE_TYPE" \
  --region "$AWS_REGION" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="$ROLE_NAME" \
  --user-data "file://$USER_DATA" \
  --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=aegis-slam}]" \
  --query 'Instances[0].InstanceId' --output text)"

PUBLIC_DNS="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"

cat > "$(dirname "$0")/.deploy-state" <<EOF
INSTANCE_ID="$INSTANCE_ID"
SG_ID="$SG_ID"
ROLE_NAME="$ROLE_NAME"
BUCKET_NAME="$BUCKET"
INSTANCE_TYPE="$INSTANCE_TYPE"
PUBLIC_DNS="$PUBLIC_DNS"
EOF

# 5) Server-side observability + cost guard (both non-fatal: never lose the box
#    because a dashboard call was denied).
export AEGIS_S3_BUCKET="${AEGIS_S3_BUCKET:-$BUCKET}"
export INSTANCE_USD_PER_HOUR="${INSTANCE_USD_PER_HOUR:-}"
export NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"
export BUDGET_USD="${BUDGET_USD:-5}"
bash "$(dirname "$0")/setup-cloudwatch.sh" || \
  echo "WARNING: CloudWatch setup incomplete — metrics/logs still work if the role allows them."
if [ "${COST_GUARD:-1}" = "1" ]; then
  INSTANCE_ID="$INSTANCE_ID" bash "$(dirname "$0")/cost-guard.sh" || \
    echo "WARNING: cost guard not armed — do NOT rely on manual shutdown discipline alone."
else
  echo "cost guard SKIPPED (COST_GUARD=0). The instance bills until you stop it."
fi

echo
echo "Launched $INSTANCE_ID on $INSTANCE_TYPE (IMDSv2 required, no SSH key)."
echo "Wait ~2-3 min for Docker + compose, then:"
echo "  UI      http://$PUBLIC_DNS:8000"
echo "  proof   curl http://$PUBLIC_DNS:8000/version   # OpenCV 5 + Graviton + IMDSv2"
echo "  shell   aws ssm start-session --target $INSTANCE_ID --region $AWS_REGION"
echo "  logs    aws logs get-log-events --log-group-name /aegis-slam/service \\ "
echo "            --log-stream-name aegis-slam --region $AWS_REGION"
echo "  stop    aws ec2 stop-instances --instance-ids $INSTANCE_ID"
echo "  teardown  bash deploy/teardown.sh"
