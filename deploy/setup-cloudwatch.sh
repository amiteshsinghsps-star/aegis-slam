#!/usr/bin/env bash
# deploy/setup-cloudwatch.sh — logs group + retention + job-metrics dashboard.
#
# The service publishes AEGISSlam/* metrics itself (aegis/metrics.py) and Docker
# writes stdout to CloudWatch Logs with the awslogs driver; this creates the
# server-side objects those two need, plus one dashboard that shows the agent's
# behaviour per job. Safe to re-run.
set -uo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
LOG_GROUP="${LOG_GROUP:-/aegis-slam/service}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
DASHBOARD_NAME="${DASHBOARD_NAME:-aegis-slam}"
ACCT="$(aws sts get-caller-identity --query Account --output text)"
STATE_FILE="$(dirname "$0")/.deploy-state"

echo "==> Log group $LOG_GROUP ($RETENTION_DAYS-day retention)"
if aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" \
     --query "logGroups[?(logGroupName=='$LOG_GROUP')].logGroupName" --output text \
     | grep -q "$LOG_GROUP"; then
  echo "    exists"
else
  aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$AWS_REGION" >/dev/null
  echo "    created"
fi
aws logs put-retention-policy --log-group-name "$LOG_GROUP" \
  --retention-in-days "$RETENTION_DAYS" --region "$AWS_REGION"

echo "==> Dashboard $DASHBOARD_NAME"
sed -e "s/__REGION__/$AWS_REGION/g" -e "s/__ACCOUNT__/$ACCT/g" \
  "$(dirname "$0")/cloudwatch/dashboard.json" > /tmp/aegis-dashboard.json
aws cloudwatch put-dashboard --dashboard-name "$DASHBOARD_NAME" \
  --dashboard-body file:///tmp/aegis-dashboard.json --region "$AWS_REGION" >/dev/null
rm -f /tmp/aegis-dashboard.json
echo "    https://$AWS_REGION.console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION#dashboards:name=$DASHBOARD_NAME"

echo "==> SSM config parameters (non-secret runtime config)"
put_param() {  # name value
  aws ssm put-parameter --name "$1" --value "$2" --type String \
    --overwrite --region "$AWS_REGION" >/dev/null \
    && echo "    $1 = $2" \
    || echo "    $1 skipped (ssm:PutParameter permission?)"
}
[ -n "${AEGIS_S3_BUCKET:-}" ] \
  && put_param /aegis-slam/config/s3_bucket "$AEGIS_S3_BUCKET"
[ -n "${INSTANCE_USD_PER_HOUR:-}" ] \
  && put_param /aegis-slam/config/instance_usd_per_hour "$INSTANCE_USD_PER_HOUR"
[ -n "${AEGIS_BEDROCK_MODEL:-}" ] \
  && put_param /aegis-slam/config/bedrock_model "$AEGIS_BEDROCK_MODEL"
# Secrets (Bedrock API keys, tokens) belong in SecureString parameters written
# by hand — never in this repo and never echoed here:
#   aws ssm put-parameter --name /aegis-slam/secret/<key> --value <v> --type SecureString

cat >> "$STATE_FILE" <<EOF
AWS_REGION="$AWS_REGION"
LOG_GROUP_NAME="$LOG_GROUP"
DASHBOARD_NAME="$DASHBOARD_NAME"
EOF

echo
echo "Metrics appear under CustomNamespaces > AEGISSlam after the first job"
echo "(Interventions, Rollbacks, DecisionsKept, CoverageTrusted, AteRmseM,"
echo " MeanConfidence, SecondsPerVideoMinute, and ComputeCostPerVideoMinute once"
echo " /aegis-slam/config/instance_usd_per_hour is set)."
