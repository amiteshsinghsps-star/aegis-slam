#!/bin/bash
# deploy/user-data.sh — cloud-init for the Graviton instance.
# Installs Docker + Compose plugin, fetches the AEGIS-SLAM repo, and starts the
# service. Assumes the instance role grants S3 read (for the repo tarball) —
# replace REPO_SOURCE with your public git URL or an S3 tarball path.
set -euxo pipefail

dnf update -y
dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user
# AL2023 has no docker-compose-plugin RPM — install the Compose v2 binary
# (arm64/Graviton build) into the Docker CLI plugin path.
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL "https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-$(uname -m)" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Stop the box if idle: configure the budget alarm (deploy/README.md "Cost
# guard") — its action stops this instance. The alarm itself is created by
# deploy/cost-guard.sh, run from deploy-graviton.sh.
REPO_URL="__REPO_URL__"          # rendered by deploy-graviton.sh (cloud-init has no access to your shell env)
cd /opt
git clone "$REPO_URL" aegis-slam          # root owns /opt; fail loudly (set -e) if the clone fails
cd /opt/aegis-slam
mkdir -p runs

# ---------------------------------------------------------------------------
# Runtime config comes from SSM Parameter Store, never from the repo.
# IMDSv2 is enforced on the instance, so metadata needs a token.
# ---------------------------------------------------------------------------
TOKEN="$(curl -s -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
  http://169.254.169.254/latest/api/token || true)"
REGION="$(curl -s -H "X-aws-ec2-metadata-token: ${TOKEN:-}" \
  http://169.254.169.254/latest/meta-data/placement/region || true)"
get_param() {   # $1 = parameter name; empty when absent or not permitted
  aws ssm get-parameter --name "$1" --region "${REGION:-us-east-1}" \
    --query Parameter.Value --output text 2>/dev/null || true
}
BUCKET_P="$(get_param /aegis-slam/config/s3_bucket)"
PRICE_P="$(get_param /aegis-slam/config/instance_usd_per_hour)"
BEDROCK_P="$(get_param /aegis-slam/config/bedrock_model)"

cat > /opt/aegis-slam/deploy/.env <<EOF
AWS_REGION=${REGION:-us-east-1}
AWS_DEFAULT_REGION=${REGION:-us-east-1}
AEGIS_S3_BUCKET=${BUCKET_P}
AEGIS_INSTANCE_USD_PER_HOUR=${PRICE_P}
AEGIS_BEDROCK_MODEL=${BEDROCK_P}
AEGIS_JUDGE_MODE=${AEGIS_JUDGE_MODE:-0}
EOF
chmod 600 /opt/aegis-slam/deploy/.env     # may hold config values; ec2-user only

# Ship container stdout to CloudWatch Logs when the group exists (created by
# deploy/setup-cloudwatch.sh). Falls back to json-file so the demo still boots
# if the log group was never made.
if aws logs describe-log-groups --log-group-name-prefix /aegis-slam/service \
     --region "${REGION:-us-east-1}" --query 'count(logGroups)' --output text 2>/dev/null | grep -q '^1$'; then
  cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "awslogs",
  "log-opts": {
    "awslogs-region": "${REGION:-us-east-1}",
    "awslogs-group": "/aegis-slam/service",
    "awslogs-stream": "aegis-slam",
    "awslogs-datetime-format": "%Y-%m-%d %H:%M:%S"
  }
}
EOF
  systemctl restart docker
  echo "CloudWatch Logs driver enabled"
fi

docker compose -f deploy/docker-compose.yml up -d --build

echo "AEGIS-SLAM started (region=${REGION:-?} bucket=${BUCKET_P:-none} \
bedrock=${BEDROCK_P:-off} cost_metric=${PRICE_P:-off})" >> /var/log/aegis-slam-bootstrap.log
