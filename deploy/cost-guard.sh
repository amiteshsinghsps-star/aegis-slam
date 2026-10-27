#!/usr/bin/env bash
# deploy/cost-guard.sh — AWS Budgets alarm that STOPS the demo instance.
#
# The instance bills a real account, so the guard is mandatory, not optional.
# Chain (all server-side AWS — no Lambda, no SSH, no third-party service):
#   AWS Budgets (monthly USD cap)
#     -> CloudWatch alarm on AWS/BudgetedServices Spend
#         -> SNS email notification (optional, NOTIFY_EMAIL=...)
#         -> EventBridge rule on alarm state = ALARM
#             -> SSM Automation AWS/StopInstancesForEC2 on our instance
#
# Every create is preceded by a get, so re-running is safe.
set -uo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
BUDGET_NAME="${BUDGET_NAME:-aegis-slam-demo}"
BUDGET_USD="${BUDGET_USD:-5}"                     # hard monthly cap, USD
INSTANCE_ID="${INSTANCE_ID:-}"                    # else read from deploy state
NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"
ACCT="$(aws sts get-caller-identity --query Account --output text)"
STATE_FILE="$(dirname "$0")/.deploy-state"

if [ -z "$INSTANCE_ID" ] && [ -f "$STATE_FILE" ]; then
  # shellcheck disable=SC1091
  . "$STATE_FILE"
fi
INSTANCE_ID="${INSTANCE_ID:-}"
if [ -z "$INSTANCE_ID" ]; then
  echo "cost-guard: no INSTANCE_ID — run deploy-graviton.sh first or export it" >&2
  exit 1
fi

STOP_ROLE="aegis-budget-stop-role"                # assumed by EventBridge
EXEC_ROLE="aegis-budget-ec2-stop-execution-role"  # assumed by SSM Automation
ALARM="aegis-${BUDGET_NAME}-spend"
RULE="aegis-${BUDGET_NAME}-stop"
AUTO_ARN="arn:aws:ssm:${AWS_REGION}::${ACCT}:automation-definition/AWS/StopInstancesForEC2:\$LATEST"

echo "==> Budget $BUDGET_NAME: \$${BUDGET_USD}/month ($AWS_REGION)"
if aws budgets describe-budget --account-id "$ACCT" --budget-name "$BUDGET_NAME" \
     --budget-type COST >/dev/null 2>&1; then
  echo "    budget already exists"
else
  MONTH="$(date +%Y-%m)"
  cat > /tmp/aegis-budget.json <<EOF
{
  "BudgetName": "$BUDGET_NAME",
  "BudgetType": "COST",
  "TimeUnit": "MONTHLY",
  "BudgetLimit": { "Amount": "$BUDGET_USD", "Unit": "USD" },
  "PlannedBudgetLimits": { "$MONTH": { "Amount": "$BUDGET_USD", "Unit": "USD" } },
  "CostTypes": {
    "IncludeTiming": true,
    "IncludeSubscriptionType": "ALL",
    "UseLatestPrices": true
  },
  "AdjustmentType": "ACTUAL"
}
EOF
  if aws budgets create-budget --account-id "$ACCT" \
       --budget file:///tmp/aegis-budget.json >/dev/null; then
    echo "    created"
  else
    echo "    FAILED — your credentials need budgets:CreateBudget (account root or a"
    echo "    Cost Explorer admin). Everything below still works as a standalone alarm."
  fi
  rm -f /tmp/aegis-budget.json
fi

echo "==> Roles for the stop action"
if ! aws iam get-role --role-name "$STOP_ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$STOP_ROLE" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"events.amazonaws.com"},
                   "Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam put-role-policy --role-name "$STOP_ROLE" \
    --policy-name aegis-budget-stop-perms --policy-document "{
      \"Version\":\"2012-10-17\",
      \"Statement\":[
        {\"Effect\":\"Allow\",\"Action\":[\"ssm:StartAutomationExecution\"],
         \"Resource\":\"$AUTO_ARN\"},
        {\"Effect\":\"Allow\",\"Action\":\"iam:PassRole\",
         \"Resource\":\"arn:aws:iam::$ACCT:role/$EXEC_ROLE\",
         \"Condition\":{\"StringEquals\":{\"iam:PassedToService\":\"ssm.amazonaws.com\"}}}]}"
  echo "    created $STOP_ROLE"
fi
if ! aws iam get-role --role-name "$EXEC_ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$EXEC_ROLE" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ssm.amazonaws.com"},
                   "Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$EXEC_ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonSSMAutomationRole
  # AmazonSSMAutomationRole alone cannot stop instances; grant exactly that on
  # exactly this instance instead of widening the managed policy.
  aws iam put-role-policy --role-name "$EXEC_ROLE" \
    --policy-name aegis-stop-only --policy-document "{
      \"Version\":\"2012-10-17\",
      \"Statement\":[{\"Effect\":\"Allow\",
                      \"Action\":[\"ec2:StopInstances\",\"ec2:DescribeInstanceStatus\"],
                      \"Resource\":\"arn:aws:ec2:*:*:instance/$INSTANCE_ID\"}]}"
  echo "    created $EXEC_ROLE"
fi

echo "==> CloudWatch alarm on the budget metric"
ALARM_ACTIONS=""
if [ -n "$NOTIFY_EMAIL" ]; then
  TOPIC_ARN="$(aws sns list-topics --query \
    "Topics[?ends_with(TopicArn,':${BUDGET_NAME}-alerts')].TopicArn" --output text)"
  if [ -z "$TOPIC_ARN" ] || [ "$TOPIC_ARN" = "None" ]; then
    TOPIC_ARN="$(aws sns create-topic --name "${BUDGET_NAME}-alerts" \
      --query TopicArn --output text)"
    aws sns subscribe --topic-arn "$TOPIC_ARN" --protocol email \
      --notification-endpoint "$NOTIFY_EMAIL" >/dev/null
    echo "    email topic created — confirm the subscription in your inbox"
  fi
  ALARM_ACTIONS="--alarm-actions $TOPIC_ARN"
fi
# shellcheck disable=SC2086
aws cloudwatch put-metric-alarm --alarm-name "$ALARM" \
  --namespace AWS/BudgetedServices --metric-name Spend --statistic Sum --period 3600 \
  --evaluation-periods 1 --comparison-operator GreaterThanOrEqualToThreshold \
  --threshold "$BUDGET_USD" \
  --dimensions "Name=Budget Name,Value=$BUDGET_NAME" \
               "Name=Resource Type,Value=Service" \
               "Name=Cost Type,Value=UnblendedCost" \
               "Name=Service Name,Value=AWSCharge" \
               "Name=Time Unit,Value=Monthly" \
               "Name=Contribution Type,Value=Total" \
  --treat-missing-data notBreaching $ALARM_ACTIONS >/dev/null
ALARM_ARN="$(aws cloudwatch describe-alarms --alarm-names "$ALARM" \
  --query 'MetricAlarms[0].AlarmArn' --output text)"
echo "    $ALARM at \$${BUDGET_USD}"

echo "==> EventBridge rule -> SSM StopInstancesForEC2"
aws events put-rule --name "$RULE" \
  --event-pattern "{
    \"source\":[\"aws.cloudwatch\"],
    \"detail-type\":[\"CloudWatch Alarm State Change\"],
    \"detail\":{\"alarmName\":[\"$ALARM\"],\"state\":{\"value\":[\"ALARM\"]}}
  }" >/dev/null
cat > /tmp/aegis-target.json <<EOF
{
  "Targets": [
    {
      "Id": "ssm-stop",
      "Arn": "$AUTO_ARN",
      "RoleArn": "arn:aws:iam::$ACCT:role/$STOP_ROLE",
      "Input": "{\"InstanceIds\":[\"$INSTANCE_ID\"],\"RoleArn\":\"arn:aws:iam::$ACCT:role/$EXEC_ROLE\"}"
    }
  ]
}
EOF
aws events put-targets --rule "$RULE" --targets file:///tmp/aegis-target.json >/dev/null
rm -f /tmp/aegis-target.json
echo "    rule $RULE armed on $INSTANCE_ID"

cat >> "$STATE_FILE" <<EOF
BUDGET_NAME="$BUDGET_NAME"
COST_ALARM_NAME="$ALARM"
COST_ALARM_ARN="$ALARM_ARN"
EVENT_RULE_NAME="$RULE"
STOP_ROLE_NAME="$STOP_ROLE"
EXEC_ROLE_NAME="$EXEC_ROLE"
EOF

echo
echo "Cost guard armed: at \$${BUDGET_USD} monthly spend the instance $INSTANCE_ID is"
echo "STOPPED (never deleted, so run artifacts survive) and you can restart it with"
echo "aws ec2 start-instances --instance-ids $INSTANCE_ID."
