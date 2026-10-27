#!/usr/bin/env bash
# deploy/teardown.sh — stop the billing. Defaults to STOPPING the instance
# (artifacts + EBS survive); pass DESTROY=1 to terminate it, and REMOVE_IAM=1 to
# delete the roles/policies/dashboard/budget as well.
#
# Nothing here is reversible by accident: it prints the plan and requires
# CONFIRM=yes.
set -uo pipefail

STATE="$(dirname "$0")/.deploy-state"
[ -f "$STATE" ] || { echo "no $STATE — was deploy-graviton.sh ever run here?"; exit 1; }
# shellcheck disable=SC1090
. "$STATE"

AWS_REGION="${AWS_REGION:-us-east-1}"
DESTROY="${DESTROY:-0}"
REMOVE_IAM="${REMOVE_IAM:-0}"
CONFIRM="${CONFIRM:-no}"

ACCT="$(aws sts get-caller-identity --query Account --output text)"
echo "Account $ACCT · region $AWS_REGION · instance ${INSTANCE_ID:-?} (${INSTANCE_TYPE:-?})"
echo "Plan:"
if [ "$DESTROY" = "1" ]; then echo "  TERMINATE $INSTANCE_ID (EBS deleted unless encrypted-volume retain set)";
else                          echo "  STOP      $INSTANCE_ID (billing for the volume continues at ~cents/GB-month)"; fi
echo "  close :8000 in $SG_ID"
[ "$REMOVE_IAM" = "1" ] && echo "  delete alarm/rule/dashboard/budget + detach and delete roles and the $ROLE_NAME policy"
[ "$REMOVE_IAM" = "1" ] && echo "  LEAVE the S3 bucket alone (1-day lifecycle already expires uploads) unless BUCKET=delete"
if [ "$CONFIRM" != "yes" ]; then
  echo; echo "Refusing to act. Re-run with CONFIRM=yes (add DESTROY=1 / REMOVE_IAM=1 as wanted)."; exit 1
fi

if [ "$DESTROY" = "1" ]; then
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" >/dev/null
  echo "terminating"
else
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" >/dev/null
  echo "stopping"
fi

# Fail closed on the network too, whether or not the box survives. Revoke every
# :8000 rule (not just the current IP's) so a changed IP cannot leave a hole.
if [ -n "${SG_ID:-}" ]; then
  for cidr in $(aws ec2 describe-security-groups --group-ids "$SG_ID" --region "$AWS_REGION" \
      --query 'SecurityGroups[0].IpPermissions[?ToPort==`8000`].IpRanges[].CidrIp' \
      --output text 2>/dev/null); do
    aws ec2 revoke-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8000 \
      --cidr "$cidr" --region "$AWS_REGION" >/dev/null 2>&1 \
      && echo "  revoked $cidr -> :8000" || echo "  could not revoke $cidr — do it in the console"
  done
fi

if [ "$REMOVE_IAM" = "1" ]; then
  for n in "${COST_ALARM_NAME:-}" "${EVENT_RULE_NAME:-}"; do
    [ -n "$n" ] && aws events remove-targets-by-id --rule "$n" --ids ssm-stop >/dev/null 2>&1 || true
    [ -n "$n" ] && aws events delete-rule --name "$n" >/dev/null 2>&1 || true
  done
  [ -n "${COST_ALARM_NAME:-}" ] && \
    aws cloudwatch delete-alarms --alarm-names "$COST_ALARM_NAME" >/dev/null 2>&1 || true
  [ -n "${DASHBOARD_NAME:-}" ] && \
    aws cloudwatch delete-dashboards --dashboard-names "$DASHBOARD_NAME" >/dev/null 2>&1 || true
  [ -n "${BUDGET_NAME:-}" ] && \
    aws budgets delete-budget --account-id "$ACCT" --budget-name "$BUDGET_NAME" \
      --budget-type COST >/dev/null 2>&1 || true
  for r in "${STOP_ROLE_NAME:-}" "${EXEC_ROLE_NAME:-}"; do
    [ -n "$r" ] || continue
    aws iam detach-role-policy --role-name "$r" \
      --policy-arn arn:aws:iam::aws:policy/service-role/AmazonSSMAutomationRole >/dev/null 2>&1 || true
    aws iam delete-role-policy --role-name "$r" --policy-name aegis-budget-stop-perms >/dev/null 2>&1 || true
    aws iam delete-role-policy --role-name "$r" --policy-name aegis-stop-only >/dev/null 2>&1 || true
    aws iam delete-role --role-name "$r" >/dev/null 2>&1 || echo "  role $r: delete it in the console (provenance)"
  done
  if [ -n "${ROLE_NAME:-}" ]; then
    aws iam detach-role-policy --role-name "$ROLE_NAME" \
      --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore >/dev/null 2>&1 || true
    aws iam detach-role-policy --role-name "$ROLE_NAME" \
      --policy-arn "arn:aws:iam::$ACCT:policy/$ROLE_NAME-policy" >/dev/null 2>&1 || true
    echo "  instance role $ROLE_NAME left in place (instance profiles make deletion painful);"
    echo "  delete it in IAM once the instance is gone."
  fi
fi

if [ "${BUCKET:-}" = "delete" ]; then
  aws s3 rb --force "s3://${BUCKET_NAME:-aegis-slam-uploads-$ACCT}" >/dev/null 2>&1 \
    && echo "  bucket removed" || echo "  bucket not removed (objects/permissions) — lifecycle expires uploads anyway"
fi
echo "Done. Verify: aws ec2 describe-instances --instance-ids ${INSTANCE_ID:-} --query 'Reservations[0].Instances[0].State'"
