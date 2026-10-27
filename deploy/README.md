# AEGIS-SLAM — AWS Graviton deployment

One-box deployment of the self-healing SLAM service on an AWS **Graviton**
(ARM64) instance. Runs the FastAPI service + job worker + OpenCV-5 SLAM core +
static web UI in a single Docker container.

> Simplest-viable deployment per the PRD: one Graviton instance, Docker
> Compose, presigned/local uploads, CloudWatch logs. Queueing and autoscaling
> are documented as future work and deliberately not built for the MVP.

## 0. Prerequisites

- AWS CLI v2 configured (`aws sts get-caller-identity` works).
- `jq`, `curl`.
- A region with Graviton4 (`c8g.*`) capacity and (optionally) Bedrock enabled.

## 1. Local run first (validates the image on your machine)

```bash
docker build -t aegis-slam -f deploy/Dockerfile .
docker run --rm -p 8000:8000 -v "$PWD:/app" aegis-slam
# open http://localhost:8000   (UI)   /version  (OpenCV build proof)
```

On Apple Silicon / ARM this runs the same arch as Graviton.

## 2. Deploy to Graviton

```bash
export AWS_REGION=us-east-1
export INSTANCE_TYPE=c8g.large          # Graviton4
export REPO_URL=https://github.com/YOUR_ORG/aegis-slam.git   # your public repo
export INSTANCE_USD_PER_HOUR=0.085      # your region's c8g.large on-demand price
                                        # (Cost Explorer page) — feeds the cost metric
export NOTIFY_EMAIL=you@example.com     # optional: budget email as well as auto-stop
bash deploy/deploy-graviton.sh
```

The script (all of it re-runnable):

1. Creates an **S3 bucket** `aegis-slam-uploads-<account>` with a
   **1-day lifecycle** (auto-delete user video) and full public-access block.
2. Creates a **least-privilege IAM instance role** (`deploy/iam/*`) scoped to
   that bucket, CloudWatch Logs/Metrics (`AEGISSlam` namespace), SSM parameter
   read under `/aegis-slam/*`, and optional Bedrock `InvokeModel`; plus the AWS
   managed **`AmazonSSMManagedInstanceCore`** policy, without which "no SSH"
   would just mean "no access". The Bedrock statement pins one foundation model
   (`anthropic.claude-3-haiku-…`); if that id is unavailable in your region, set
   `AEGIS_BEDROCK_MODEL` (compose) or `--llm-model` (CLI) **and update the
   matching `Resource` ARN in `deploy/iam/instance-policy.json`** — do not widen
   it to `foundation-model/*`. The planner is optional; rules are the
   reproducible floor without it.
3. Opens **port 8000 to your IP only** and **fails closed** if your IP cannot be
   determined (never `0.0.0.0/0`). Admin access is SSM Session Manager only.
4. Launches a **c8g** instance with **IMDSv2 enforced**
   (`--metadata-options HttpTokens=required`) running `deploy/user-data.sh`,
   which installs Docker + Compose, clones the repo, pulls runtime config from
   SSM Parameter Store into `deploy/.env` (never committed), enables the
   `awslogs` driver when the log group exists, and starts the stack.
5. Runs `deploy/setup-cloudwatch.sh`: log group `/aegis-slam/service` with
   **14-day retention**, the **`aegis-slam` dashboard**, and SSM config
   parameters (`s3_bucket`, `instance_usd_per_hour`, `bedrock_model`).
6. Runs `deploy/cost-guard.sh` (skip with `COST_GUARD=0`): **AWS Budgets**
   monthly cap → **CloudWatch alarm** on `AWS/BudgetedServices/Spend` →
   **EventBridge** rule → **SSM Automation `AWS/StopInstancesForEC2`** on that
   instance, with an optional SNS email notification. Roles are scoped to the
   single instance id; nothing in the chain can delete data.

It prints the instance id, public DNS, the SSM session command and the teardown
command, and records them in `deploy/.deploy-state` (git-ignored) so the other
scripts need no arguments.

Confirm the OpenCV-5-on-Graviton claim from the live host:

```bash
curl -s http://<dns>:8000/version | jq '{opencv_version, is_arm_graviton,
      ec2_instance_type, aws_region, cloud}'
```

`cloud` reports which integrations are actually live (`imdsv2_only`,
`s3_presign_enabled`, `bedrock_planner`, `cost_metric_usd_per_hour`,
`judge_mode`) — if a box is unchecked there, the docs do not claim it works.

### Connect without SSH (SSM)

```bash
sudo amazon-ssm-agent --version 2>/dev/null || true   # session-manager-plugin on your laptop
aws ssm start-session --target <INSTANCE_ID>
docker logs -f aegis-slam
```

## 3. Observability (CloudWatch)

The container's `HEALTHCHECK` pings `/healthz`. The service publishes custom
metrics itself at job completion (`aegis/metrics.py`, via the instance role's
`cloudwatch:PutMetricData`) — no CloudWatch agent required:

- `Interventions` (agent decisions per job), `Rollbacks`, `DecisionsKept`
- `CoverageTrusted`, `AteRmseM`, `MeanConfidence`
- `SecondsPerVideoMinute`, and `ComputeCostPerVideoMinute` when
  `AEGIS_INSTANCE_USD_PER_HOUR` is set (we never guess billing numbers)

Metrics land in namespace `AEGISSlam` with dimensions `JobId` and
`InstanceType`, and one structured `METRICS {json}` line per job goes to stdout,
which the `awslogs` driver ships to `/aegis-slam/service`. Publishing is
best-effort: absent region/credentials is a silent no-op, so observability can
never fail a run. These feed the honest cost/latency table in the report.

```bash
aws cloudwatch get-metric-statistics --namespace AEGISSlam --metric-name Interventions \
  --dimensions Name=JobId,Value=<job> --start-time 2026-10-26T00:00:00Z \
  --end-time 2026-10-27T00:00:00Z --period 300 --statistics Sum --region $AWS_REGION
```

### 3.1 Judge access mode

By default the security group opens `:8000` **only to the owner's IP**, so
judges cannot reach the instance. Two supported ways to give access:

1. **Judge mode on the same instance.** Temporarily open `:8000` to
   `0.0.0.0/0`, then start the service with the demo lock-down on:
   ```bash
   AEGIS_JUDGE_MODE=1 AEGIS_JUDGE_TOKEN=<shared-secret> docker compose up -d
   ```
   Judge mode: uploads disabled, arbitrary dataset paths rejected, at most
   `AEGIS_JUDGE_MAX_RUNNING` (default 2) concurrent jobs (429 otherwise), and
   `POST /jobs` requires the shared token. **Close the port again after
   judging.**
2. **Recorded fallback (always works, even with the instance stopped).**
   `python -m scripts.preload_demo` records three supervised demo runs —
   clean / motion-blur / frame-drop — into `docs/demo/`, and `GET /preloaded`
   / `GET /preloaded/{run}` serve them read-only; the UI preset buttons render
   them without starting a job.

Either way, state the access method (and teardown date) in the Devpost text.

## 4. Presigned S3 uploads (optional)

`AEGIS_S3_BUCKET` is seeded into SSM by `setup-cloudwatch.sh` and handed to the
container by `user-data.sh`; the instance role from step 2 already has
`PutObject`/`GetObject`/`DeleteObject` on that one bucket prefix.
`GET /api/presign-upload?filename=<name>` returns a 15-minute presigned `PUT`
URL plus the key; the browser uploads directly to S3 and the worker pulls from
there, so video bytes never transit the API process. Public access is blocked
and the 1-day lifecycle expires the objects. (The streamed upload path in
`POST /jobs` — capped at 200 MB **while reading**, not via `Content-Length` — is
used when no bucket is configured, and is disabled entirely in judge mode.)

## 5. Secrets and runtime config

Never baked into the image, never in the repo:

| Kind | Where | Who can read it |
|---|---|---|
| Non-secret config | SSM `/aegis-slam/config/{s3_bucket,instance_usd_per_hour,bedrock_model}` (String) | instance role (`ssm:GetParameter*`, that path only) |
| Secrets/tokens | SSM `/aegis-slam/secret/*` (**SecureString**, written by hand) | instance role; add `kms:Decrypt` on the key if you use a CMK |
| Inside the container | `deploy/.env`, written by `user-data.sh`, `chmod 600`, git-ignored | root/ec2-user on the box |

```bash
aws ssm put-parameter --name /aegis-slam/secret/<key> --value <v> \
  --type SecureString --region $AWS_REGION          # never echo the value
```

The app reads environment variables only; nothing in `aegis/` knows the
difference between SSM-supplied and hand-exported config.

## 6. Cost guard / responsible operation

`deploy/cost-guard.sh` arms it automatically during deploy (default monthly cap
`BUDGET_USD=5`; override before running). Chain: Budgets → CloudWatch alarm on
the budget `Spend` metric → EventBridge → SSM Automation `AWS/StopInstancesForEC2`.
The instance is **stopped, never deleted**, so run artifacts and the trace
survive; restart with `aws ec2 start-instances`.

- The two helper roles are least-privilege: `PassRole` is conditioned on
  `ssm.amazonaws.com` and the stop permission is scoped to the single
  `instance/<id>` ARN.
- Container Compose caps CPU (4) and memory (8 GB) so one runaway job cannot
  exceed the instance.
- User uploads auto-delete in 1 day; we do not train on user data; the UI warns
  against uploading footage you lack rights to; untrusted trajectory segments
  are never presented as reliable.
- Report cost from **real billing** (Cost Explorer / Budgets), not estimates, and
  only publish `ComputeCostPerVideoMinute` when you supplied the real on-demand
  hourly price.

## 7. Teardown (do this after judging to avoid cost)

```bash
CONFIRM=yes bash deploy/teardown.sh                      # stop the instance + close :8000
CONFIRM=yes DESTROY=1 bash deploy/teardown.sh            # terminate it instead
CONFIRM=yes DESTROY=1 REMOVE_IAM=1 BUCKET=delete bash deploy/teardown.sh   # full cleanup
```

Same three as make targets: `make teardown CONFIRM=yes`, `make deploy`,
`make deploy-cloudwatch`, `make deploy-guard BUDGET_USD=5`.

`teardown.sh` reads `deploy/.deploy-state`, enumerates the SG and revokes
**every** `:8000` ingress rule (so a changed client IP cannot leave a stale hole)
even when you only stop the box, and removes the alarm, EventBridge rule,
dashboard, budget and helper roles under `REMOVE_IAM=1`. The instance role itself
is left in place deliberately (instance profiles make IAM deletion painful) —
delete it in the console once the instance is gone. Manual equivalent:

```bash
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
aws s3 rb s3://aegis-slam-uploads-<account> --force
aws iam detach-role-policy --role-name aegis-slam-role \
  --policy-arn arn:aws:iam::<account>:policy/aegis-slam-role-policy
```

## Files

| Path | Purpose |
|---|---|
| `Dockerfile` | Multi-arch (amd64/arm64), headless OpenCV-5 image; optional deps best-effort |
| `requirements-server.txt` | Pinned lean runtime deps (incl. boto3 for the AWS paths) |
| `docker-compose.yml` | Single service, CPU/RAM caps, `.env` wiring (optional file) |
| `deploy-graviton.sh` | S3 + IAM + SG + EC2 (IMDSv2) + observability + cost guard |
| `user-data.sh` | Bootstrap: Docker/Compose, SSM config → `.env`, awslogs driver, up |
| `setup-cloudwatch.sh` | Log group + retention, dashboard, SSM config parameters |
| `cost-guard.sh` | Budgets → alarm → EventBridge → SSM stop (+ optional email) |
| `teardown.sh` | Stop/terminate, close the port, remove the guard chain |
| `cloudwatch/dashboard.json` | Dashboard body (`__REGION__`/`__ACCOUNT__` templated) |
| `iam/*.json` | Least-privilege instance role policy + trust |
| `s3/lifecycle.json` | 1-day retention + incomplete-MPU cleanup |
| `.deploy-state` | Generated by the deploy script; git-ignored, read by the others |
