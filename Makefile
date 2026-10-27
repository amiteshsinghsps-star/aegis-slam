.PHONY: install synth test demo bench eval stats trust sweep real serve serve-judge preload figure deploy deploy-guard deploy-cloudwatch teardown clean

# Override on the command line, e.g. `make teardown CONFIRM=yes DESTROY=1`.
BUDGET_USD ?=
CONFIRM ?=
DESTROY ?=
REMOVE_IAM ?=

install:
	pip install -r requirements.txt

synth:
	python -m aegis.synth --out Dataset/synthetic --frames 200 --no_mp4

test:
	pytest tests -q

demo:
	bash scripts/demo.sh

bench:
	python -m aegis.bench --frames 120 --reps 3 --out eval/bench.json

# --seeds S evaluates every condition on distinct synthetic scenes S, so the
# paired delta and its bootstrap CI are real. Dev seeds are 0-4; the locked
# test seeds (10-17) are run once, after docs/PREREG.md is frozen.
SEEDS ?= 0,1,2,3,4
eval:
	python -m aegis.evaluate --data Dataset/synthetic --frames 90 \
	  --conditions baseline_classical,supervised,supervised_multi,ablation \
	  --seeds $(SEEDS) --out eval/dev --jobs 6

# Paired ΔATE with bootstrap CI + Wilcoxon, and trust/risk-coverage metrics.
stats:
	python scripts/stats.py --results eval/dev/results.json \
	  --out eval/dev/stats.json --markdown eval/dev/stats.md

trust:
	python scripts/trust.py --eval eval/dev \
	  --out eval/dev/trust.json --markdown eval/dev/trust.md

preload:
	python -m scripts.preload_demo --frames 90

# U4: baseline vs supervised across fault levels (paired per scene).
sweep:
	python scripts/severity_sweep.py --data Dataset/synthetic --frames 90 --runs 2 \
	  --severities 0.35,0.7,1.0 --scenarios motion_blur,low_light,texture_loss \
	  --conditions baseline_classical,supervised --out eval/severity --jobs 4

# Real footage (TUM RGB-D): TUM is a space-separated list of dirs prepared by
# `python -m aegis.datasets`, e.g. make real TUM="C:/tum/prepared/fr1_xyz C:/tum/prepared/fr1_desk"
TUM ?=
REAL_FRAMES ?= 90
real:
	python scripts/real_bench.py $(foreach d,$(TUM),--data $(d)) --frames $(REAL_FRAMES) \
	  --severity 0.6 --conditions baseline_classical,supervised,supervised_multi \
	  --jobs 2 --out eval/real

figure:
	python scripts/trace_figure.py eval/dev/frame_drop__supervised__s0 \
	  --out docs/diagrams/trace_walkthrough.png

serve:
	python -m aegis.server --port 8000

# Read-only judge demo (uploads disabled, built-in dataset only, concurrency cap)
serve-judge:
	AEGIS_JUDGE_MODE=1 AEGIS_JUDGE_TOKEN=$(TOKEN) python -m aegis.server --port 8000

# One command: S3 + IAM + SG + IMDSv2 EC2 + CloudWatch + budget stop-guard.
deploy:
	bash deploy/deploy-graviton.sh

# Re-run either half standalone (both are idempotent) if deploy skipped them.
deploy-cloudwatch:
	bash deploy/setup-cloudwatch.sh

deploy-guard:
	BUDGET_USD=$(BUDGET_USD) bash deploy/cost-guard.sh

# Stops (not terminates) the instance and closes :8000. DESTROY=1 / REMOVE_IAM=1
# widen it; it always prints the plan and refuses without CONFIRM=yes.
teardown:
	CONFIRM=$(CONFIRM) DESTROY=$(DESTROY) REMOVE_IAM=$(REMOVE_IAM) bash deploy/teardown.sh

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache runs/ci_smoke
