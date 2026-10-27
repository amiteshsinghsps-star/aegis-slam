"""CloudWatch custom metrics (deploy/README §3) — published once per job.

Namespace ``AEGISSlam`` with dimensions ``JobId`` / ``InstanceType``.
Fully best-effort: no-op without boto3/credentials/region and never raises,
so observability can not break a run.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

log = logging.getLogger("aegis.metrics")

NAMESPACE = "AEGISSlam"


def _instance_type() -> Optional[str]:
    itype = os.environ.get("AEGIS_INSTANCE_TYPE")
    if itype:
        return itype
    try:
        from aegis.server import _ec2_instance_type
        return _ec2_instance_type()
    except Exception:
        return None


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def hourly_usd() -> Optional[float]:
    """On-demand $/hour for the running instance, from an env-supplied price
    table (kept honest: we never guess billing numbers in code)."""
    price = os.environ.get("AEGIS_INSTANCE_USD_PER_HOUR")
    if not price:
        return None
    try:
        return float(price)
    except ValueError:
        return None


def publish_job_metrics(result: dict, job_id: str, wall_seconds: float) -> dict:
    """Publish per-job metrics; returns the metric set that was sent (empty
    dict when CloudWatch is unavailable)."""
    if not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
        return {}
    ate = result.get("ate") or {}
    conf = result.get("confidence") or {}
    trace = result.get("trace_summary") or {}
    n_frames = int(result.get("frames") or 0)
    wall_seconds = float(result.get("wall_seconds") or wall_seconds)
    # playback fps of the source clip; recorded in result.json, 30 fallback
    src_fps = float(result.get("source_fps") or os.environ.get("AEGIS_VIDEO_FPS", "30"))
    video_minutes = n_frames / src_fps / 60.0 if n_frames else 0.0

    metrics = [
        ("Interventions", float(trace.get("num_decisions") or 0), "Count"),
        ("Rollbacks", float(trace.get("num_rolled_back")
                            or (trace.get("by_verdict") or {}).get("rolled_back") or 0), "Count"),
        ("DecisionsKept", float(trace.get("num_kept")
                                or (trace.get("by_verdict") or {}).get("improved") or 0), "Count"),
        ("CoverageTrusted", 100.0 * float(result.get("coverage_trusted") or 0.0), "Percent"),
        ("AteRmseM", float(ate.get("rmse")), "None") if _finite(ate.get("rmse")) else None,
        ("MeanConfidence", float(conf.get("mean") or 0.0), "None"),
    ]
    metrics = [m for m in metrics if m is not None and _finite(m[1])]
    if wall_seconds > 0 and video_minutes > 0:
        sec_per_video_min = wall_seconds / video_minutes
        metrics.append(("SecondsPerVideoMinute", round(sec_per_video_min, 3), "Seconds"))
        price = hourly_usd()
        if price is not None:
            metrics.append(("ComputeCostPerVideoMinute",
                            round(price * sec_per_video_min / 3600.0, 6), "USD"))

    dims = {"JobId": job_id}
    itype = _instance_type()
    if itype:
        dims["InstanceType"] = itype

    try:
        import boto3
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[
                {
                    "MetricName": name,
                    "Dimensions": [{"Name": k, "Value": v} for k, v in dims.items()],
                    "Value": value,
                    "Unit": unit,
                }
                for name, value, unit in metrics
            ],
        )
        # structured JSON line for CloudWatch Logs collection
        log.info("METRICS %s", {"job_id": job_id, "wall_s": round(wall_seconds, 2),
                                **{n: v for n, v, _ in metrics}})
        return {n: v for n, v, _ in metrics}
    except Exception as e:  # observability must never fail the run
        log.debug("CloudWatch metrics skipped: %s", e)
        return {}
