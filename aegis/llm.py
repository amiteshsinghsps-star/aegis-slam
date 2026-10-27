"""Optional LLM planner on Amazon Bedrock (FR-11).

The rule engine always runs and is the source of truth for reproducibility.
This planner is consulted only for ambiguous diagnoses, and its output MUST
validate against the tool schema or it is discarded (falling back to rules).
If Bedrock is not configured, :class:`BedrockPlanner` is inert and returns None,
so the system is fully functional without any network dependency.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from aegis.tools import TOOL_SCHEMA

log = logging.getLogger("aegis.llm")

_VALID_TOOLS = set(TOOL_SCHEMA.keys())


def _validate(choice: dict) -> bool:
    if not isinstance(choice, dict):
        return False
    tool = choice.get("tool")
    if tool not in _VALID_TOOLS:
        return False
    args = choice.get("args", {}) or {}
    schema = TOOL_SCHEMA.get(tool, {})
    for k, allowed in schema.items():
        if isinstance(allowed, list) and k in args and args[k] not in allowed:
            return False
    bounds = {"ransac_thresh": (0.5, 8.0), "pnp_min_inliers": (8, 200), "max_features": (500, 8000)}
    for k, (lo, hi) in bounds.items():
        if k in args:
            try:
                if not (lo <= float(args[k]) <= hi):
                    return False
            except (TypeError, ValueError):
                return False
    if tool == "rebootstrap":
        return False          # destructive: only the persistence-gated rule path may re-bootstrap
    return True


SYSTEM_PROMPT = (
    "You are the planner inside a self-healing visual SLAM system. Given image "
    "and tracking health evidence you MUST choose exactly one action from this "
    "JSON tool schema and reply ONLY with a JSON object of the form "
    '{"tool": <name>, "args": {...}}. '
    "Tools: set_feature_backend(backend in orb|sift|aliked), "
    "apply_preprocessing(mode in none|clahe|denoise|sharpen), "
    "retune_thresholds(ransac_thresh,pnp_min_inliers,max_features,robust_estimator in ransac|usac), "
    "rebootstrap, mark_segment_untrusted(start,end), request_recapture(reason), "
    "abstain(reason). "
    "Prefer the least invasive action that plausibly restores tracking; when the "
    "evidence does not support any intervention, abstain instead of guessing."
)


class BedrockPlanner:
    def __init__(self, model_id: Optional[str] = None, region: Optional[str] = None,
                 enabled: bool = False):
        self.enabled = enabled and bool(model_id)
        self.model_id = model_id
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3  # imported lazily; absent => planner stays inert
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def propose(self, window_stats, fallback_tool: str, fallback_args: dict, cfg) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            client = self._get_client()
            evidence = {
                "window": window_stats.as_evidence(),
                "current_config": cfg.snapshot(),
                "rule_engine_suggestion": {"tool": fallback_tool, "args": fallback_args},
            }
            body = {
                "modelId": self.model_id,
                "system": [{"text": SYSTEM_PROMPT}],
                "messages": [{"role": "user", "content": [{"text": json.dumps(evidence)}]}],
                "inferenceConfig": {"maxTokens": 512, "temperature": 0.0},
            }
            resp = client.converse(**body)
            text = "".join(c["text"] for c in resp["output"]["message"]["content"] if "text" in c)
            m = re.search(r"\{.*\}", text, re.S)          # tolerate ```json fences / prose
            choice = json.loads(m.group(0)) if m else None
            if _validate(choice):
                return choice
            log.info("Bedrock proposal invalid, using rules: %r", choice)
        except Exception as e:  # any failure (no boto3, no creds, network) → inert
            log.debug("Bedrock planner unavailable (%s); using rule engine", e)
        return None
