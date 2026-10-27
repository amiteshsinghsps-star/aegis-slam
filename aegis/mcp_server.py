"""MCP tool server for the AEGIS-SLAM registry (PRD v2 U8).

Lets ANY MCP-capable agent (Claude Code, Kiro, a small script) drive the same
bounded tools the runtime supervisor uses, and read OpenCV-5 health signals —
the "agent may invoke OpenCV-5 through APIs, tools, or MCP" path from the rules.

Dependency-free newline-delimited JSON-RPC 2.0 over stdio (no `mcp` package
needed), so it runs anywhere the pipeline does:

    python -m aegis.mcp_server
    # then pipe JSON-RPC lines, or point an MCP client at this command.

Tools exposed (read + the two safe, reversible writes):
    health_signals   read blur/brightness/contrast/texture for a frame image
    list_tools       the bounded registry + current config
    get_config / set_config
    apply_preprocessing / retune_thresholds / set_feature_backend
Destructive tools (rebootstrap, request_recapture, mark_segment_untrusted) are
intentionally NOT exposed to an external agent — they stay under the runtime
supervisor's guard + human-approval path.
"""
from __future__ import annotations

import json
import sys
from typing import Optional

import numpy as np

from aegis.tools import RuntimeConfig, TOOLS, invoke
from aegis.health import image_quality

_CFG = RuntimeConfig()

_EXPOSED = {
    "set_feature_backend": {
        "backend": {"type": "string", "enum": ["orb", "sift", "aliked"]}},
    "apply_preprocessing": {
        "mode": {"type": "string", "enum": ["none", "clahe", "denoise", "sharpen"]}},
    "retune_thresholds": {
        "ransac_thresh": {"type": "number"}, "pnp_min_inliers": {"type": "integer"},
        "max_features": {"type": "integer"},
        "robust_estimator": {"type": "string", "enum": ["ransac", "usac", "usac_magsac"]}},
}

_TOOL_DEFS = [
    {"name": "health_signals",
     "description": "OpenCV-5 per-frame health signals for a base64 PNG/JPEG image.",
     "inputSchema": {"type": "object", "properties": {
         "image_b64": {"type": "string"}}, "required": ["image_b64"]}},
    {"name": "list_tools", "description": "Bounded tool registry + current config.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "get_config", "description": "Current runtime config snapshot.",
     "inputSchema": {"type": "object", "properties": {}}},
] + [
    {"name": n, "description": f"AEGIS action {n} (guarded, reversible).",
     "inputSchema": {"type": "object", "properties": props}}
    for n, props in _EXPOSED.items()
]


def _call(name: str, args: dict) -> dict:
    if name == "health_signals":
        import base64, cv2
        buf = np.frombuffer(base64.b64decode(args["image_b64"]), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return {"isError": True, "content": [{"type": "text", "text": "undecodable image"}]}
        q = image_quality(img)
        return {"content": [{"type": "text", "text": json.dumps(q)}]}
    if name == "list_tools":
        return {"content": [{"type": "text", "text": json.dumps(
            {"tools": sorted(TOOLS), "exposed_to_agent": sorted(_EXPOSED),
             "config": _CFG.snapshot()})}]}
    if name == "get_config":
        return {"content": [{"type": "text", "text": json.dumps(_CFG.snapshot())}]}
    if name in _EXPOSED:
        res = invoke(name, _CFG, args, frame=int(args.get("frame", 0)))
        return {"content": [{"type": "text",
                             "text": json.dumps({"ok": res.ok, "changed": res.changed,
                                                 "note": res.note, "config": _CFG.snapshot()})}]}
    return {"isError": True, "content": [{"type": "text", "text": f"unknown/forbidden tool {name}"}]}


def _handle(req: dict) -> Optional[dict]:
    rid = req.get("id")
    method = req.get("method")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "aegis-slam", "version": "2.0"}}}
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": _TOOL_DEFS}}
    if method == "tools/call":
        p = req.get("params", {})
        out = _call(p.get("name", ""), p.get("arguments", {}) or {})
        return {"jsonrpc": "2.0", "id": rid, "result": out}
    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = _handle(req)
        except Exception as e:  # never die on one bad line
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": str(e)}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
