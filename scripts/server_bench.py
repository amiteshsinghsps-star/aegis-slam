"""Benchmark the two AEGIS-SLAM servers with real measurements.

  * the FastAPI job service (`python -m aegis.server`): startup-to-ready, endpoint
    latency, one end-to-end supervised job, trace/result/frame endpoints and a
    small concurrency probe;
  * the stdio MCP server (`python -m aegis.mcp_server`): spawn-to-initialize and
    per-call JSON-RPC round trips.

Everything is measured on this machine, once. Results are written to
`eval/server_bench.md` plus a raw `eval/server_bench.json`; `docs/BENCHMARK.md`
quotes both. No AWS is required: `/api/presign-upload` is expected to return 501
without `AEGIS_S3_BUCKET`, and that is recorded as such.

Usage::

    python scripts/server_bench.py --frames 30
    python scripts/server_bench.py --port 8811 --out eval/server_bench.md
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def _http(method: str, url: str, data: Optional[bytes] = None,
          headers: Optional[dict] = None, timeout: float = 60.0) -> Tuple[int, bytes, float]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, body, (time.perf_counter() - t0) * 1000.0
    except urllib.error.HTTPError as e:
        body = e.read()
        return e.code, body, (time.perf_counter() - t0) * 1000.0


def _stats(xs: List[float]) -> dict:
    s = sorted(xs)
    return {"n": len(s), "median_ms": round(statistics.median(s), 2),
            "p95_ms": round(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], 2),
            "min_ms": round(s[0], 2), "max_ms": round(s[-1], 2)}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _mem_gb() -> Optional[float]:
    try:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = M()
        m.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.ullTotalPhys / 1e9, 1)
    except Exception:
        return None


def _ensure_dataset(rel_path: str, frames: int) -> str:
    abs_path = os.path.join(ROOT, rel_path)
    rgb = os.path.join(abs_path, "rgb")
    have = len([n for n in os.listdir(rgb)]) if os.path.isdir(rgb) else 0
    if have != frames:
        from aegis.synth import make_dataset
        print(f"[bench] generating {rel_path} ({frames} frames)")
        make_dataset(abs_path, n_frames=frames, seed=3, write_mp4=False)
    return rel_path


# --------------------------------------------------------------------------- #
#  HTTP service
# --------------------------------------------------------------------------- #
def _start_server(port: int, dataset: str, runs_dir: str) -> Tuple[subprocess.Popen, str]:
    os.makedirs(os.path.join(ROOT, runs_dir), exist_ok=True)
    log_path = os.path.join(ROOT, runs_dir, "server.log")
    env = dict(os.environ, AEGIS_HOST="127.0.0.1", AEGIS_PORT=str(port),
               AEGIS_DATASET=dataset, AEGIS_RUNS=runs_dir, PYTHONPATH=ROOT)
    log = open(log_path, "w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-m", "aegis.server",
                          "--host", "127.0.0.1", "--port", str(port)],
                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return p, log_path


def _wait_ready(base: str, timeout: float = 90.0) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if p_alive is not None and p_alive.poll() is not None:
            raise RuntimeError(f"server exited during startup (code {p_alive.returncode})")
        try:
            st, _, _ = _http("GET", base + "/healthz", timeout=1.0)
            if st == 200:
                return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(0.05)
    raise RuntimeError("server did not become ready in %.0fs" % timeout)


def _bench_get(base: str, path: str, n: int = 30) -> dict:
    try:
        _http("GET", base + path, timeout=5.0)          # warmup (caches, JIT-ish paths)
    except Exception:
        pass
    times, size, status = [], 0, 0
    for _ in range(n):
        status, body, ms = _http("GET", base + path)
        times.append(ms)
        size = len(body)
    return {"path": path, "status": status, "bytes": size, **_stats(times)}


def _run_job(base: str, dataset: str, mode: str = "supervised",
             timeout: float = 300.0) -> dict:
    form = urllib.parse.urlencode({"mode": mode, "dataset": dataset}).encode()
    st, body, post_ms = _http("POST", base + "/jobs", data=form,
                              headers={"Content-Type": "application/x-www-form-urlencoded"})
    if st != 200:
        raise RuntimeError(f"POST /jobs -> {st}: {body[:200]!r}")
    job_id = json.loads(body)["job_id"]

    polls, t0 = [], time.perf_counter()
    info = {}
    while time.perf_counter() - t0 < timeout:
        st, body, ms = _http("GET", base + f"/jobs/{job_id}")
        polls.append(ms)
        info = json.loads(body)
        if info.get("status") in ("done", "error"):
            break
        time.sleep(0.1)
    wall = time.perf_counter() - t0

    out = {"job_id": job_id, "mode": mode, "status": info.get("status"),
           "wall_seconds": round(wall, 2), "frames": info.get("total"),
           "polls": len(polls), "poll_stats": _stats(polls) if polls else None,
           "create_ms": round(post_ms, 2), "decisions": (info.get("trace_summary") or {}).get("num_decisions")}
    st, body, ms = _http("GET", base + f"/jobs/{job_id}/result", timeout=120.0)
    out["result_ms"] = round(ms, 2)
    out["result_bytes"] = len(body)
    if st == 200:
        res = json.loads(body)
        r = res.get("result") or {}
        out["ate_rmse"] = (r.get("ate") or {}).get("rmse")
        out["poses_estimated"] = r.get("poses_estimated")
        out["coverage_trusted"] = r.get("coverage_trusted")
        out["initialised"] = r.get("initialised")
    st, body, ms = _http("GET", base + f"/jobs/{job_id}/trace")
    out["trace_ms"] = round(ms, 2)
    out["trace_bytes"] = len(body)

    st, body, ms = _http("GET", base + f"/jobs/{job_id}/frame/10", timeout=60.0)
    out["frame_cold_ms"] = round(ms, 2)
    out["frame_bytes"] = len(body)
    _, _, ms = _http("GET", base + f"/jobs/{job_id}/frame/10", timeout=60.0)
    out["frame_cached_ms"] = round(ms, 2)
    return out


def _bench_concurrency(base: str, workers: int = 16, per_worker: int = 3) -> dict:
    def one(_):
        try:
            return _http("GET", base + "/healthz")[0] == 200
        except Exception:
            return False

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        oks = list(pool.map(one, range(workers * per_worker)))
    wall = time.perf_counter() - t0
    return {"workers": workers, "requests": len(oks), "ok": sum(oks),
            "wall_seconds": round(wall, 3),
            "req_per_second": round(len(oks) / wall, 1) if wall else None}


# --------------------------------------------------------------------------- #
#  MCP (stdio) server
# --------------------------------------------------------------------------- #
def _mcp_bench(samples: int = 10) -> dict:
    import cv2
    import numpy as np
    p = subprocess.Popen([sys.executable, "-m", "aegis.mcp_server"], cwd=ROOT,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)

    def rpc(obj: dict) -> Tuple[Optional[dict], float]:
        s = time.perf_counter()
        p.stdin.write(json.dumps(obj) + "\n")
        p.stdin.flush()
        line = p.stdout.readline()
        return (json.loads(line) if line else None), (time.perf_counter() - s) * 1000.0

    resp, spawn_ms = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    out = {"spawn_to_initialize_ms": round(spawn_ms, 2),
           "protocol": (resp or {}).get("result", {}).get("protocolVersion"),
           "calls": []}

    def timed(label: str, obj: dict, reps: int = 1) -> Optional[dict]:
        times, ok = [], True
        r = None
        for _ in range(reps):
            r, ms = rpc(obj)
            times.append(ms)
            ok = ok and isinstance(r, dict) and "result" in r
        out["calls"].append({"call": label, "reps": reps, "ok": ok,
                             **_stats(times)})
        return r

    r = timed("tools/list", {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, 5)
    if r:
        out["tools"] = len(r["result"]["tools"])

    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    ok_, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf.tobytes()).decode()
    timed("tools/call health_signals",
          {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
           "params": {"name": "health_signals", "arguments": {"image_b64": b64}}}, samples)
    timed("tools/call get_config",
          {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
           "params": {"name": "get_config", "arguments": {}}}, 5)
    timed("tools/call retune_thresholds",
          {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
           "params": {"name": "retune_thresholds",
                      "arguments": {"ransac_thresh": 2.0, "robust_estimator": "usac"}}}, 5)
    r = timed("tools/call rebootstrap (must be refused)",
              {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
               "params": {"name": "rebootstrap", "arguments": {}}}, 1)
    if r is not None:
        txt = (r.get("result", {}).get("content") or [{}])[0].get("text", "")
        out["rebootstrap_refused"] = (r.get("result", {}).get("isError") is True and
                                      "forbidden" in txt.lower())
    try:
        p.stdin.close()
    except Exception:
        pass
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
    return out


# --------------------------------------------------------------------------- #
#  Markdown
# --------------------------------------------------------------------------- #
def _table(rows: List[List[str]], head: List[str]) -> List[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _fmt_num(v, dash="-"):
    return dash if v is None else v


def write_md(path: str, payload: dict) -> None:
    e = payload["environment"]
    sv = payload["http_server"]
    job = payload.get("job") or {}
    mcp = payload["mcp_server"]
    cc = payload["concurrency"]

    lines = [
        "# AEGIS-SLAM - server benchmark",
        "",
        f"_Measured {e['date']} on one machine, one run. Both servers were started "
        f"by `scripts/server_bench.py` (FastAPI on `127.0.0.1:{e['http_port']}`, MCP "
        f"over stdio) and exercised with real requests. Reproduce with "
        f"`python scripts/server_bench.py`. Raw numbers: "
        f"[`eval/server_bench.json`](../eval/server_bench.json)._",
        "",
        "## Environment",
        "",
    ]
    env_rows = [
        ["Host", e["platform"]],
        ["CPU logical cores", e["cpu_count"]],
        ["RAM (GB)", _fmt_num(e["ram_gb"], "n/a")],
        ["Python", e["python"]],
        ["OpenCV", e["opencv"]],
        ["NumPy", e["numpy"]],
        ["FastAPI server", f"`python -m aegis.server` on 127.0.0.1:{e['http_port']}"],
        ["HTTP dataset", f"`{e['dataset']}` ({e['dataset_frames']} frames, synthetic seed 3)"],
        ["MCP server", "`python -m aegis.mcp_server` (stdio JSON-RPC)"],
        ["AWS", "not configured (CloudWatch/S3 no-op; presign expected 501)"],
    ]
    lines += _table(env_rows, ["Item", "Value"])
    lines += [
        "",
        "## FastAPI job service",
        "",
        f"Startup: **{sv['startup_seconds']:.2f} s** from process spawn to the first "
        f"`200 /healthz` (includes Python imports and FastAPI app creation). "
        f"`GET /healthz` returns OpenCV {e['opencv']}.",
        "",
    ]
    rows = []
    for ep in sv["endpoints"] + [sv["presign"]]:
        rows.append([f"`{ep['path'].split('?')[0]}`", ep["status"], ep["n"],
                     ep["median_ms"], ep["p95_ms"], ep["max_ms"], ep["bytes"]])
    lines += _table(rows, ["Endpoint", "Status", "N", "Median (ms)", "p95 (ms)",
                           "Max (ms)", "Response bytes"])
    lines += [
        "",
        "Notes: `/api/presign-upload` returns 501 without `AEGIS_S3_BUCKET` "
        "(expected, local mode); `/preloaded` is the judge read-only fallback and "
        "scans finished run directories. Off EC2, `/version` pays one 0.4 s IMDSv2 "
        "metadata timeout by design (it reports the instance type when it exists) — "
        "on Graviton that probe is local and the endpoint is not this slow.",
        "",
        "### End-to-end job (HTTP API)",
        "",
    ]
    if job:
        job_rows = [
            ["Mode", f"`{job['mode']}`"],
            ["Frames", _fmt_num(job.get("frames"))],
            ["Status", job.get("status")],
            ["Wall time (client-observed)", f"{job.get('wall_seconds')} s"],
            ["Throughput (incl. server overhead)", f"{(job.get('frames') or 0) / job['wall_seconds']:.2f} frames/s"
             if job.get("wall_seconds") else "n/a"],
            ["Status polls", f"{job.get('polls')} (median {job['poll_stats']['median_ms']:.1f} ms)"],
            ["Decisions", _fmt_num(job.get("decisions"))],
            ["Initialised / poses", f"{job.get('initialised')} / {_fmt_num(job.get('poses_estimated'))}"],
            ["Coverage (trusted)", _fmt_num(job.get("coverage_trusted"))],
            ["ATE RMSE (Sim(3))", (f"{job['ate_rmse']:.4f} m" if isinstance(job.get("ate_rmse"), (int, float))
                                    else "null (no GT)")],
            ["`GET /result`", f"{job['result_ms']} ms, {job['result_bytes']} bytes"],
            ["`GET /trace`", f"{job['trace_ms']} ms, {job['trace_bytes']} bytes"],
            ["`GET /frame/10` cold", f"{job['frame_cold_ms']} ms, {job['frame_bytes']} bytes"],
            ["`GET /frame/10` cached", f"{job['frame_cached_ms']} ms"],
        ]
        lines += _table(job_rows, ["Measurement", "Value"])
    else:
        lines.append("_Job did not complete; see the raw JSON._")
    lines += [
        "",
        "### Concurrency probe",
        "",
        f"`GET /healthz` from {cc['workers']} threads x {cc['requests'] // cc['workers']} "
        f"requests: **{cc['ok']}/{cc['requests']} ok** in {cc['wall_seconds']} s "
        f"(**{cc['req_per_second']} req/s**). Uvicorn runs the sync handlers in its "
        f"worker thread pool; this is a liveness probe, not a load test.",
        "",
        "## MCP tool server (stdio)",
        "",
        f"Spawn to first `initialize` response: **{mcp['spawn_to_initialize_ms']} ms** "
        f"(protocol {mcp.get('protocol')}); {mcp.get('tools')} tools advertised.",
        "",
    ]
    rows = [[f"`{c['call']}`", c["reps"], "yes" if c["ok"] else "**no**",
             c["median_ms"], c["p95_ms"], c["max_ms"]] for c in mcp["calls"]]
    lines += _table(rows, ["Call", "Reps", "OK", "Median (ms)", "p95 (ms)", "Max (ms)"])
    lines += [
        "",
        f"Destructive `rebootstrap` refused by the MCP surface: "
        f"**{'yes' if mcp.get('rebootstrap_refused') else 'no'}** (the destructive "
        f"tools stay behind the runtime supervisor + approval gate).",
        "",
        "## Reading the numbers",
        "",
        "* Latency is local-loopback HTTP on the machine above; it measures the "
        "service (route handling, JSON, file reads), **not** network or AWS.",
        "* The job wall time is dominated by the SLAM pipeline itself (local BA "
        "and descriptor matching), not by the HTTP layer - the same clip takes the "
        "same order of time through `aegis.run`. See the report's §6.2 for "
        "per-component timings and §5.7 for real-footage runtime.",
        "* First `/result`/`/frame` calls include disk reads and JSON serialisation; "
        "the frame endpoint renders on demand from the source on a cache miss.",
        "* Job wall time is sensitive to machine load (other processes, thermal "
        "state, filesystem cache). Re-run the benchmark before quoting a "
        "throughput figure; the endpoint latencies above are the stable part.",
        "* No significance is claimed from one run; timings are informational and "
        "x86-only (Graviton re-run is future work).",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
p_alive: Optional[subprocess.Popen] = None


def main(argv: Optional[List[str]] = None) -> int:
    global p_alive
    ap = argparse.ArgumentParser("AEGIS-SLAM server benchmark")
    ap.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--mode", default="supervised")
    ap.add_argument("--mcp-samples", type=int, default=10)
    ap.add_argument("--out", default="eval/server_bench.md")
    ap.add_argument("--json", default="eval/server_bench.json")
    a = ap.parse_args(argv)

    import cv2
    import numpy as np

    dataset = _ensure_dataset("Dataset/server_bench", a.frames)
    port = a.port or _free_port()
    base = f"http://127.0.0.1:{port}"

    print(f"[bench] starting FastAPI on {base}")
    p_alive, log_path = _start_server(port, dataset, "runs/server_bench")
    try:
        startup = _wait_ready(base)
        print(f"[bench] ready in {startup:.2f}s")

        endpoints = ["/healthz", "/version", "/api/config", "/", "/jobs", "/preloaded"]
        # /version also probes IMDS (0.4 s timeouts off EC2), so sample it less
        eps = [_bench_get(base, path, n=(10 if path == "/version" else 30))
               for path in endpoints]
        presign = _bench_get(base, "/api/presign-upload?filename=x.mp4", n=5)

        print(f"[bench] running one {a.mode} job over {a.frames} frames")
        job = _run_job(base, dataset, a.mode)
        print(f"[bench] job {job['status']} in {job['wall_seconds']}s")

        concurrency = _bench_concurrency(base)
    finally:
        p_alive.terminate()
        try:
            p_alive.wait(timeout=10)
        except Exception:
            p_alive.kill()
        p_alive = None

    print("[bench] benchmarking the MCP stdio server")
    mcp = _mcp_bench(a.mcp_samples)

    payload = {
        "environment": {
            "date": _dt.datetime.now().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "ram_gb": _mem_gb(),
            "http_port": port,
            "dataset": dataset,
            "dataset_frames": a.frames,
        },
        "http_server": {"startup_seconds": round(startup, 2),
                        "endpoints": eps, "presign": presign},
        "job": job,
        "concurrency": concurrency,
        "mcp_server": mcp,
        "server_log": log_path,
    }
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.json)) or ".", exist_ok=True)
    with open(os.path.join(ROOT, a.json), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_md(os.path.join(ROOT, a.out), payload)
    print(f"[bench] wrote {a.out} and {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
