"""AEGIS-SLAM: a self-healing visual SLAM service on OpenCV 5 + AWS Graviton.

The SLAM perception core (tracking, mapping, bundle adjustment) is reused from
the Apache-2.0 `opencv-SimpleSLAM` project under ``slam.core``. Everything in
this package is new, original work: health-signal extraction, the agent
supervisor and its OpenCV-5 tool registry, the structured decision trace, the
human approval gate, the fault-injection benchmark/evaluation harness, the
FastAPI job service and the web UI.

See NOTICE and the technical report for the reused-versus-new boundary.
"""

__all__ = ["version"]
__version__ = "1.0.0"


def version() -> str:
    return __version__
