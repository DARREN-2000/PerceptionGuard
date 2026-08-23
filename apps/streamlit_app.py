"""PerceptionGuard inspection UI.

NOTE: Streamlit is not installed in the development sandbox, so this file has
never been executed. It is written against the stable public API of the
pipeline (see tests/test_pipeline_integration.py for the same calls exercised
headlessly, which DO run). Treat it as reviewed-but-unrun code.

Run with:
    pip install streamlit
    PYTHONPATH=src streamlit run apps/streamlit_app.py

The UI is deliberately thin. It renders state the pipeline already computes and
adds no analysis of its own -- the perception and reliability system is the
project; this is a window onto it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perceptionguard.data.degradations import DEGRADATIONS, apply_degradation
from perceptionguard.data.scenes import build_scene
from perceptionguard.data.synthetic import default_intrinsics, render_frame
from perceptionguard.perception.detector import ColorModelDetector
from perceptionguard.perception.pipeline import PerceptionPipeline
from perceptionguard.reliability.monitor import ReliabilityMonitor
from perceptionguard.tracking.tracker import Tracker

CLASS_HEIGHTS = {"vehicle": 1.6, "pedestrian": 1.75, "cyclist": 1.7}
BASE_DT = 0.05
NUM_FRAMES = 24
STATUS_COLOR = {"TRUSTED": "#2ca02c", "CAUTION": "#ff9900", "DEGRADED": "#d62728"}

st.set_page_config(page_title="PerceptionGuard", layout="wide")


@st.cache_resource
def build_pipeline() -> PerceptionPipeline:
    """Build the pipeline once and fit the monitor's clean reference."""
    intr = default_intrinsics(640, 480)
    clean = []
    for i in range(NUM_FRAMES):
        boxes, T_cw = build_scene("multi", i, NUM_FRAMES)
        frame = render_frame(boxes, intr, T_cw, index=i, timestamp=i * BASE_DT)
        clean.append(frame.image)

    monitor = ReliabilityMonitor(class_heights=CLASS_HEIGHTS)
    monitor.fit_reference(clean, dt=BASE_DT)
    return PerceptionPipeline(ColorModelDetector(), Tracker(), monitor)


def draw_overlay(image: np.ndarray, output) -> np.ndarray:
    """Draw boxes, track IDs and depth onto a copy of the frame."""
    canvas = image.copy()
    for est in output.estimates:
        x1, y1, x2, y2 = (int(v) for v in est.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        depth = f"{est.depth:.1f}m" if np.isfinite(est.depth) else "no depth"
        cv2.putText(
            canvas,
            f"#{est.track_id} {est.label} {depth}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def depth_to_display(depth: np.ndarray) -> np.ndarray:
    """Colourize depth; NaN background renders black."""
    valid = np.isfinite(depth)
    out = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        d = depth[valid]
        lo, hi = float(d.min()), float(d.max())
        norm = (depth - lo) / max(hi - lo, 1e-6)
        out[valid] = np.clip(255 * (1.0 - norm[valid]), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(out, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


st.title("PerceptionGuard")
st.caption("Does the system know when to distrust itself?")

with st.sidebar:
    st.header("Scene")
    scene = st.selectbox("Scene", ["multi", "approach", "crossing", "occlusion"])
    frame_idx = st.slider("Frame", 0, NUM_FRAMES - 1, 8)
    st.header("Injected degradation")
    degradation = st.selectbox("Type", ["none", *sorted(DEGRADATIONS)])
    severity = st.slider(
        "Severity", 0.0, 1.0, 0.5, 0.25, disabled=degradation == "none"
    )

pipeline = build_pipeline()
pipeline.reset()
intr = default_intrinsics(640, 480)

# Replay from frame 0 so tracking and temporal signals have real history --
# scoring a single cold frame would make every temporal signal meaningless.
progress = st.progress(0.0, text="Replaying frames...")
output = None
for i in range(frame_idx + 1):
    boxes, T_cw = build_scene(scene, i, NUM_FRAMES)
    frame = render_frame(boxes, intr, T_cw, index=i, timestamp=i * BASE_DT)
    image, depth, cam, dt = frame.image, frame.depth, intr, BASE_DT
    if degradation != "none":
        deg = apply_degradation(
            degradation, severity, image, depth, intr, frame_index=i
        )
        image, depth, cam = deg.image, deg.depth, deg.intrinsics
        dt = getattr(deg, "dt", BASE_DT) or BASE_DT
    output = pipeline.process(
        image=image, depth=depth, intrinsics=cam, frame_index=i, dt=dt
    )
    progress.progress((i + 1) / (frame_idx + 1))
progress.empty()

report = output.report
color = STATUS_COLOR.get(report.status, "#888888")
st.markdown(
    f"<h2 style='color:{color};margin-bottom:0'>PERCEPTION STATUS: {report.status}</h2>"
    f"<p style='color:{color};font-size:1.4rem;margin-top:0'>reliability {report.score:.3f}</p>",
    unsafe_allow_html=True,
)
if report.diagnosis:
    st.warning("Likely cause: " + ", ".join(report.diagnosis))

c1, c2 = st.columns(2)
c1.image(
    draw_overlay(image, output),
    caption="Detections, track IDs, depth",
    use_container_width=True,
)
c2.image(
    depth_to_display(depth), caption="Depth (near = red)", use_container_width=True
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tracks", report.n_tracks)
m2.metric("Detections", len(output.detections))
m3.metric("Frame latency", f"{sum(output.timings_ms.values()):.1f} ms")
m4.metric("Monitor cost", f"{output.timings_ms.get('monitor', 0.0):.1f} ms")

st.subheader("Reliability signals")
st.caption(
    "1.0 = healthy, 0.0 = fully degraded. Fusion is a weighted geometric mean, "
    "so any single signal can veto the frame."
)
signals = dict(sorted(report.signals.items(), key=lambda kv: kv[1]))
for name, value in signals.items():
    st.progress(float(np.clip(value, 0.0, 1.0)), text=f"{name}  {value:.3f}")

with st.expander("Raw signal values (physical units)"):
    st.json(report.raw)

st.caption(
    "Known limitation: on clean frames the monitor scores below TRUSTED 66.7% of the time. "
    "It errs toward distrust. See README for the full honest-findings section."
)
