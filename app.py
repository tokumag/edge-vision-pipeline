"""
Edge Vision Pipeline — Dashboard
"""

from __future__ import annotations

import time
import json
import yaml
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.core.stream import SyntheticStream
from src.core.preprocessor import LetterboxPreprocessor
from src.inference.factory import create_engine
from src.postprocess.nms import YOLOOutputDecoder
from src.postprocess.tracker import EdgeTracker
from src.postprocess.visualizer import FrameVisualizer
from src.benchmark.profiler import EdgeProfiler


# ---------------------------------------------------------------------------
# Theme & layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Edge Vision Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS overrides — let Streamlit's native dark theme do most of the work.
st.markdown(
    """
    <style>
    /* tighter top padding */
    .block-container { padding-top: 1.5rem; }

    /* metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 14px 16px;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }

    /* hide the default Streamlit menu & footer for a cleaner look */
    #MainMenu, footer { visibility: hidden; }

    /* sidebar title */
    [data-testid="stSidebar"] h1 { font-size: 1.15rem; letter-spacing: 0.03em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@st.cache_resource
def load_config() -> Dict[str, Any]:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Color palette — one palette, used everywhere
# ---------------------------------------------------------------------------

PALETTE = {
    "normal_component": "#3b82f6",
    "surface_scratch":  "#eab308",
    "structural_crack": "#ef4444",
    "dent_defect":      "#f97316",
    "misalignment":     "#a855f7",
    "foreign_particle": "#06b6d4",
}

SEVERITY = {
    "normal_component": 0,
    "surface_scratch":  1,
    "structural_crack": 2,
    "dent_defect":      1,
    "misalignment":     1,
    "foreign_particle": 1,
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar(config: Dict[str, Any]):
    st.sidebar.markdown("# Edge Vision Pipeline")
    st.sidebar.caption("v1.0.0")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Page",
        ["Inspect", "Benchmark"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Detection")
    conf = st.sidebar.slider("Confidence", 0.20, 0.90, 0.45, 0.05, format="%.2f")
    iou = st.sidebar.slider("NMS IoU", 0.20, 0.90, 0.50, 0.05, format="%.2f")

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Pipeline")
    st.sidebar.code(
        f"Backend   {config['model']['backend']}\n"
        f"Input     {config['model']['input_size'][0]}x{config['model']['input_size'][1]}\n"
        f"Classes   {len(config.get('classes', {}))}",
        language=None,
    )

    return page, conf, iou


# ---------------------------------------------------------------------------
# Page: Inspect
# ---------------------------------------------------------------------------

def page_inspect(config: Dict[str, Any], conf: float, iou: float) -> None:
    st.markdown("## Live Inspection")

    # --- controls row ---
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        n_frames = st.number_input("Frames", min_value=5, max_value=200, value=30, step=5)
    with c2:
        st.markdown("")  # vertical spacer
        run = st.button("Run", type="primary", use_container_width=True)
    with c3:
        pass  # breathing room

    # --- metrics ---
    m1, m2, m3, m4 = st.columns(4)
    slot_fps = m1.empty()
    slot_lat = m2.empty()
    slot_det = m3.empty()
    slot_trk = m4.empty()

    # --- main canvas ---
    left, right = st.columns([5, 3])
    slot_frame = left.empty()
    slot_chart = right.empty()

    # --- detection log ---
    slot_log = st.empty()

    if not run:
        slot_frame.info("Press **Run** to start the inspection loop.")
        return

    # Build pipeline
    target = tuple(config["model"]["input_size"])
    preprocessor = LetterboxPreprocessor(target_size=target)
    engine = create_engine(config["model"], force_mock=True)
    decoder = YOLOOutputDecoder(
        conf_threshold=conf,
        iou_threshold=iou,
        class_names=config.get("classes", {}),
    )
    tracker = EdgeTracker()
    visualizer = FrameVisualizer()
    stream = SyntheticStream(resolution=(640, 640))

    class_accum: Dict[str, int] = {}
    latencies = []
    log_rows = []

    for i in range(int(n_frames)):
        t0 = time.perf_counter()
        frame = stream.generate_frame()

        tensor, meta = preprocessor.preprocess(frame)
        raw = engine.infer(tensor)
        dets = decoder.decode(raw, meta)
        dets = tracker.update(dets)

        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        fps = 1000.0 / max(lat, 0.01)

        for d in dets:
            cn = d.get("class_name", "unknown")
            class_accum[cn] = class_accum.get(cn, 0) + 1

        annotated = visualizer.render(
            frame, dets, fps=fps, latency_ms=lat, backend=engine.backend_name,
        )

        # --- update UI ---
        slot_fps.metric("FPS", f"{fps:.0f}")
        slot_lat.metric("Latency", f"{lat:.1f} ms")
        slot_det.metric("Detections", len(dets))
        slot_trk.metric("Tracks", tracker.next_track_id - 1)

        frame_rgb = annotated[:, :, ::-1]
        slot_frame.image(frame_rgb, use_container_width=True)

        # cumulative class bar
        if class_accum:
            names = list(class_accum.keys())
            counts = [class_accum[n] for n in names]
            colors = [PALETTE.get(n, "#888") for n in names]

            fig = go.Figure(go.Bar(
                y=[n.replace("_", " ") for n in names],
                x=counts,
                orientation="h",
                marker_color=colors,
                text=counts,
                textposition="outside",
            ))
            fig.update_layout(
                height=260,
                margin=dict(l=0, r=20, t=10, b=10),
                xaxis_title="Count",
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"),
            )
            slot_chart.plotly_chart(fig, use_container_width=True, key=f"bar_{i}")

        # collect log
        for d in dets:
            sev = SEVERITY.get(d["class_name"], 0)
            if sev > 0:
                log_rows.append({
                    "frame": i + 1,
                    "class": d["class_name"].replace("_", " "),
                    "confidence": f"{d['confidence']:.0%}",
                    "track": d.get("track_id", "—"),
                })

        time.sleep(0.02)

    # --- summary after loop ---
    st.markdown("---")

    avg_lat = np.mean(latencies)
    avg_fps = 1000.0 / max(avg_lat, 0.01)

    r1, r2, r3 = st.columns(3)
    r1.metric("Avg Latency", f"{avg_lat:.1f} ms")
    r2.metric("Avg FPS", f"{avg_fps:.0f}")
    r3.metric("Total Frames", int(n_frames))

    if log_rows:
        st.markdown("#### Defect Log")
        st.dataframe(
            pd.DataFrame(log_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No defects detected in this run.")


# ---------------------------------------------------------------------------
# Page: Benchmark
# ---------------------------------------------------------------------------

def page_benchmark(config: Dict[str, Any]) -> None:
    st.markdown("## Inference Benchmark")
    st.caption("Measures actual latency percentiles and throughput on this machine.")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        runs = st.number_input("Iterations", 20, 500, 50, 10)
    with c2:
        warmup = st.number_input("Warmup", 5, 50, 10, 5)
    with c3:
        st.markdown("")
        go_btn = st.button("Run benchmark", type="primary", use_container_width=True)

    if not go_btn:
        st.divider()
        st.info("Configure iterations and warmup, then press **Run benchmark**.")
        return

    engine = create_engine(config["model"], force_mock=True)
    target = tuple(config["model"]["input_size"])
    profiler = EdgeProfiler(
        engine=engine,
        warmup_runs=int(warmup),
        test_runs=int(runs),
        input_shape=(1, 3, target[0], target[1]),
    )

    with st.spinner("Running..."):
        res = profiler.run_benchmark()

    st.divider()

    # --- headline metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Throughput", f"{res['throughput_fps']:.1f} FPS")
    m2.metric("Median (p50)", f"{res['latency']['p50_ms']:.2f} ms")
    m3.metric("p95 Latency", f"{res['latency']['p95_ms']:.2f} ms")
    m4.metric("Memory RSS", f"{res['memory']['rss_mb']:.0f} MB")

    # --- latency distribution chart ---
    lat = res["latency"]
    percentiles = ["min", "p50", "p90", "p95", "p99", "max"]
    values = [lat["min_ms"], lat["p50_ms"], lat["p90_ms"], lat["p95_ms"], lat["p99_ms"], lat["max_ms"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=percentiles,
        y=values,
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
    ))

    # real-time target line
    fig.add_hline(
        y=33.3, line_dash="dot", line_color="#ef4444",
        annotation_text="30 FPS threshold (33.3 ms)",
        annotation_position="top left",
        annotation_font_color="#ef4444",
    )

    fig.update_layout(
        title="Latency Distribution",
        yaxis_title="Latency (ms)",
        xaxis_title="Percentile",
        height=340,
        margin=dict(l=10, r=10, t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- raw results ---
    with st.expander("Raw benchmark data"):
        st.json(res)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config()
    page, conf, iou = sidebar(config)

    if page == "Inspect":
        page_inspect(config, conf, iou)
    else:
        page_benchmark(config)


if __name__ == "__main__":
    main()
