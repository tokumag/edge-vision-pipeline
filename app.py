"""
Edge Vision Pipeline — Live Dashboard
"""

from __future__ import annotations

import time
import copy
import yaml
from io import BytesIO
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from src.core.stream import SyntheticStream
from src.pipeline import DetectionPipeline
from src.benchmark.profiler import EdgeProfiler
from src.inference.factory import create_engine


# ---------------------------------------------------------------------------
# Theme & Layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Edge Vision Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 14px 16px;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stSidebar"] h1 { font-size: 1.15rem; letter-spacing: 0.03em; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_config() -> Dict[str, Any]:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PALETTE = {
    "normal_component": "#3b82f6",
    "surface_scratch":  "#eab308",
    "structural_crack": "#ef4444",
    "dent_defect":      "#f97316",
    "misalignment":     "#a855f7",
    "foreign_particle": "#06b6d4",
}


def sidebar(config: Dict[str, Any]):
    st.sidebar.markdown("# Edge Vision Pipeline")
    st.sidebar.caption("Production Edge Computer Vision")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Page",
        ["Inspect Live Stream", "Upload & Inspect", "Benchmark"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Detection Settings")
    conf = st.sidebar.slider("Confidence", 0.20, 0.90, 0.45, 0.05, format="%.2f")
    iou = st.sidebar.slider("NMS IoU", 0.20, 0.90, 0.50, 0.05, format="%.2f")
    show_heatmap = st.sidebar.checkbox("Overlay Defect Heatmap", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Pipeline Specs")
    st.sidebar.code(
        f"Backend   {config['model']['backend']}\n"
        f"Input     {config['model']['input_size'][0]}x{config['model']['input_size'][1]}\n"
        f"Classes   {len(config.get('classes', {}))}",
        language=None,
    )

    return page, conf, iou, show_heatmap


def page_inspect_stream(
    config: Dict[str, Any], conf: float, iou: float, show_heatmap: bool
) -> None:
    st.markdown("## Live Inspection Stream")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        n_frames = st.number_input("Frames to process", min_value=5, max_value=200, value=30, step=5)
    with c2:
        st.markdown("")
        run = st.button("Start Inspection Loop", type="primary", use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    slot_fps = m1.empty()
    slot_lat = m2.empty()
    slot_det = m3.empty()
    slot_trk = m4.empty()

    left, right = st.columns([5, 3])
    slot_frame = left.empty()
    slot_chart = right.empty()

    if not run:
        slot_frame.info("Click **Start Inspection Loop** to run real-time inference on the streaming buffer.")
        return

    active_config = copy.deepcopy(config)
    active_config["model"]["confidence_threshold"] = conf
    active_config["model"]["iou_threshold"] = iou

    pipeline = DetectionPipeline(config=active_config, force_mock=True)
    stream = SyntheticStream(resolution=(640, 640))

    class_accum: Dict[str, int] = {}
    defect_log = []

    for i in range(int(n_frames)):
        frame = stream.generate_frame()
        res = pipeline.process_frame(frame, apply_heatmap=show_heatmap)

        slot_fps.metric("FPS", f"{res.fps:.0f}")
        slot_lat.metric("Latency", f"{res.latency_ms:.1f} ms")
        slot_det.metric("Detections", len(res.detections))
        slot_trk.metric("Active Tracks", pipeline.tracker.next_track_id - 1)

        frame_rgb = res.annotated_frame[:, :, ::-1]
        slot_frame.image(frame_rgb, use_container_width=True)

        for d in res.detections:
            cn = d.get("class_name", "unknown")
            class_accum[cn] = class_accum.get(cn, 0) + 1
            if cn != "normal_component":
                defect_log.append({
                    "Frame": i + 1,
                    "Track ID": d.get("track_id", "—"),
                    "Defect Type": cn.replace("_", " ").title(),
                    "Confidence": f"{d['confidence']:.1%}",
                })

        if class_accum:
            names = list(class_accum.keys())
            counts = [class_accum[n] for n in names]
            colors = [PALETTE.get(n, "#888") for n in names]

            fig = go.Figure(go.Bar(
                y=[n.replace("_", " ").title() for n in names],
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

        time.sleep(0.02)

    st.markdown("---")
    summary = pipeline.get_summary_metrics()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Avg Latency", f"{summary['avg_latency_ms']:.1f} ms")
    r2.metric("Median (p50)", f"{summary['p50_latency_ms']:.1f} ms")
    r3.metric("Avg Throughput", f"{summary['avg_fps']:.0f} FPS")
    r4.metric("Defect Frame Rate", f"{summary['defect_rate']:.1%}")

    if show_heatmap:
        hotspots = pipeline.heatmap.get_top_hotspots(k=3)
        if hotspots:
            st.markdown("#### Top Recurrent Defect Hotspots")
            spot_cols = st.columns(len(hotspots))
            for idx, spot in enumerate(hotspots):
                spot_cols[idx].metric(
                    label=f"Hot-Zone #{idx + 1}",
                    value=f"X:{spot['center'][0]} Y:{spot['center'][1]}",
                    delta=f"{spot['density']:.0%} Peak Density",
                )

    if defect_log:
        st.markdown("#### Defect Telemetry Log")
        st.dataframe(pd.DataFrame(defect_log), use_container_width=True, hide_index=True)


def page_upload_inspect(
    config: Dict[str, Any], conf: float, iou: float, show_heatmap: bool
) -> None:
    st.markdown("## Upload & Inspect Media")
    st.caption("Upload an image to run single-shot edge detection and defect hotspot analysis.")

    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        # Convert RGB to BGR for pipeline
        frame_bgr = img_np[:, :, ::-1]

        active_config = copy.deepcopy(config)
        active_config["model"]["confidence_threshold"] = conf
        active_config["model"]["iou_threshold"] = iou

        pipeline = DetectionPipeline(config=active_config, force_mock=True)
        res = pipeline.process_frame(frame_bgr, apply_heatmap=show_heatmap)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.image(image, caption="Original Input", use_container_width=True)
        with col_right:
            annotated_rgb = res.annotated_frame[:, :, ::-1]
            st.image(annotated_rgb, caption=f"Detection Result ({res.latency_ms:.1f} ms)", use_container_width=True)

        st.markdown("---")
        st.markdown(f"**Found {len(res.detections)} objects** (Inference Latency: `{res.latency_ms:.1f} ms`)")

        if res.detections:
            rows = []
            for d in res.detections:
                rows.append({
                    "Class": d["class_name"].replace("_", " ").title(),
                    "Confidence": f"{d['confidence']:.1%}",
                    "Bounding Box [x1, y1, x2, y2]": str(d["bbox"]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_benchmark(config: Dict[str, Any]) -> None:
    st.markdown("## Inference Benchmark")
    st.caption("Measures real hardware latency distributions and throughput on this machine.")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        runs = st.number_input("Iterations", 20, 500, 50, 10)
    with c2:
        warmup = st.number_input("Warmup Passes", 5, 50, 10, 5)
    with c3:
        st.markdown("")
        go_btn = st.button("Run Hardware Benchmark", type="primary", use_container_width=True)

    if not go_btn:
        st.divider()
        st.info("Configure parameters and click **Run Hardware Benchmark**.")
        return

    engine = create_engine(config["model"], force_mock=True)
    target = tuple(config["model"]["input_size"])
    profiler = EdgeProfiler(
        engine=engine,
        warmup_runs=int(warmup),
        test_runs=int(runs),
        input_shape=(1, 3, target[0], target[1]),
    )

    with st.spinner("Benchmarking edge engine..."):
        res = profiler.run_benchmark()

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Throughput", f"{res['throughput_fps']:.1f} FPS")
    m2.metric("Median (p50)", f"{res['latency']['p50_ms']:.2f} ms")
    m3.metric("p95 Latency", f"{res['latency']['p95_ms']:.2f} ms")
    m4.metric("Memory RSS", f"{res['memory']['rss_mb']:.0f} MB")

    lat = res["latency"]
    percentiles = ["Min", "p50", "p90", "p95", "p99", "Max"]
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
    fig.add_hline(
        y=33.3, line_dash="dot", line_color="#ef4444",
        annotation_text="30 FPS Threshold (33.3 ms)",
        annotation_position="top left",
        annotation_font_color="#ef4444",
    )
    fig.update_layout(
        title="Hardware Latency Distribution",
        yaxis_title="Latency (ms)",
        xaxis_title="Percentile",
        height=340,
        margin=dict(l=10, r=10, t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Structured JSON Telemetry"):
        st.json(res)


def main() -> None:
    config = load_config()
    page, conf, iou, show_heatmap = sidebar(config)

    if page == "Inspect Live Stream":
        page_inspect_stream(config, conf, iou, show_heatmap)
    elif page == "Upload & Inspect":
        page_upload_inspect(config, conf, iou, show_heatmap)
    else:
        page_benchmark(config)


if __name__ == "__main__":
    main()
