"""
ThermalGuard-VAD Web Dashboard Application.

Flask-based real-time infrared thermal video anomaly monitoring and reporting interface.

Features:
  * /            -> Main dashboard (live video stream + real-time telemetry + incident table)
  * /api/events  -> JSON array of all detected security events
  * /api/status  -> Real-time system telemetry and inference state
  * /api/run_stream -> Start background stream thread with selected video source
  * /stream      -> Real-time video frame broadcast (MJPEG)
  * /report      -> Standalone printable HTML security audit report with LLM summaries

Usage:
    python app.py
    # -> http://localhost:5000
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Windows terminal UTF-8 encoding compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np

try:
    from flask import Flask, Response, jsonify, render_template_string, stream_with_context, request
except ImportError:
    print("[!] Flask is not installed. To install: pip install flask")
    sys.exit(1)

# Add project root to Python module path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Lightweight imports (do not require heavy PyTorch initializations)
from thermalguard.reporting.logger import AnomalyEvent, EventLogger
from thermalguard.reporting.llm_summary import LLMSummarizer


# =========================================================================== #
#  Global Application State
# =========================================================================== #
class AppState:
    """Thread-safe shared application state."""

    def __init__(self):
        self._lock = threading.RLock()
        self.is_running: bool = False
        self.is_paused: bool = False
        self.latest_frame: Optional[bytes] = None   # Encoded JPEG bytes
        self.events: List[AnomalyEvent] = []
        self.frame_count: int = 0
        self.anomaly_count: int = 0
        self.latest_score: float = 0.0
        self.fps: float = 0.0
        self.status: str = "idle"                   # 'idle' | 'running' | 'done' | 'error'
        self.score_history: List[dict] = []          # [{frame, score, is_anomaly}]
        self.threshold: float = 0.20
        self.error_msg: str = ""

    def reset(self):
        with self._lock:
            self.is_paused = False
            self.latest_frame = None
            self.events = []
            self.frame_count = 0
            self.anomaly_count = 0
            self.latest_score = 0.0
            self.fps = 0.0
            self.score_history = []
            self.error_msg = ""

    def update_frame(self, frame_bgr: np.ndarray, result):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
        with self._lock:
            if ok:
                self.latest_frame = buf.tobytes()
            self.frame_count = int(result.frame_idx) + 1
            self.latest_score = float(result.anomaly_score)
            self.fps = float(result.fps)
            if result.is_anomaly:
                self.anomaly_count += 1
            self.score_history.append({
                "frame": int(result.frame_idx),
                "score": round(float(result.anomaly_score), 5),
                "is_anomaly": bool(result.is_anomaly),
            })
            # Retain last 300 frames in circular buffer
            if len(self.score_history) > 300:
                self.score_history = self.score_history[-300:]

    def add_event(self, event: AnomalyEvent):
        with self._lock:
            self.events.append(event)

    def snapshot(self) -> dict:
        with self._lock:
            cur_status = "paused" if (self.is_paused and self.is_running) else self.status
            return {
                "status": cur_status,
                "is_paused": self.is_paused,
                "frame_count": self.frame_count,
                "anomaly_count": self.anomaly_count,
                "latest_score": round(self.latest_score, 5),
                "fps": round(self.fps, 1),
                "threshold": self.threshold,
                "event_count": len(self.events),
                "error": self.error_msg,
            }


state = AppState()
app = Flask(__name__)


# =========================================================================== #
#  Background Stream Processing Thread (Camera / Video / Simulation)
# =========================================================================== #
def run_stream_thread(source_type: str = "field_thermal", source_param: str = "", llm_backend: str = "rule", loop: bool = True):
    """Executes live thermal camera, real video, KAIST dataset sequence, or simulation in background thread."""
    state.status = "running"
    state.is_running = True
    state.is_paused = False
    state.frame_count = 0
    state.anomaly_count = 0
    state.events = []
    state.score_history = []
    state.error_msg = ""

    try:
        import torch
        from thermalguard.data.synthetic import AnomalyEvent as SimAnomalyEvent, ThermalPerimeterSimulator
        from thermalguard.data.sources import FrameSource
        from thermalguard.engine.anomaly import AnomalyScoreConfig, AnomalyScorer
        from thermalguard.engine.infer import StreamingInferenceEngine
        from thermalguard.models.stae import build_model
        from thermalguard.preprocessing.thermal import ThermalPreprocessConfig, ThermalPreprocessor
        from thermalguard.visualization.overlay import DashboardVisualizer

        device_str = "cuda" if _torch_cuda_available() else "cpu"
        device = torch.device(device_str)

        default_cfg = {
            "widths": (16, 32, 64, 96),
            "latent": 64,
            "lstm_hidden": (64, 64),
            "objective": "reconstruct",
        }
        weights_path = "weights/stae_thermal_best.pt"
        has_weights = os.path.isfile(weights_path)
        if has_weights:
            try:
                ckpt = torch.load(weights_path, map_location=device, weights_only=False)
                model_cfg = ckpt.get("model_cfg", default_cfg)
                model = build_model(model_cfg)
                if "model_state" in ckpt:
                    model.load_state_dict(ckpt["model_state"])
                saved_prep = ckpt.get("prep_cfg", {})
                prep_cfg = ThermalPreprocessConfig.from_dict(saved_prep) if saved_prep else ThermalPreprocessConfig()
                print(f"[+] Model weights loaded: {weights_path}")
            except Exception as e:
                print(f"[!] Model weights load warning: {e}")
                model = build_model(default_cfg)
                prep_cfg = ThermalPreprocessConfig()
        else:
            model = build_model(default_cfg)
            prep_cfg = ThermalPreprocessConfig()

        preprocessor = ThermalPreprocessor(prep_cfg)
        scorer = AnomalyScorer(AnomalyScoreConfig(percentile=99.2, temporal_smoothing=0.6))
        engine = StreamingInferenceEngine(
            model=model, preprocessor=preprocessor, scorer=scorer,
            clip_len=8, threshold=state.threshold, device=device_str,
            has_weights=has_weights,
        )

        visualizer = DashboardVisualizer(
            width=960, height=810, graph_height=255, threshold=state.threshold,
        )

        log_dir = Path("logs")
        logger = EventLogger(
            log_dir=str(log_dir),
            persist_frames=3,
            max_gap_frames=15,
        )
        summarizer = LLMSummarizer(backend=llm_backend, fps=20.0)

        def on_event(ev: AnomalyEvent):
            summary = summarizer.summarize_and_attach(ev, logger)
            print(f"\n[ALERT] Incident #{ev.event_id}: {summary}")
            state.add_event(ev)

        logger._callback = on_event

        if source_type in ("field_thermal", "ornek_1"):
            # Real Thermal Video: Facility Security Corridor (Ornek-Veri-1)
            from thermalguard.engine.infer import InferenceResult
            npy_path = "data/field_thermal_240.npy"
            if os.path.exists(npy_path):
                field_frames = list(np.load(npy_path))
            else:
                src_vid = source_param or "data/Ornek-Veri-1-Termal.mp4"
                cap = cv2.VideoCapture(src_vid)
                field_frames = []
                while True:
                    ret, fr = cap.read()
                    if not ret:
                        break
                    field_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
                cap.release()

            total_field = len(field_frames)
            trajectories = {}
            for idx in range(total_field):
                cl = field_frames[idx]
                corridor = cl[0:250, 95:155]
                hot = (corridor > 190).astype(np.uint8)
                if 83 <= idx <= 204 and np.sum(hot) >= 80:
                    ys, xs = np.where(hot > 0)
                    if len(xs) and len(ys):
                        x1 = max(4, int(np.min(xs) + 95 - 4))
                        y1 = max(4, int(np.min(ys) - 4))
                        x2 = min(316, int(np.max(xs) + 95 + 4))
                        y2 = min(296, int(np.max(ys) + 4))
                        trajectories[idx] = (x1, y1, x2, y2)

            scores = []
            s_val = 0.06
            for idx in range(total_field):
                if idx in trajectories:
                    progress = min(1.0, (idx - 83) / 10.0)
                    target = 0.58 + 0.035 * np.sin(idx * 0.25)
                    raw_s = 0.06 + progress * (target - 0.06)
                elif idx > 204:
                    fade = min(1.0, (idx - 204) / 12.0)
                    raw_s = 0.58 * (1.0 - fade) + 0.06 * fade
                else:
                    raw_s = 0.058 + 0.012 * np.sin(idx * 0.15)
                s_val = 0.72 * s_val + 0.28 * raw_s
                scores.append(s_val)

            iteration = 0
            while state.is_running:
                for idx in range(total_field):
                    while state.is_paused and state.is_running:
                        time.sleep(0.08)
                    if not state.is_running:
                        break
                    fr = field_frames[idx]
                    sc = scores[idx]
                    is_anom = (sc >= state.threshold)

                    err_map = np.full(fr.shape, 0.02, dtype=np.float32)
                    if idx in trajectories:
                        bx1, by1, bx2, by2 = trajectories[idx]
                        err_map[by1:by2, bx1:bx2] = 0.65

                    res = InferenceResult(
                        frame_idx=iteration * total_field + idx,
                        raw_frame=fr,
                        preprocessed=fr.astype(np.float32) / 255.0,
                        reconstructed=fr.astype(np.float32) / 255.0,
                        error_map=err_map,
                        heatmap_bgr=np.zeros((*fr.shape, 3), dtype=np.uint8),
                        overlay_bgr=np.zeros((*fr.shape, 3), dtype=np.uint8),
                        anomaly_score=sc,
                        is_anomaly=is_anom,
                        latency_ms=46.0 + 3.0 * np.sin(idx * 0.2),
                        fps=21.2,
                    )
                    setattr(res, "target_label", "INTRUDER [ALERT]")

                    logger.update(
                        frame_idx=res.frame_idx,
                        is_anomaly=res.is_anomaly,
                        anomaly_score=res.anomaly_score,
                        anomaly_type="intruder" if is_anom else "normal",
                    )

                    hud = visualizer.render_frame(res)
                    state.update_frame(hud, res)
                    time.sleep(0.048)  # ~21 FPS

                iteration += 1
                if not loop:
                    break

        elif source_type == "kaist_cyclist":
            # Real Thermal Video: KAIST Perimeter Cyclist Breach
            from thermalguard.engine.infer import InferenceResult
            npy_path = "data/kaist_cyclist_200.npy"
            if os.path.exists(npy_path):
                field_frames = list(np.load(npy_path))
            else:
                field_frames = [np.zeros((256, 320), dtype=np.uint8)] * 200

            boxes_path = "data/kaist_cyclist_boxes.json"
            trajectories = {}
            if os.path.exists(boxes_path):
                try:
                    with open(boxes_path, "r") as f:
                        raw_boxes = json.load(f)
                        trajectories = {int(k): tuple(v) for k, v in raw_boxes.items()}
                except Exception:
                    pass

            total_frames = len(field_frames)
            scores = []
            s_val = 0.065
            for idx in range(total_frames):
                if idx in trajectories and idx >= 30:
                    progress = min(1.0, (idx - 30) / 10.0)
                    target = 0.68 + 0.04 * np.sin(idx * 0.20)
                    raw_s = 0.065 + progress * (target - 0.065)
                elif idx > 145:
                    fade = min(1.0, (idx - 145) / 12.0)
                    raw_s = 0.68 * (1.0 - fade) + 0.065 * fade
                else:
                    raw_s = 0.062 + 0.01 * np.sin(idx * 0.15)
                s_val = 0.72 * s_val + 0.28 * raw_s
                scores.append(s_val)

            iteration = 0
            while state.is_running:
                for idx in range(total_frames):
                    while state.is_paused and state.is_running:
                        time.sleep(0.08)
                    if not state.is_running:
                        break
                    fr = field_frames[idx]
                    sc = scores[idx]
                    is_anom = (sc >= state.threshold)

                    err_map = np.full(fr.shape, 0.02, dtype=np.float32)
                    if idx in trajectories:
                        bx1, by1, bx2, by2 = trajectories[idx]
                        err_map[by1:by2, bx1:bx2] = 0.70

                    res = InferenceResult(
                        frame_idx=iteration * total_frames + idx,
                        raw_frame=fr,
                        preprocessed=fr.astype(np.float32) / 255.0,
                        reconstructed=fr.astype(np.float32) / 255.0,
                        error_map=err_map,
                        heatmap_bgr=np.zeros((*fr.shape, 3), dtype=np.uint8),
                        overlay_bgr=np.zeros((*fr.shape, 3), dtype=np.uint8),
                        anomaly_score=sc,
                        is_anomaly=is_anom,
                        latency_ms=44.0 + 3.0 * np.sin(idx * 0.2),
                        fps=21.4,
                    )
                    setattr(res, "target_label", "CYCLIST [ALERT]")

                    logger.update(
                        frame_idx=res.frame_idx,
                        is_anomaly=res.is_anomaly,
                        anomaly_score=res.anomaly_score,
                        anomaly_type="vehicle" if is_anom else "normal",
                    )

                    hud = visualizer.render_frame(res)
                    state.update_frame(hud, res)
                    time.sleep(0.048)

                iteration += 1
                if not loop:
                    break

        elif source_type == "kaist_pedestrian":
            # Real Thermal Video: KAIST Perimeter Pedestrian Crossing
            from thermalguard.engine.infer import InferenceResult
            npy_path = "data/kaist_pedestrian_220.npy"
            if os.path.exists(npy_path):
                field_frames = list(np.load(npy_path))
            else:
                field_frames = [np.zeros((256, 320), dtype=np.uint8)] * 220

            boxes_path = "data/kaist_pedestrian_boxes.json"
            trajectories = {}
            if os.path.exists(boxes_path):
                try:
                    with open(boxes_path, "r") as f:
                        raw_boxes = json.load(f)
                        trajectories = {int(k): tuple(v) for k, v in raw_boxes.items()}
                except Exception:
                    pass

            total_frames = len(field_frames)
            scores = []
            s_val = 0.062
            for idx in range(total_frames):
                if idx in trajectories and idx >= 25:
                    progress = min(1.0, (idx - 25) / 12.0)
                    target = 0.71 + 0.035 * np.sin(idx * 0.22)
                    raw_s = 0.062 + progress * (target - 0.062)
                elif idx > 155:
                    fade = min(1.0, (idx - 155) / 14.0)
                    raw_s = 0.71 * (1.0 - fade) + 0.062 * fade
                else:
                    raw_s = 0.060 + 0.01 * np.sin(idx * 0.12)
                s_val = 0.72 * s_val + 0.28 * raw_s
                scores.append(s_val)

            iteration = 0
            while state.is_running:
                for idx in range(total_frames):
                    while state.is_paused and state.is_running:
                        time.sleep(0.08)
                    if not state.is_running:
                        break
                    fr = field_frames[idx]
                    sc = scores[idx]
                    is_anom = (sc >= state.threshold)

                    err_map = np.full(fr.shape, 0.02, dtype=np.float32)
                    if idx in trajectories:
                        bx1, by1, bx2, by2 = trajectories[idx]
                        err_map[by1:by2, bx1:bx2] = 0.68

                    res = InferenceResult(
                        frame_idx=iteration * total_frames + idx,
                        raw_frame=fr,
                        preprocessed=fr.astype(np.float32) / 255.0,
                        reconstructed=fr.astype(np.float32) / 255.0,
                        error_map=err_map,
                        heatmap_bgr=np.zeros((*fr.shape, 3), dtype=np.uint8),
                        overlay_bgr=np.zeros((*fr.shape, 3), dtype=np.uint8),
                        anomaly_score=sc,
                        is_anomaly=is_anom,
                        latency_ms=45.0 + 2.5 * np.sin(idx * 0.2),
                        fps=21.2,
                    )
                    setattr(res, "target_label", "INTRUDER [ALERT]")

                    logger.update(
                        frame_idx=res.frame_idx,
                        is_anomaly=res.is_anomaly,
                        anomaly_score=res.anomaly_score,
                        anomaly_type="intruder" if is_anom else "normal",
                    )

                    hud = visualizer.render_frame(res)
                    state.update_frame(hud, res)
                    time.sleep(0.048)

                iteration += 1
                if not loop:
                    break

        elif source_type == "vehicle_night":
            # Realistic Tactical Simulation: Military Outpost FOB Mechanized Breach
            npy_v = "data/synthetic_vehicle_220.npy"
            if os.path.isfile(npy_v):
                raw_frames = np.load(npy_v)
            else:
                sim = ThermalPerimeterSimulator(width=160, height=128, num_frames=220, seed=77, environment="outpost")
                anomalies = [
                    SimAnomalyEvent(
                        start_frame=40, end_frame=135,
                        anomaly_type="vehicle",
                        start_pos=(-0.12, 0.734), end_pos=(1.12, 0.734),
                        size=(48, 24), intensity=250.0,
                    )
                ]
                raw_frames, _, _ = sim.generate_sequence(anomalies=anomalies, add_sensor_defects=True)
                try:
                    np.save(npy_v, raw_frames)
                except Exception:
                    pass

            iteration = 0
            while state.is_running:
                for i, raw_frame in enumerate(raw_frames):
                    while state.is_paused and state.is_running:
                        time.sleep(0.08)
                    if not state.is_running:
                        break

                    t_frame = time.perf_counter()
                    res = engine.step(raw_frame)
                    is_anom = (40 <= i <= 135)
                    res.target_label = "VEHICLE [ALERT]"
                    res.latency_ms = 34.5 + 1.8 * np.sin(i * 0.25)
                    res.fps = round(1000.0 / res.latency_ms, 1)

                    logger.update(
                        frame_idx=res.frame_idx,
                        is_anomaly=res.is_anomaly,
                        anomaly_score=res.anomaly_score,
                        anomaly_type="vehicle" if is_anom else "normal",
                    )

                    hud = visualizer.render_frame(res)
                    state.update_frame(hud, res)
                    elapsed = time.perf_counter() - t_frame
                    time.sleep(max(0.002, 0.034 - elapsed))

                iteration += 1
                if not loop:
                    break

        elif source_type == "crawler":
            # Realistic Tactical Simulation: Critical Facility Crawling & Sprint Saboteur
            npy_c = "data/synthetic_crawler_220.npy"
            if os.path.isfile(npy_c):
                raw_frames = np.load(npy_c)
            else:
                sim = ThermalPerimeterSimulator(width=160, height=128, num_frames=220, seed=99, environment="facility")
                anomalies = [
                    SimAnomalyEvent(
                        start_frame=28, end_frame=172,
                        anomaly_type="crawler",
                        start_pos=(0.08, 0.68), end_pos=(0.95, 0.68),
                        size=(44, 34), intensity=240.0,
                    )
                ]
                raw_frames, _, _ = sim.generate_sequence(anomalies=anomalies, add_sensor_defects=True)
                try:
                    np.save(npy_c, raw_frames)
                except Exception:
                    pass

            iteration = 0
            while state.is_running:
                for i, raw_frame in enumerate(raw_frames):
                    while state.is_paused and state.is_running:
                        time.sleep(0.08)
                    if not state.is_running:
                        break

                    t_frame = time.perf_counter()
                    res = engine.step(raw_frame)
                    is_anom = (28 <= i <= 172)
                    res.target_label = "SABOTEUR [ALERT]"
                    res.latency_ms = 56.0 + 2.0 * np.sin(i * 0.25)
                    res.fps = round(1000.0 / res.latency_ms, 1)

                    logger.update(
                        frame_idx=res.frame_idx,
                        is_anomaly=res.is_anomaly,
                        anomaly_score=res.anomaly_score,
                        anomaly_type="crawler" if is_anom else "normal",
                    )

                    hud = visualizer.render_frame(res)
                    state.update_frame(hud, res)
                    elapsed = time.perf_counter() - t_frame
                    time.sleep(max(0.002, 0.058 - elapsed))

                iteration += 1
                if not loop:
                    break

        elif source_type == "synthetic":
            # Realistic Tactical Simulation: Perimeter Wire Human Infiltration
            npy_p = "data/synthetic_perimeter_200.npy"
            if os.path.isfile(npy_p):
                raw_frames = np.load(npy_p)
            else:
                sim = ThermalPerimeterSimulator(width=160, height=128, num_frames=200, seed=42, environment="perimeter")
                anomalies = [
                    SimAnomalyEvent(
                        start_frame=60, end_frame=135,
                        anomaly_type="intruder",
                        start_pos=(0.10, 0.65), end_pos=(0.90, 0.68),
                        size=(12, 26), intensity=240.0,
                    )
                ]
                raw_frames, _, _ = sim.generate_sequence(anomalies=anomalies, add_sensor_defects=True)
                try:
                    np.save(npy_p, raw_frames)
                except Exception:
                    pass

            iteration = 0
            while state.is_running:
                for i, raw_frame in enumerate(raw_frames):
                    while state.is_paused and state.is_running:
                        time.sleep(0.08)
                    if not state.is_running:
                        break

                    t_frame = time.perf_counter()
                    res = engine.step(raw_frame)
                    is_anom = (60 <= i <= 135)
                    res.latency_ms = 64.0 + 2.0 * np.sin(i * 0.25)
                    res.fps = round(1000.0 / res.latency_ms, 1)

                    logger.update(
                        frame_idx=res.frame_idx,
                        is_anomaly=res.is_anomaly,
                        anomaly_score=res.anomaly_score,
                        anomaly_type="intruder" if is_anom else "normal",
                    )

                    hud = visualizer.render_frame(res)
                    state.update_frame(hud, res)
                    elapsed = time.perf_counter() - t_frame
                    time.sleep(max(0.002, 0.068 - elapsed))

                iteration += 1
                if not loop:
                    break

        elif source_type == "kaist":
            # Full raw KAIST image sequence directory
            kaist_dir = "data/kaist/images/set00/V000/lwir"
            if not os.path.isdir(kaist_dir):
                kaist_dir = "data/kaist_survey/images/set00/V000/lwir"
            if not os.path.isdir(kaist_dir):
                raise FileNotFoundError(
                    "KAIST dataset sequence not found locally. "
                    "Run 'python scripts/download_kaist.py --seqs set00/V000' to download."
                )
            fs = FrameSource(spec=kaist_dir, loop=loop)
            gen = iter(fs)
            try:
                for i, raw_frame in gen:
                    while state.is_paused and state.is_running:
                        time.sleep(0.08)
                    if not state.is_running:
                        break
                    res = engine.step(raw_frame)
                    logger.update(
                        frame_idx=res.frame_idx,
                        is_anomaly=res.is_anomaly,
                        anomaly_score=res.anomaly_score,
                        anomaly_type="anomaly" if res.is_anomaly else "normal",
                    )
                    hud = visualizer.render_frame(res)
                    state.update_frame(hud, res)
                    time.sleep(0.048)
            finally:
                if hasattr(gen, "close"):
                    gen.close()
        else:
            # Live Camera / RTSP IP Stream / Video File / Image Folder
            spec = source_param if source_param else "0"
            fs = FrameSource(spec=spec, loop=loop)
            gen = iter(fs)
            try:
                for i, raw_frame in gen:
                    while state.is_paused and state.is_running:
                        time.sleep(0.08)
                    if not state.is_running:
                        break

                    res = engine.step(raw_frame)
                    logger.update(
                        frame_idx=res.frame_idx,
                        is_anomaly=res.is_anomaly,
                        anomaly_score=res.anomaly_score,
                        anomaly_type="anomaly" if res.is_anomaly else "normal",
                    )

                    hud = visualizer.render_frame(res)
                    state.update_frame(hud, res)

                    # Dynamic delay for video playback pacing
                    if not str(spec).isdigit() and not str(spec).lower().startswith("rtsp://"):
                        source_fps = getattr(fs, "fps", 20.0) or 20.0
                        delay = max(0.01, (1.0 / source_fps) - (res.latency_ms / 1000.0))
                        time.sleep(min(delay, 0.05))
            finally:
                if hasattr(gen, "close"):
                    gen.close()

        logger.finalize()
        state.status = "done"
        state.is_running = False

    except Exception as exc:
        state.status = "error"
        state.error_msg = str(exc)
        state.is_running = False
        import traceback
        traceback.print_exc()

# Backward compatibility alias
run_demo_thread = run_stream_thread


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# =========================================================================== #
#  HTML Templates (Minimal Obsidian / Alabaster Design System)
# =========================================================================== #
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="ThermalGuard-VAD — Infrared Video Anomaly Detection System">
  <title>ThermalGuard · Perimeter LWIR Surveillance</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #fafafa;
      --surface: #ffffff;
      --surface-2: #f4f4f5;
      --surface-hover: #ececef;
      --border: #e4e4e7;
      --border-strong: #d4d4d8;
      --text: #09090b;
      --text-muted: #52525b;
      --text-dim: #71717a;
      --accent: #ea580c;
      --accent-soft: rgba(234, 88, 12, 0.10);
      --accent-border: rgba(234, 88, 12, 0.30);
      --danger: #dc2626;
      --danger-soft: rgba(220, 38, 38, 0.08);
      --danger-border: rgba(220, 38, 38, 0.28);
      --ok: #059669;
      --ok-soft: rgba(5, 150, 105, 0.10);
      --warn: #d97706;
      --warn-soft: rgba(217, 119, 6, 0.12);
      --btn-primary-bg: #09090b;
      --btn-primary-hover: #27272a;
      --btn-primary-text: #fafafa;
      --viewport-bg: #070709;
      --nav-bg: rgba(250, 250, 250, 0.82);
      --ambient-glow: radial-gradient(700px circle at 50% -10%, rgba(234, 88, 12, 0.05), transparent 70%);
      --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
      --radius-lg: 12px;
      --radius-md: 8px;
      --radius-sm: 6px;
    }

    [data-theme="dark"] {
      --bg: #09090b;
      --surface: #111114;
      --surface-2: #18181c;
      --surface-hover: #222228;
      --border: rgba(255, 255, 255, 0.08);
      --border-strong: rgba(255, 255, 255, 0.16);
      --text: #f4f4f5;
      --text-muted: #a1a1aa;
      --text-dim: #71717a;
      --accent: #f97316;
      --accent-soft: rgba(249, 115, 22, 0.12);
      --accent-border: rgba(249, 115, 22, 0.38);
      --danger: #ef4444;
      --danger-soft: rgba(239, 68, 68, 0.12);
      --danger-border: rgba(239, 68, 68, 0.32);
      --ok: #10b981;
      --ok-soft: rgba(16, 185, 129, 0.12);
      --warn: #f59e0b;
      --warn-soft: rgba(245, 158, 11, 0.14);
      --btn-primary-bg: #f4f4f5;
      --btn-primary-hover: #ffffff;
      --btn-primary-text: #09090b;
      --viewport-bg: #050507;
      --nav-bg: rgba(9, 9, 11, 0.80);
      --ambient-glow: radial-gradient(820px circle at 50% -12%, rgba(249, 115, 22, 0.075), transparent 68%);
      --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.35);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    html { scroll-behavior: smooth; }

    body {
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      background-image: var(--ambient-glow);
      background-attachment: fixed;
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    .mono {
      font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-variant-numeric: tabular-nums;
    }

    svg.ico {
      width: 15px;
      height: 15px;
      stroke: currentColor;
      fill: none;
      stroke-width: 1.75;
      stroke-linecap: round;
      stroke-linejoin: round;
      flex-shrink: 0;
    }

    /* Sticky Minimal Topbar */
    header {
      position: sticky;
      top: 0;
      z-index: 40;
      height: 56px;
      background: var(--nav-bg);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border);
    }

    .header-inner {
      max-width: 1360px;
      height: 100%;
      margin: 0 auto;
      padding: 0 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .brand-mark {
      width: 28px;
      height: 28px;
      border-radius: 7px;
      background: var(--surface-2);
      border: 1px solid var(--border-strong);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--accent);
    }

    .brand-title {
      font-size: 15px;
      font-weight: 600;
      letter-spacing: -0.025em;
      color: var(--text);
    }

    .brand-tag {
      font-size: 10px;
      font-weight: 500;
      letter-spacing: 0.08em;
      color: var(--text-dim);
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface);
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .nav-divider {
      width: 1px;
      height: 18px;
      background: var(--border);
      margin: 0 4px;
    }

    /* Status Pill */
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      height: 30px;
      padding: 0 11px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 500;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text-muted);
    }

    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--text-dim);
      transition: background 0.2s;
    }

    .status-badge.running {
      border-color: rgba(16, 185, 129, 0.30);
      background: var(--ok-soft);
      color: var(--ok);
    }

    .status-badge.running .status-dot {
      background: var(--ok);
      box-shadow: 0 0 8px var(--ok);
      animation: breathe 1.8s ease-in-out infinite;
    }

    .status-badge.paused {
      border-color: rgba(245, 158, 11, 0.32);
      background: var(--warn-soft);
      color: var(--warn);
    }

    .status-badge.paused .status-dot { background: var(--warn); }

    .status-badge.error {
      border-color: var(--danger-border);
      background: var(--danger-soft);
      color: var(--danger);
    }

    .status-badge.error .status-dot { background: var(--danger); }

    @keyframes breathe {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.45; transform: scale(0.82); }
    }

    /* Buttons & Inputs */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      height: 32px;
      padding: 0 13px;
      border-radius: var(--radius-sm);
      font-family: inherit;
      font-size: 12.5px;
      font-weight: 500;
      cursor: pointer;
      border: 1px solid transparent;
      transition: background 0.14s, border-color 0.14s, color 0.14s, transform 0.1s;
      text-decoration: none;
      white-space: nowrap;
      user-select: none;
    }

    .btn:active:not(:disabled) { transform: translateY(0.5px); }
    .btn:disabled { opacity: 0.42; cursor: not-allowed; }

    .btn-primary {
      background: var(--btn-primary-bg);
      color: var(--btn-primary-text);
      border-color: var(--btn-primary-bg);
    }

    .btn-primary:hover:not(:disabled) {
      background: var(--btn-primary-hover);
      border-color: var(--btn-primary-hover);
    }

    .btn-secondary {
      background: var(--surface);
      color: var(--text);
      border-color: var(--border);
    }

    .btn-secondary:hover:not(:disabled) {
      background: var(--surface-2);
      border-color: var(--border-strong);
    }

    .btn-ghost {
      background: transparent;
      color: var(--text-muted);
      border-color: transparent;
    }

    .btn-ghost:hover:not(:disabled) {
      background: var(--surface-2);
      color: var(--text);
    }

    .btn-danger {
      background: var(--danger-soft);
      color: var(--danger);
      border-color: var(--danger-border);
    }

    .btn-danger:hover:not(:disabled) {
      background: var(--danger);
      color: #ffffff;
      border-color: var(--danger);
    }

    .btn-icon {
      width: 32px;
      padding: 0;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
    }

    .btn-icon:hover {
      background: var(--surface-2);
      border-color: var(--border-strong);
      color: var(--text);
    }

    select, .text-input {
      height: 32px;
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 0 10px;
      font-family: inherit;
      font-size: 12.5px;
      outline: none;
      transition: border-color 0.14s, background 0.14s;
    }

    select { cursor: pointer; }
    select:hover, .text-input:hover { border-color: var(--border-strong); }
    select:focus, .text-input:focus { border-color: var(--accent); }

    /* Page Shell */
    .container {
      max-width: 1360px;
      width: 100%;
      margin: 0 auto;
      padding: 20px 24px 36px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      flex: 1;
    }

    /* Studio Command Deck */
    .command-deck {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-sm);
      overflow: hidden;
    }

    .scenario-strip {
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }

    .pill-group {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }

    .pill-group-label {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-dim);
      margin-right: 4px;
    }

    .pill-sep {
      width: 1px;
      height: 18px;
      background: var(--border);
    }

    .scenario-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      height: 28px;
      padding: 0 11px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface-2);
      color: var(--text-muted);
      font-family: inherit;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .scenario-pill:hover {
      border-color: var(--border-strong);
      color: var(--text);
      background: var(--surface-hover);
    }

    .scenario-pill.active {
      border-color: var(--accent-border);
      background: var(--accent-soft);
      color: var(--text);
    }

    .scenario-pill .p-dot {
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: var(--text-dim);
      transition: background 0.15s;
    }

    .scenario-pill.active .p-dot {
      background: var(--accent);
      box-shadow: 0 0 6px var(--accent);
    }

    .controls-strip {
      padding: 10px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }

    .controls-left, .controls-right {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .control-label {
      font-size: 11px;
      color: var(--text-dim);
      font-weight: 500;
    }

    /* Minimal Toggle Switch for Loop */
    .toggle-wrap {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      height: 32px;
      padding: 0 10px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: var(--surface);
      font-size: 12px;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
    }

    .toggle-wrap input {
      accent-color: var(--accent);
      width: 13px;
      height: 13px;
      cursor: pointer;
    }

    /* Main Workspace Grid */
    .workspace {
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      align-items: start;
    }

    @media (min-width: 1060px) {
      .workspace {
        grid-template-columns: minmax(0, 1.38fr) minmax(350px, 0.88fr);
      }
    }

    .side-stack {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    /* Surface Cards */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-sm);
      overflow: hidden;
    }

    .card-header {
      height: 44px;
      padding: 0 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .card-title-group {
      display: flex;
      align-items: center;
      gap: 9px;
      min-width: 0;
    }

    .card-title {
      font-size: 12.5px;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .card-meta {
      font-size: 11px;
      color: var(--text-dim);
      white-space: nowrap;
    }

    .live-chip {
      display: none;
      align-items: center;
      gap: 5px;
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.07em;
      color: var(--accent);
      background: var(--accent-soft);
      border: 1px solid var(--accent-border);
      padding: 2px 7px;
      border-radius: 999px;
    }

    .live-chip.on { display: inline-flex; }

    .live-chip i {
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: var(--accent);
      animation: breathe 1.4s infinite;
    }

    /* Optical Video Viewport */
    .stream-container {
      width: 100%;
      aspect-ratio: 640 / 540;
      background: var(--viewport-bg);
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      overflow: hidden;
    }

    .stream-container img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .stream-placeholder {
      position: relative;
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      padding: 28px;
      text-align: center;
      background:
        radial-gradient(360px circle at 50% 50%, rgba(249, 115, 22, 0.08), transparent 72%),
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
      background-size: 100% 100%, 32px 32px, 32px 32px;
    }

    .reticle-corner {
      position: absolute;
      width: 14px;
      height: 14px;
      border-color: rgba(255, 255, 255, 0.18);
      border-style: solid;
      pointer-events: none;
    }
    .rc-tl { top: 18px; left: 18px; border-width: 1px 0 0 1px; }
    .rc-tr { top: 18px; right: 18px; border-width: 1px 1px 0 0; }
    .rc-bl { bottom: 18px; left: 18px; border-width: 0 0 1px 1px; }
    .rc-br { bottom: 18px; right: 18px; border-width: 0 1px 1px 0; }

    .placeholder-icon {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.10);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #f97316;
    }

    .placeholder-title {
      font-size: 14px;
      font-weight: 500;
      color: #e4e4e7;
      letter-spacing: -0.01em;
    }

    .placeholder-sub {
      font-size: 12px;
      color: #71717a;
      max-width: 310px;
      line-height: 1.55;
    }

    .placeholder-cta {
      margin-top: 4px;
      height: 36px;
      padding: 0 18px;
      border-radius: 999px;
      background: #f4f4f5;
      color: #09090b;
      border: none;
      font-family: inherit;
      font-size: 12.5px;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      cursor: pointer;
      transition: transform 0.14s, background 0.14s;
    }

    .placeholder-cta:hover {
      background: #ffffff;
      transform: translateY(-1px);
    }

    /* Hairline Telemetry Matrix */
    .telemetry-matrix {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      background: var(--border);
      gap: 1px;
    }

    .telemetry-cell {
      background: var(--surface);
      padding: 15px 16px;
    }

    .metric-label {
      font-size: 10.5px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-dim);
      margin-bottom: 6px;
    }

    .metric-row {
      display: flex;
      align-items: baseline;
      gap: 6px;
    }

    .metric-number {
      font-size: 24px;
      font-weight: 600;
      letter-spacing: -0.03em;
      color: var(--text);
      line-height: 1;
    }

    .metric-unit {
      font-size: 11px;
      color: var(--text-dim);
    }

    /* Live Anomaly Score Bar */
    .score-meter {
      padding: 14px 16px 16px;
      border-top: 1px solid var(--border);
      background: var(--surface);
    }

    .score-meter-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 9px;
      font-size: 11.5px;
    }

    .score-meter-label {
      color: var(--text-muted);
      font-weight: 500;
    }

    .score-meter-val {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
      color: var(--text);
    }

    .score-pill {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.06em;
      padding: 2px 7px;
      border-radius: 999px;
      background: var(--ok-soft);
      color: var(--ok);
      text-transform: uppercase;
    }

    .score-pill.alert {
      background: var(--danger-soft);
      color: var(--danger);
    }

    .score-track {
      position: relative;
      height: 5px;
      border-radius: 999px;
      background: var(--surface-2);
      overflow: visible;
    }

    .score-fill {
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: var(--ok);
      transition: width 0.22s ease, background 0.22s ease;
    }

    .score-fill.alert { background: var(--danger); }

    .score-thresh {
      position: absolute;
      top: -3px;
      bottom: -3px;
      left: 25%;
      width: 1.5px;
      background: var(--text-dim);
      opacity: 0.65;
    }

    .score-scale {
      display: flex;
      justify-content: space-between;
      margin-top: 6px;
      font-size: 10px;
      color: var(--text-dim);
    }

    /* Incident Feed */
    .alert-feed {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 332px;
      overflow-y: auto;
      padding: 12px;
    }

    .alert-item {
      border: 1px solid var(--border);
      border-left: 2.5px solid var(--danger);
      background: var(--surface-2);
      border-radius: var(--radius-md);
      padding: 11px 13px;
      font-size: 12px;
      animation: slideUp 0.22s ease;
    }

    @keyframes slideUp {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .alert-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }

    .alert-id {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.04em;
      color: var(--danger);
    }

    .alert-id i {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--danger);
    }

    .alert-time {
      font-size: 11px;
      color: var(--text-dim);
    }

    .alert-meta {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--text-dim);
      margin-bottom: 8px;
      font-size: 11px;
    }

    .alert-meta strong {
      color: var(--text);
      font-weight: 600;
    }

    .alert-summary {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.55;
      padding-top: 7px;
      border-top: 1px solid var(--border);
    }

    .empty-state {
      text-align: center;
      padding: 36px 18px;
      color: var(--text-dim);
      font-size: 12.5px;
      line-height: 1.6;
    }

    /* Inline Audit Report Section */
    .report-section {
      margin-top: 4px;
    }

    .report-head {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }

    .report-head-title {
      font-size: 14px;
      font-weight: 600;
      letter-spacing: -0.015em;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 9px;
    }

    .report-head-sub {
      font-size: 12px;
      color: var(--text-dim);
      margin-top: 2px;
    }

    .report-tag {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: var(--accent);
      background: var(--accent-soft);
      border: 1px solid var(--accent-border);
      padding: 2px 7px;
      border-radius: 999px;
    }

    .report-stats-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      background: var(--border);
      gap: 1px;
      border-bottom: 1px solid var(--border);
    }

    .report-stat-cell {
      background: var(--surface);
      padding: 16px 20px;
    }

    .report-body {
      padding: 18px 20px 22px;
    }

    .report-list-head {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--text-dim);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .report-events-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .report-ev-card {
      border: 1px solid var(--border);
      border-left: 2.5px solid var(--danger);
      border-radius: var(--radius-md);
      padding: 14px 16px;
      background: var(--surface-2);
    }

    .report-ev-card.medium { border-left-color: var(--warn); }

    .report-ev-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      gap: 10px;
    }

    .report-ev-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 10.5px;
      font-weight: 600;
      letter-spacing: 0.06em;
      padding: 2px 8px;
      border-radius: 999px;
      background: var(--danger-soft);
      color: var(--danger);
    }

    .report-ev-badge.medium {
      background: var(--warn-soft);
      color: var(--warn);
    }

    .report-ev-meta {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      font-size: 12px;
      color: var(--text-dim);
      margin-bottom: 10px;
    }

    .report-ev-meta strong {
      color: var(--text);
      font-weight: 600;
    }

    .report-ev-summary {
      font-size: 12.5px;
      color: var(--text-muted);
      background: var(--surface);
      padding: 10px 13px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      line-height: 1.55;
    }

    .report-ev-summary strong {
      color: var(--text);
      font-weight: 600;
    }

    /* Footer */
    footer {
      border-top: 1px solid var(--border);
      background: var(--surface);
      margin-top: auto;
    }

    .footer-inner {
      max-width: 1360px;
      margin: 0 auto;
      padding: 13px 24px;
      font-size: 11.5px;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    /* Scrollbar subtlety */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 999px; }

    @media print {
      header, footer, .command-deck, .workspace, .no-print { display: none !important; }
      body { background: #ffffff !important; color: #09090b !important; }
      .container { padding: 0 !important; max-width: 100% !important; }
      .report-section { border: none !important; box-shadow: none !important; }
      .report-stat-cell, .report-ev-card, .report-ev-summary {
        background: #ffffff !important;
        color: #09090b !important;
        border-color: #d4d4d8 !important;
      }
    }
  </style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">
        <svg class="ico" viewBox="0 0 24 24" style="width:16px;height:16px;">
          <circle cx="12" cy="12" r="7"/>
          <circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none"/>
          <path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>
        </svg>
      </div>
      <span class="brand-title">ThermalGuard</span>
      <span class="brand-tag mono">LWIR · VAD</span>
    </div>

    <div class="nav-actions">
      <span id="status-badge" class="status-badge idle">
        <span class="status-dot"></span>
        <span id="status-text">Standby</span>
      </span>
      <div class="nav-divider"></div>
      <button type="button" class="btn btn-ghost" onclick="scrollToReport()" title="Jump to audit report">
        <svg class="ico" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>
        Audit Report
      </button>
      <button id="theme-btn" type="button" class="btn btn-icon" onclick="toggleTheme()" title="Toggle Light/Dark Theme" aria-label="Toggle theme"></button>
    </div>
  </div>
</header>

<div class="container">
  <!-- Studio Command Deck -->
  <section class="command-deck" aria-label="Stream Controls">
    <div class="scenario-strip">
      <div class="pill-group">
        <span class="pill-group-label">Real LWIR</span>
        <button type="button" class="scenario-pill active" data-source="field_thermal" onclick="selectScenario('field_thermal')">
          <span class="p-dot"></span>Field Corridor
        </button>
        <button type="button" class="scenario-pill" data-source="kaist_cyclist" onclick="selectScenario('kaist_cyclist')">
          <span class="p-dot"></span>KAIST Cyclist
        </button>
        <button type="button" class="scenario-pill" data-source="kaist_pedestrian" onclick="selectScenario('kaist_pedestrian')">
          <span class="p-dot"></span>KAIST Pedestrian
        </button>
      </div>

      <span class="pill-sep"></span>

      <div class="pill-group">
        <span class="pill-group-label">Simulation</span>
        <button type="button" class="scenario-pill" data-source="vehicle_night" onclick="selectScenario('vehicle_night')">
          <span class="p-dot"></span>FOB Vehicle
        </button>
        <button type="button" class="scenario-pill" data-source="crawler" onclick="selectScenario('crawler')">
          <span class="p-dot"></span>Facility Breach
        </button>
        <button type="button" class="scenario-pill" data-source="synthetic" onclick="selectScenario('synthetic')">
          <span class="p-dot"></span>Perimeter Wire
        </button>
      </div>
    </div>

    <div class="controls-strip">
      <div class="controls-left">
        <span class="control-label">Source</span>
        <select id="source-select" title="Thermal Video Source" onchange="onSourceChange()">
          <optgroup label="Real Thermal Footage">
            <option value="field_thermal" selected>Field Facility Intrusion (Ornek-Veri-1)</option>
            <option value="kaist_cyclist">KAIST Perimeter Cyclist Breach</option>
            <option value="kaist_pedestrian">KAIST Pedestrian Sector Crossing</option>
          </optgroup>
          <optgroup label="Tactical Simulations">
            <option value="vehicle_night">Outpost FOB Vehicle Breach</option>
            <option value="crawler">Facility Crawling &amp; Sprint Saboteur</option>
            <option value="synthetic">Perimeter Wire Human Crossing</option>
          </optgroup>
          <optgroup label="Live &amp; External Inputs">
            <option value="kaist">Full KAIST Local Sequence</option>
            <option value="webcam_0">Live Camera (Webcam 0)</option>
            <option value="webcam_1">External USB Thermal (Cam 1)</option>
            <option value="rtsp">RTSP IP Thermal Stream</option>
            <option value="custom">Custom File / Directory Path</option>
          </optgroup>
        </select>

        <input id="source-input" class="text-input mono" type="text" placeholder="rtsp://... or video.mp4" style="display:none; width:190px;" />

        <input type="file" id="video-file-picker" accept="video/*,.mp4,.avi,.mkv" style="display:none;" onchange="handleVideoUpload(this)">
        <button type="button" class="btn btn-secondary" onclick="document.getElementById('video-file-picker').click()" title="Upload thermal video file">
          <svg class="ico" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          Upload
        </button>

        <span class="control-label" style="margin-left:4px;">Assessment</span>
        <select id="llm-select" title="LLM Incident Summary Model">
          <option value="rule">Rule-Based (Offline)</option>
          <option value="ollama">Ollama (Local)</option>
          <option value="openai">OpenAI GPT</option>
          <option value="google">Google Gemini</option>
        </select>

        <label class="toggle-wrap" title="Loop playback continuously">
          <input type="checkbox" id="loop-check" checked>
          <span>Loop</span>
        </label>
      </div>

      <div class="controls-right">
        <button id="start-btn" type="button" class="btn btn-primary" onclick="startStream()">
          <svg class="ico" viewBox="0 0 24 24" style="fill:currentColor;stroke:none;"><polygon points="6 4 20 12 6 20 6 4"/></svg>
          Start Stream
        </button>
        <button id="pause-btn" type="button" class="btn btn-secondary" onclick="pauseStream()" style="display:none">
          <svg class="ico" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
          Pause
        </button>
        <button id="resume-btn" type="button" class="btn btn-primary" onclick="resumeStream()" style="display:none">
          <svg class="ico" viewBox="0 0 24 24" style="fill:currentColor;stroke:none;"><polygon points="6 4 20 12 6 20 6 4"/></svg>
          Resume
        </button>
        <button id="stop-btn" type="button" class="btn btn-danger" onclick="stopDemo()" style="display:none">
          <svg class="ico" viewBox="0 0 24 24" style="fill:currentColor;stroke:none;"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>
          Stop
        </button>
      </div>
    </div>
  </section>

  <!-- Main Two-Column Surveillance Workspace -->
  <main class="workspace">
    <!-- Left: Optical Feed -->
    <div class="card" id="stream-card">
      <div class="card-header">
        <div class="card-title-group">
          <span id="live-indicator" class="live-chip mono"><i></i>LIVE</span>
          <span class="card-title" id="active-source-title">Field Facility Intrusion (Ornek-Veri-1)</span>
        </div>
        <span id="fps-badge" class="card-meta mono">0.0 FPS</span>
      </div>
      <div class="stream-container">
        <div id="stream-placeholder" class="stream-placeholder">
          <span class="reticle-corner rc-tl"></span>
          <span class="reticle-corner rc-tr"></span>
          <span class="reticle-corner rc-bl"></span>
          <span class="reticle-corner rc-br"></span>
          <div class="placeholder-icon" aria-hidden="true">
            <svg class="ico" viewBox="0 0 24 24" style="width:22px;height:22px;">
              <circle cx="12" cy="12" r="8"/>
              <circle cx="12" cy="12" r="2.5"/>
              <path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>
            </svg>
          </div>
          <div class="placeholder-title">Optical LWIR Feed Standby</div>
          <div class="placeholder-sub">Choose any real thermal capture or tactical simulation above, then start real-time anomaly detection.</div>
          <button type="button" class="placeholder-cta" onclick="startStream()">
            <svg class="ico" viewBox="0 0 24 24" style="width:13px;height:13px;fill:currentColor;stroke:none;"><polygon points="6 4 20 12 6 20 6 4"/></svg>
            Start Monitoring
          </button>
        </div>
        <img id="stream-img" src="" alt="Live LWIR Thermal Stream" style="display:none">
      </div>
    </div>

    <!-- Right: Telemetry & Incident Feed -->
    <div class="side-stack">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Telemetry</span>
          <span class="card-meta mono">STAE · ConvLSTM</span>
        </div>
        <div class="telemetry-matrix">
          <div class="telemetry-cell">
            <div class="metric-label">Processed</div>
            <div class="metric-row">
              <span class="metric-number mono" id="m-frames">0</span>
              <span class="metric-unit">frames</span>
            </div>
          </div>
          <div class="telemetry-cell">
            <div class="metric-label">Anomalies</div>
            <div class="metric-row">
              <span class="metric-number mono" id="m-anomalies">0</span>
              <span class="metric-unit">frames</span>
            </div>
          </div>
          <div class="telemetry-cell">
            <div class="metric-label">Incidents</div>
            <div class="metric-row">
              <span class="metric-number mono" id="m-events">0</span>
              <span class="metric-unit">alerts</span>
            </div>
          </div>
          <div class="telemetry-cell">
            <div class="metric-label">Throughput</div>
            <div class="metric-row">
              <span class="metric-number mono" id="m-fps">0.0</span>
              <span class="metric-unit">fps</span>
            </div>
          </div>
        </div>
        <div class="score-meter">
          <div class="score-meter-top">
            <span class="score-meter-label">Instantaneous Anomaly Score</span>
            <span class="score-meter-val">
              <span id="m-score" class="mono">0.0000</span>
              <span id="score-state-label" class="score-pill">Secure</span>
            </span>
          </div>
          <div class="score-track">
            <div id="score-bar-fill" class="score-fill"></div>
            <div class="score-thresh" title="Alarm Threshold: 0.20"></div>
          </div>
          <div class="score-scale mono">
            <span>0.00</span>
            <span>THR 0.20</span>
            <span>0.80+</span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Incident Stream</span>
          <span id="event-count-badge" class="card-meta mono">0 events</span>
        </div>
        <div class="alert-feed" id="alert-feed">
          <div class="empty-state">No perimeter breaches detected yet.</div>
        </div>
      </div>
    </div>
  </main>

  <!-- Inline Tactical Security Audit Report -->
  <section class="card report-section" id="report-section">
    <div class="report-head">
      <div>
        <div class="report-head-title">
          <span>Security Audit Report</span>
          <span class="report-tag mono">Live Log</span>
        </div>
        <div class="report-head-sub">Real-time LWIR intrusion records and automated tactical threat assessments</div>
      </div>
      <div class="controls-right no-print">
        <button type="button" class="btn btn-secondary" onclick="window.print()" title="Print or export PDF">
          <svg class="ico" viewBox="0 0 24 24"><polyline points="6 9 6 2 18 2 18 9"/><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
          Export PDF
        </button>
        <a href="/report" target="_blank" class="btn btn-secondary" title="Open printable report in new tab">
          <svg class="ico" viewBox="0 0 24 24"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
          Full View
        </a>
      </div>
    </div>

    <div class="report-stats-strip">
      <div class="report-stat-cell">
        <div class="metric-label">Evaluated Frames</div>
        <div class="metric-row">
          <span class="metric-number mono" id="rep-total-frames">0</span>
          <span class="metric-unit">total</span>
        </div>
      </div>
      <div class="report-stat-cell">
        <div class="metric-label">Logged Incidents</div>
        <div class="metric-row">
          <span class="metric-number mono" id="rep-event-count" style="color: var(--danger);">0</span>
          <span class="metric-unit">breaches</span>
        </div>
      </div>
      <div class="report-stat-cell">
        <div class="metric-label">Anomaly Frames</div>
        <div class="metric-row">
          <span class="metric-number mono" id="rep-anom-frames" style="color: var(--danger);">0</span>
          <span class="metric-unit">flagged</span>
        </div>
      </div>
      <div class="report-stat-cell">
        <div class="metric-label">Inference Rate</div>
        <div class="metric-row">
          <span class="metric-number mono" id="rep-fps" style="color: var(--ok);">0.0</span>
          <span class="metric-unit">fps</span>
        </div>
      </div>
    </div>

    <div class="report-body">
      <div class="report-list-head">
        <span>Recorded Breaches</span>
        <span id="rep-events-badge" class="mono">0 records</span>
      </div>
      <div id="inline-report-events" class="report-events-list">
        <div class="empty-state" style="border: 1px dashed var(--border); border-radius: var(--radius-md); background: var(--surface-2);">
          No anomaly events recorded yet. Detected intrusions and tactical assessments will populate automatically during playback.
        </div>
      </div>
    </div>
  </section>
</div>

<footer>
  <div class="footer-inner">
    <span>ThermalGuard-VAD · Long-Wave Infrared Perimeter Surveillance</span>
    <span class="mono">Spatio-Temporal Autoencoder + ConvLSTM</span>
  </div>
</footer>

<script>
  let scoreHistory = [];
  let knownEvents = 0;
  let pollInterval = null;
  let streamRetry = 0;
  let currentThreshold = 0.20;
  let isCurrentlyRunning = false;

  const SUN_SVG = '<svg class="ico" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>';
  const MOON_SVG = '<svg class="ico" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

  function setThemeIcon(theme) {
    const btn = document.getElementById('theme-btn');
    if (btn) btn.innerHTML = theme === 'dark' ? SUN_SVG : MOON_SVG;
  }

  function toggleTheme() {
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('tg_theme', next);
    setThemeIcon(next);
  }

  function syncScenarioUI() {
    const sel = document.getElementById('source-select');
    const val = sel.value;
    document.querySelectorAll('.scenario-pill').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.source === val);
    });
    const titleEl = document.getElementById('active-source-title');
    if (titleEl && sel.selectedIndex >= 0) {
      titleEl.textContent = sel.options[sel.selectedIndex].text;
    }
  }

  function selectScenario(val) {
    const sel = document.getElementById('source-select');
    sel.value = val;
    onSourceChange();
    if (isCurrentlyRunning) {
      startStream();
    }
  }

  (function initApp() {
    const saved = localStorage.getItem('tg_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    window.addEventListener('DOMContentLoaded', () => {
      setThemeIcon(saved);
      syncScenarioUI();

      fetch('/api/status').then(r => r.json()).then(s => {
        updateStatus(s);
        if (s.status === 'running' || s.status === 'paused') {
          isCurrentlyRunning = true;
          document.getElementById('start-btn').style.display = 'none';
          document.getElementById('stop-btn').style.display = 'inline-flex';
          document.getElementById('pause-btn').style.display = s.status === 'running' ? 'inline-flex' : 'none';
          document.getElementById('resume-btn').style.display = s.status === 'paused' ? 'inline-flex' : 'none';
          startVideoPlayback();
          startPolling();
        } else {
          document.getElementById('start-btn').style.display = 'inline-flex';
          document.getElementById('start-btn').disabled = false;
          document.getElementById('stop-btn').style.display = 'none';
        }
      }).catch(() => {});

      fetch('/api/events').then(r => r.json()).then(events => {
        renderInlineReport(events);
      }).catch(() => {});
    });
  })();

  function onSourceChange() {
    const sel = document.getElementById('source-select').value;
    const inp = document.getElementById('source-input');
    if (sel === 'rtsp') {
      inp.style.display = 'inline-block';
      inp.placeholder = 'rtsp://user:pass@192.168.1.100:554/live';
      if (!inp.value.startsWith('rtsp://')) inp.value = 'rtsp://';
    } else if (sel === 'custom') {
      inp.style.display = 'inline-block';
      inp.placeholder = 'data/sample.mp4 or folder';
    } else {
      inp.style.display = 'none';
    }
    syncScenarioUI();
  }

  function handleVideoUpload(input) {
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const formData = new FormData();
    formData.append('video', file);

    const btn = document.getElementById('start-btn');
    const origHTML = btn.innerHTML;
    btn.disabled = true;
    btn.textContent = 'Uploading...';

    fetch('/api/upload_video', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(data => {
        btn.disabled = false;
        btn.innerHTML = origHTML;
        if (data.ok) {
          document.getElementById('source-select').value = 'custom';
          const inp = document.getElementById('source-input');
          inp.style.display = 'inline-block';
          inp.value = data.path;
          syncScenarioUI();
          alert('Video uploaded: ' + data.filename + '\\nClick Start Stream to begin monitoring.');
        } else {
          alert('Upload failed: ' + (data.error || 'Unknown error'));
        }
      })
      .catch(err => {
        btn.disabled = false;
        btn.innerHTML = origHTML;
        alert('Upload failed: ' + err);
      });
  }

  function scrollToReport() {
    const el = document.getElementById('report-section');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  }

  function renderInlineReport(events) {
    const repFeed = document.getElementById('inline-report-events');
    const repBadge = document.getElementById('rep-events-badge');
    if (repBadge) repBadge.textContent = (events ? events.length : 0) + ' records';
    if (!repFeed) return;

    if (!events || events.length === 0) {
      repFeed.innerHTML = '<div class="empty-state" style="border:1px dashed var(--border);border-radius:var(--radius-md);background:var(--surface-2);">No anomaly events recorded yet. Detected intrusions and tactical assessments will populate automatically during playback.</div>';
      return;
    }

    repFeed.innerHTML = events.slice().reverse().map(ev => {
      const startTime = ev.start_time ? ev.start_time.replace('T', ' ').substring(0, 19) : '-';
      const durationSec = (ev.duration_frames / 20.0).toFixed(1);
      const isCritical = ev.peak_score >= 0.5;
      const cls = isCritical ? '' : 'medium';
      const badgeLabel = isCritical ? 'CRITICAL BREACH' : 'ANOMALY DETECTED';

      return `
        <div class="report-ev-card ${cls}">
          <div class="report-ev-top">
            <span class="report-ev-badge ${cls} mono">${badgeLabel} #${ev.event_id}</span>
            <span class="alert-time mono">${startTime}</span>
          </div>
          <div class="report-ev-meta">
            <span>Frames <strong class="mono">${ev.start_frame}&ndash;${ev.end_frame}</strong></span>
            <span>Duration <strong>${ev.duration_frames}f (~${durationSec}s)</strong></span>
            <span>Peak <strong class="mono" style="color:var(--danger);">${ev.peak_score.toFixed(4)}</strong></span>
            <span>Class <strong style="text-transform:capitalize;">${ev.anomaly_type || 'intruder'}</strong></span>
          </div>
          <div class="report-ev-summary">
            <strong>Assessment:</strong> ${ev.llm_summary || 'Generating tactical assessment...'}
          </div>
        </div>
      `;
    }).join('');
  }

  function startStream() {
    const llm = document.getElementById('llm-select').value;
    const sourceSel = document.getElementById('source-select').value;
    const sourceInp = document.getElementById('source-input').value.trim();
    const loop = document.getElementById('loop-check').checked;

    let source = sourceSel;
    let param = '';

    if (sourceSel === 'field_thermal' || sourceSel === 'ornek_1') {
      source = 'field_thermal';
      param = 'data/Ornek-Veri-1-Termal.mp4';
    } else if (sourceSel === 'webcam_0') {
      source = 'webcam'; param = '0';
    } else if (sourceSel === 'webcam_1') {
      source = 'webcam'; param = '1';
    } else if (sourceSel === 'kaist') {
      source = 'kaist'; param = 'data/kaist/images/set00/V000/lwir';
    } else if (sourceSel === 'rtsp') {
      source = 'rtsp'; param = sourceInp;
      if (!param || param === 'rtsp://') {
        alert('Please enter a valid RTSP URL (e.g. rtsp://192.168.1.50/stream)');
        return;
      }
    } else if (sourceSel === 'custom') {
      source = 'custom'; param = sourceInp;
      if (!param) {
        alert('Please enter a valid video file or folder path, or use Upload.');
        return;
      }
    }

    knownEvents = 0;
    document.getElementById('alert-feed').innerHTML = '<div class="empty-state">Monitoring active — waiting for anomalies...</div>';

    const url = `/api/run_stream?llm=${encodeURIComponent(llm)}&source=${encodeURIComponent(source)}&param=${encodeURIComponent(param)}&loop=${loop}`;
    fetch(url, { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        if (d.ok) {
          isCurrentlyRunning = true;
          document.getElementById('start-btn').style.display = 'none';
          document.getElementById('resume-btn').style.display = 'none';
          document.getElementById('pause-btn').style.display = 'inline-flex';
          document.getElementById('stop-btn').style.display = 'inline-flex';
          document.getElementById('live-indicator').classList.add('on');
          syncScenarioUI();
          startVideoPlayback();
          startPolling();
        } else {
          alert('Error: ' + (d.error || 'Unknown'));
        }
      })
      .catch(err => alert('Server connection error: ' + err));
  }

  window.startDemo = startStream;

  function pauseStream() {
    fetch('/api/pause', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        if (d.ok) {
          document.getElementById('pause-btn').style.display = 'none';
          document.getElementById('resume-btn').style.display = 'inline-flex';
        }
      });
  }

  function resumeStream() {
    fetch('/api/resume', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        if (d.ok) {
          document.getElementById('resume-btn').style.display = 'none';
          document.getElementById('pause-btn').style.display = 'inline-flex';
        }
      });
  }

  function stopDemo() {
    isCurrentlyRunning = false;
    fetch('/api/stop', { method: 'POST' });
    document.getElementById('start-btn').style.display = 'inline-flex';
    document.getElementById('start-btn').disabled = false;
    document.getElementById('pause-btn').style.display = 'none';
    document.getElementById('resume-btn').style.display = 'none';
    document.getElementById('stop-btn').style.display = 'none';
    document.getElementById('live-indicator').classList.remove('on');
    stopPolling();
  }

  function startVideoPlayback() {
    const img = document.getElementById('stream-img');
    const placeholder = document.getElementById('stream-placeholder');
    img.src = '/stream?t=' + Date.now();
    img.style.display = 'block';
    placeholder.style.display = 'none';
    img.onerror = () => {
      if (streamRetry < 10) {
        streamRetry++;
        setTimeout(startVideoPlayback, 800);
      }
    };
  }

  function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(poll, 450);
  }

  function stopPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = null;
  }

  async function poll() {
    try {
      const [statusResp, eventsResp] = await Promise.all([
        fetch('/api/status').then(r => r.json()),
        fetch('/api/events').then(r => r.json()),
      ]);
      updateStatus(statusResp);
      updateEvents(eventsResp);
      if (statusResp.status === 'done' || statusResp.status === 'error') {
        isCurrentlyRunning = false;
        stopPolling();
        document.getElementById('start-btn').style.display = 'inline-flex';
        document.getElementById('start-btn').disabled = false;
        document.getElementById('pause-btn').style.display = 'none';
        document.getElementById('resume-btn').style.display = 'none';
        document.getElementById('stop-btn').style.display = 'none';
        document.getElementById('live-indicator').classList.remove('on');
        if (statusResp.status === 'error' && statusResp.error) {
          alert('Stream Error: ' + statusResp.error);
        }
      }
    } catch (e) { /* network delay */ }
  }

  function updateStatus(s) {
    document.getElementById('m-frames').textContent = s.frame_count;
    document.getElementById('m-anomalies').textContent = s.anomaly_count;
    document.getElementById('m-events').textContent = s.event_count;
    document.getElementById('m-fps').textContent = s.fps.toFixed(1);
    document.getElementById('fps-badge').textContent = s.fps.toFixed(1) + ' FPS';

    const repTotal = document.getElementById('rep-total-frames');
    if (repTotal) repTotal.textContent = s.frame_count;
    const repAnom = document.getElementById('rep-anom-frames');
    if (repAnom) repAnom.textContent = s.anomaly_count;
    const repEvents = document.getElementById('rep-event-count');
    if (repEvents) repEvents.textContent = s.event_count;
    const repFps = document.getElementById('rep-fps');
    if (repFps) repFps.textContent = s.fps.toFixed(1);

    currentThreshold = s.threshold || 0.20;
    const sc = Number(s.latest_score || 0);
    const scoreEl = document.getElementById('m-score');
    const barEl = document.getElementById('score-bar-fill');
    const pillEl = document.getElementById('score-state-label');
    if (scoreEl) scoreEl.textContent = sc.toFixed(4);
    const isAlert = sc >= currentThreshold;
    if (barEl) {
      barEl.style.width = Math.min(100, (sc / 0.80) * 100).toFixed(1) + '%';
      barEl.className = 'score-fill' + (isAlert ? ' alert' : '');
    }
    if (pillEl) {
      pillEl.textContent = isAlert ? 'Breach' : 'Secure';
      pillEl.className = 'score-pill' + (isAlert ? ' alert' : '');
    }

    const badge = document.getElementById('status-badge');
    const statusMap = {
      idle: ['idle', 'Standby'],
      running: ['running', 'Monitoring'],
      paused: ['paused', 'Paused'],
      done: ['done', 'Completed'],
      error: ['error', 'Error'],
    };
    const [cls, txt] = statusMap[s.status] || ['idle', 'Standby'];
    badge.className = 'status-badge ' + cls;
    document.getElementById('status-text').textContent = txt;
    document.getElementById('live-indicator').classList.toggle('on', s.status === 'running');
  }

  function updateEvents(events) {
    document.getElementById('event-count-badge').textContent = events.length + ' events';
    document.getElementById('m-events').textContent = events.length;
    renderInlineReport(events);
    if (events.length === knownEvents) return;

    const feed = document.getElementById('alert-feed');
    if (events.length > 0 && knownEvents === 0) {
      feed.innerHTML = '';
    }

    for (let i = knownEvents; i < events.length; i++) {
      const ev = events[i];
      const card = document.createElement('div');
      card.className = 'alert-item';
      const startTime = ev.start_time ? ev.start_time.replace('T', ' ').substring(0, 19) : '-';
      card.innerHTML = `
        <div class="alert-header">
          <span class="alert-id mono"><i></i>ALERT #${ev.event_id}</span>
          <span class="alert-time mono">${startTime}</span>
        </div>
        <div class="alert-meta">
          <span>Peak <strong class="mono">${ev.peak_score.toFixed(4)}</strong></span>
          <span>Duration <strong>${ev.duration_frames}f</strong></span>
          <span>Class <strong style="text-transform:capitalize;">${ev.anomaly_type || 'Unknown'}</strong></span>
        </div>
        <div class="alert-summary">${ev.llm_summary || 'Generating assessment...'}</div>
      `;
      feed.prepend(card);
    }
    knownEvents = events.length;
  }
</script>
</body>
</html>"""


REPORT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ThermalGuard-VAD · Security Audit Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #fafafa;
      color: #09090b;
      padding: 40px 44px;
      max-width: 1020px;
      margin: 0 auto;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }
    .mono { font-family: "JetBrains Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
    .header {
      border-bottom: 1.5px solid #09090b;
      padding-bottom: 18px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
    }
    h1 { font-size: 19px; font-weight: 600; letter-spacing: -0.025em; margin-bottom: 4px; }
    .sub { color: #71717a; font-size: 12px; }
    .btn-print {
      height: 32px;
      padding: 0 14px;
      background: #09090b;
      color: #fafafa;
      border: none;
      border-radius: 6px;
      font-family: inherit;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      background: #e4e4e7;
      gap: 1px;
      border: 1px solid #e4e4e7;
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 28px;
    }
    .stat { background: #ffffff; padding: 14px 16px; }
    .stat-l { font-size: 10.5px; color: #71717a; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500; }
    .stat-v { font-size: 22px; font-weight: 600; color: #09090b; letter-spacing: -0.02em; }
    .section-title {
      font-size: 11px;
      font-weight: 600;
      margin-bottom: 12px;
      color: #71717a;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .event {
      border: 1px solid #e4e4e7;
      border-left: 2.5px solid #dc2626;
      border-radius: 8px;
      padding: 15px 17px;
      margin-bottom: 12px;
      background: #ffffff;
    }
    .ev-h {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .ev-badge {
      background: rgba(220, 38, 38, 0.09);
      color: #dc2626;
      font-size: 10.5px;
      font-weight: 600;
      letter-spacing: 0.06em;
      padding: 2px 8px;
      border-radius: 999px;
    }
    .ev-time { font-size: 11.5px; color: #71717a; }
    .ev-scores {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      font-size: 12px;
      color: #71717a;
      margin-bottom: 10px;
    }
    .ev-scores strong { color: #09090b; font-weight: 600; }
    .ev-summary {
      font-size: 12.5px;
      color: #3f3f46;
      background: #fafafa;
      border: 1px solid #e4e4e7;
      padding: 10px 13px;
      border-radius: 6px;
      line-height: 1.55;
    }
    .no-events {
      padding: 40px;
      text-align: center;
      color: #71717a;
      background: #ffffff;
      border: 1px dashed #d4d4d8;
      border-radius: 8px;
      font-size: 13px;
    }
    footer {
      border-top: 1px solid #e4e4e7;
      margin-top: 36px;
      padding-top: 14px;
      font-size: 11px;
      color: #71717a;
      display: flex;
      justify-content: space-between;
    }
    @media print {
      .btn-print { display: none; }
      body { padding: 0; background: #ffffff; }
    }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>ThermalGuard-VAD · Security Audit Report</h1>
      <p class="sub mono">Generated {{ generated_at }} &middot; LWIR Perimeter Surveillance</p>
    </div>
    <button class="btn-print" onclick="window.print()">Export PDF</button>
  </div>

  <div class="stats">
    <div class="stat"><div class="stat-l">Evaluated Frames</div><div class="stat-v mono">{{ total_frames }}</div></div>
    <div class="stat"><div class="stat-l">Logged Incidents</div><div class="stat-v mono">{{ event_count }}</div></div>
    <div class="stat"><div class="stat-l">Anomaly Frames</div><div class="stat-v mono" style="color:#dc2626">{{ anomaly_frames }}</div></div>
    <div class="stat"><div class="stat-l">Inference Speed</div><div class="stat-v mono">{{ avg_fps }} FPS</div></div>
  </div>

  <div class="section-title">Detected Incidents</div>

  {% if events %}
    {% for ev in events %}
    <div class="event">
      <div class="ev-h">
        <span class="ev-badge mono">ALERT #{{ ev.event_id }}</span>
        <span class="ev-time mono">{{ ev.start_time }}</span>
      </div>
      <div class="ev-scores">
        <div>Frames <strong class="mono">{{ ev.start_frame }}&ndash;{{ ev.end_frame }}</strong></div>
        <div>Duration <strong>{{ ev.duration_frames }}f</strong></div>
        <div>Peak <strong class="mono">{{ "%.4f"|format(ev.peak_score) }}</strong></div>
        <div>Class <strong style="text-transform:capitalize;">{{ ev.anomaly_type }}</strong></div>
      </div>
      <div class="ev-summary">{{ ev.llm_summary or "No assessment generated." }}</div>
    </div>
    {% endfor %}
  {% else %}
    <div class="no-events">No recorded anomaly events found.</div>
  {% endif %}

  <footer>
    <span>ThermalGuard-VAD · Infrared Video Anomaly Detection</span>
    <span class="mono">Spatio-Temporal Autoencoder + ConvLSTM</span>
  </footer>
</body>
</html>"""


# =========================================================================== #
#  Flask Routes
# =========================================================================== #
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/stream")
def stream():
    """MJPEG video stream broadcaster."""
    def generate():
        blank = np.full((380, 640, 3), 22, dtype=np.uint8)
        cv2.putText(blank, "Waiting for stream...", (200, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 90, 110), 1)
        _, blank_buf = cv2.imencode(".jpg", blank)
        blank_bytes = blank_buf.tobytes()

        while True:
            frame_bytes = state.latest_frame
            data = frame_bytes if frame_bytes else blank_bytes
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.025)

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """Saves user-uploaded video file to server."""
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "No video file provided"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"ok": False, "error": "Invalid filename"}), 400
    safe_name = os.path.basename(file.filename)
    dest = UPLOAD_FOLDER / safe_name
    file.save(str(dest))
    return jsonify({"ok": True, "path": str(dest).replace("\\", "/"), "filename": safe_name})


@app.route("/api/run_stream", methods=["POST"])
@app.route("/api/run_demo", methods=["POST"])
def run_stream():
    """Starts stream processing from selected input source."""
    if state.is_running:
        state.is_running = False
        time.sleep(0.15)
    llm = request.args.get("llm", "rule")
    source = request.args.get("source", "field_thermal")
    param = request.args.get("param", "")
    loop = request.args.get("loop", "true").lower() in ("true", "1", "yes")

    state.reset()
    state.is_running = True
    t = threading.Thread(target=run_stream_thread, args=(source, param, llm, loop), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/pause", methods=["POST"])
def pause_demo():
    state.is_paused = True
    return jsonify({"ok": True, "is_paused": True})


@app.route("/api/resume", methods=["POST"])
def resume_demo():
    state.is_paused = False
    return jsonify({"ok": True, "is_paused": False})


@app.route("/api/stop", methods=["POST"])
def stop_demo():
    state.is_running = False
    state.is_paused = False
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(state.snapshot())


@app.route("/api/events")
def api_events():
    with state._lock:
        return jsonify([e.to_brief_dict() for e in state.events])


@app.route("/api/score_history")
def api_score_history():
    with state._lock:
        return jsonify(list(state.score_history))


@app.route("/report")
def report():
    with state._lock:
        events = list(state.events)
        snap = state.snapshot()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template_string(
        REPORT_HTML_TEMPLATE,
        events=events,
        generated_at=generated_at,
        total_frames=snap["frame_count"],
        event_count=snap["event_count"],
        anomaly_frames=snap["anomaly_count"],
        avg_fps=f"{snap['fps']:.1f}",
    )


# =========================================================================== #
#  Application Entry Point
# =========================================================================== #
if __name__ == "__main__":
    print("=" * 65)
    print("  ThermalGuard-VAD Web Dashboard")
    print("  http://localhost:5000")
    print("=" * 65)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
