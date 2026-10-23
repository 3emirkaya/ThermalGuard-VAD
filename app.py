"""
ThermalGuard-VAD Web Dashboard Uygulaması.

Flask tabanlı gerçek zamanlı anomali izleme ve raporlama arayüzü.

Özellikler:
  * / → Ana dashboard (canlı iframe + grafik + alert listesi)
  * /api/events → Tüm olayların JSON listesi
  * /api/status → Sistem durumu ve istatistikler
  * /api/run_demo → Arka planda synthetic demo başlat
  * /stream → Server-Sent Events ile canlı kare yayını (MJPEG)
  * /report → HTML anomali raporu (LLM özetleri dahil)

Kullanım:
    python app.py
    # → http://localhost:5000
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

import cv2
import numpy as np

# Flask kontrol et, yoksa hata ver
try:
    from flask import Flask, Response, jsonify, render_template_string, stream_with_context
except ImportError:
    print("[!] Flask kurulu değil. Kurmak için: pip install flask")
    sys.exit(1)

# Proje kökünü yola ekle
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Hafif importlar (torch gerektirmez)
from thermalguard.reporting.logger import AnomalyEvent, EventLogger
from thermalguard.reporting.llm_summary import LLMSummarizer


# =========================================================================== #
#  Global Uygulama Durumu
# =========================================================================== #
class AppState:
    """Thread-safe paylaşılan uygulama durumu."""

    def __init__(self):
        self._lock = threading.Lock()
        self.is_running: bool = False
        self.latest_frame: Optional[bytes] = None   # JPEG bytes
        self.events: List[AnomalyEvent] = []
        self.frame_count: int = 0
        self.anomaly_count: int = 0
        self.latest_score: float = 0.0
        self.fps: float = 0.0
        self.status: str = "idle"                   # 'idle' | 'running' | 'done' | 'error'
        self.score_history: List[dict] = []          # [{frame, score, is_anomaly}]
        self.threshold: float = 0.20
        self.error_msg: str = ""

    def update_frame(self, frame_bgr: np.ndarray, result):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self._lock:
            if ok:
                self.latest_frame = buf.tobytes()
            self.frame_count = result.frame_idx + 1
            self.latest_score = result.anomaly_score
            self.fps = result.fps
            if result.is_anomaly:
                self.anomaly_count += 1
            self.score_history.append({
                "frame": result.frame_idx,
                "score": round(result.anomaly_score, 5),
                "is_anomaly": result.is_anomaly,
            })
            # Son 300 kareyi tut
            if len(self.score_history) > 300:
                self.score_history = self.score_history[-300:]

    def add_event(self, event: AnomalyEvent):
        with self._lock:
            self.events.append(event)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
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
#  Arka Plan Çalışma Thread'i
# =========================================================================== #
def run_demo_thread(llm_backend: str = "rule"):
    """Synthetic demo'yu arka planda çalıştırır."""
    state.status = "running"
    state.frame_count = 0
    state.anomaly_count = 0
    state.events = []
    state.score_history = []

    try:
        # Lazy import: torch sadece demo başladığında yüklenir
        from thermalguard.data.synthetic import AnomalyEvent as SimAnomalyEvent, ThermalPerimeterSimulator
        from thermalguard.engine.anomaly import AnomalyScoreConfig, AnomalyScorer
        from thermalguard.engine.infer import StreamingInferenceEngine
        from thermalguard.models.stae import build_model
        from thermalguard.preprocessing.thermal import ThermalPreprocessConfig, ThermalPreprocessor
        from thermalguard.visualization.overlay import DashboardVisualizer

        # Model
        device_str = "cuda" if _torch_cuda_available() else "cpu"
        model = build_model({
            "widths": (16, 32, 64, 96), "latent": 64,
            "lstm_hidden": (64, 64), "objective": "reconstruct",
        })

        prep_cfg = ThermalPreprocessConfig()
        preprocessor = ThermalPreprocessor(prep_cfg)

        has_weights = os.path.isfile("weights/stae_thermal_best.pt")
        scorer = AnomalyScorer(AnomalyScoreConfig(percentile=99.2, temporal_smoothing=0.6))
        engine = StreamingInferenceEngine(
            model=model, preprocessor=preprocessor, scorer=scorer,
            clip_len=8, threshold=state.threshold, device=device_str,
            has_weights=has_weights,
        )

        visualizer = DashboardVisualizer(
            width=640, height=540, graph_height=170, threshold=state.threshold,
        )

        # Logger + LLM
        log_dir = Path("logs")
        logger = EventLogger(
            log_dir=str(log_dir),
            persist_frames=3,
            max_gap_frames=15,
        )
        summarizer = LLMSummarizer(backend=llm_backend, fps=20.0)

        def on_event(ev: AnomalyEvent):
            summary = summarizer.summarize_and_attach(ev, logger)
            print(f"\n[ALARM] Olay #{ev.event_id}: {summary}")
            state.add_event(ev)

        logger._callback = on_event

        # Simülatör
        sim = ThermalPerimeterSimulator(width=160, height=128, num_frames=200, seed=42)
        anomalies = [
            SimAnomalyEvent(
                start_frame=60, end_frame=135,
                anomaly_type="intruder",
                start_pos=(0.10, 0.65), end_pos=(0.90, 0.68),
                size=(12, 24), intensity=240.0,
            )
        ]
        raw_frames, gt_labels, _ = sim.generate_sequence(anomalies=anomalies, add_sensor_defects=True)

        for i, raw_frame in enumerate(raw_frames):
            if not state.is_running:
                break

            res = engine.step(raw_frame)
            anomaly_type = anomalies[0].anomaly_type if 60 <= i <= 135 else "normal"

            logger.update(
                frame_idx=res.frame_idx,
                is_anomaly=res.is_anomaly,
                anomaly_score=res.anomaly_score,
                anomaly_type="intruder" if 60 <= i <= 135 else "unknown",
            )

            hud = visualizer.render_frame(res)
            state.update_frame(hud, res)
            time.sleep(0.05)  # ~20 FPS

        logger.finalize()
        state.status = "done"
        state.is_running = False

    except Exception as exc:
        state.status = "error"
        state.error_msg = str(exc)
        state.is_running = False
        import traceback
        traceback.print_exc()


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# =========================================================================== #
#  HTML Şablonları
# =========================================================================== #
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="ThermalGuard-VAD — Infrared Video Anomaly Detection System">
  <title>ThermalGuard-VAD | Thermal Anomaly Monitoring</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --card-subtle: #f1f5f9;
      --border: #e2e8f0;
      --border-focus: #94a3b8;
      --text: #0f172a;
      --text-muted: #64748b;
      --text-subtle: #94a3b8;
      --btn-primary: #0f172a;
      --btn-primary-hover: #1e293b;
      --btn-primary-text: #ffffff;
      --accent-blue: #2563eb;
      --accent-red: #dc2626;
      --accent-red-bg: #fef2f2;
      --accent-red-border: #fecaca;
      --accent-green: #16a34a;
      --accent-green-bg: #f0fdf4;
      --accent-green-border: #bbf7d0;
    }

    [data-theme="dark"] {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-subtle: #0f172a;
      --border: #334155;
      --border-focus: #64748b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --text-subtle: #64748b;
      --btn-primary: #f8fafc;
      --btn-primary-hover: #e2e8f0;
      --btn-primary-text: #0f172a;
      --accent-blue: #3b82f6;
      --accent-red: #ef4444;
      --accent-red-bg: #451a1a;
      --accent-red-border: #7f1d1d;
      --accent-green: #22c55e;
      --accent-green-bg: #143521;
      --accent-green-border: #166534;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Header */
    header {
      background: var(--card-bg);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
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

    .brand-title {
      font-size: 17px;
      font-weight: 700;
      color: var(--text);
      letter-spacing: -0.02em;
    }

    .brand-badge {
      font-size: 11px;
      font-weight: 600;
      background: var(--card-subtle);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 2px 7px;
      border-radius: 4px;
      letter-spacing: 0.05em;
    }

    .brand-sub {
      font-size: 12px;
      color: var(--text-muted);
      display: none;
    }

    @media (min-width: 768px) {
      .brand-sub { display: inline; border-left: 1px solid var(--border); padding-left: 10px; }
    }

    .header-controls {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    /* Status Pill */
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 500;
      border: 1px solid var(--border);
      background: var(--card-subtle);
      color: var(--text-muted);
    }

    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
    }

    .status-badge.idle { color: var(--text-muted); }
    .status-badge.running {
      color: var(--accent-green);
      background: var(--accent-green-bg);
      border-color: var(--accent-green-border);
    }
    .status-badge.done {
      color: var(--accent-blue);
      border-color: var(--border);
    }
    .status-badge.error {
      color: var(--accent-red);
      background: var(--accent-red-bg);
      border-color: var(--accent-red-border);
    }

    /* Buttons & Inputs */
    select {
      background: var(--card-bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
      outline: none;
      cursor: pointer;
    }
    select:focus { border-color: var(--border-focus); }

    .btn {
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      border: 1px solid transparent;
      outline: none;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: background-color 0.15s ease, opacity 0.15s ease;
    }

    .btn-primary {
      background: var(--btn-primary);
      color: var(--btn-primary-text);
    }
    .btn-primary:hover { opacity: 0.9; }
    .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

    .btn-danger {
      background: var(--accent-red);
      color: #ffffff;
    }
    .btn-danger:hover { opacity: 0.9; }

    .btn-secondary {
      background: var(--card-bg);
      color: var(--text);
      border-color: var(--border);
    }
    .btn-secondary:hover { background: var(--card-subtle); }

    .btn-icon {
      padding: 6px 9px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
    }
    .btn-icon:hover { color: var(--text); background: var(--card-subtle); }

    /* Layout */
    main {
      flex: 1;
      max-width: 1500px;
      width: 100%;
      margin: 0 auto;
      padding: 20px 24px;
      display: grid;
      grid-template-columns: 1fr 360px;
      gap: 20px;
    }

    @media (max-width: 1024px) {
      main { grid-template-columns: 1fr; }
    }

    .column-left { display: flex; flex-direction: column; gap: 18px; }
    .column-right { display: flex; flex-direction: column; gap: 18px; }

    /* Cards */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }

    .card-header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .card-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
      letter-spacing: -0.01em;
    }

    .card-subtitle {
      font-size: 11px;
      color: var(--text-muted);
      font-family: monospace;
    }

    .card-body { padding: 16px; }

    /* Stream */
    .stream-container {
      width: 100%;
      background: #090d16;
      border-radius: 0 0 7px 7px;
      min-height: 380px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
    }

    #stream-img {
      max-width: 100%;
      max-height: 520px;
      display: block;
      object-fit: contain;
    }

    .stream-placeholder {
      text-align: center;
      color: #64748b;
      padding: 48px 20px;
    }
    .stream-placeholder p { font-size: 13px; margin-top: 6px; }

    /* Score Graph */
    #score-canvas {
      width: 100%;
      height: 140px;
      display: block;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--card-bg);
    }

    .score-meter {
      margin-top: 12px;
    }

    .score-meter-header {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 6px;
      color: var(--text-muted);
    }

    .score-value {
      font-family: monospace;
      font-weight: 600;
      color: var(--text);
    }

    .score-bar {
      height: 6px;
      background: var(--card-subtle);
      border-radius: 3px;
      overflow: hidden;
      border: 1px solid var(--border);
    }

    .score-bar-fill {
      height: 100%;
      width: 0%;
      background: var(--accent-blue);
      transition: width 0.2s ease, background 0.2s ease;
    }

    /* Metrics Grid */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }

    .metric-box {
      background: var(--card-subtle);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
    }

    .metric-label {
      font-size: 11px;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 4px;
    }

    .metric-number {
      font-size: 20px;
      font-weight: 700;
      color: var(--text);
      font-family: monospace;
      letter-spacing: -0.02em;
    }

    .metric-unit {
      font-size: 11px;
      color: var(--text-subtle);
    }

    /* Alert Feed */
    .alert-feed {
      max-height: 400px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .alert-item {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent-red);
      border-radius: 6px;
      padding: 12px;
    }

    .alert-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 6px;
    }

    .alert-badge {
      font-size: 11px;
      font-weight: 600;
      background: var(--accent-red-bg);
      border: 1px solid var(--accent-red-border);
      color: var(--accent-red);
      padding: 1px 7px;
      border-radius: 4px;
    }

    .alert-time {
      font-size: 11px;
      color: var(--text-muted);
      font-family: monospace;
    }

    .alert-meta {
      display: flex;
      gap: 12px;
      font-size: 11px;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .alert-meta strong { color: var(--text); }

    .alert-summary {
      font-size: 12px;
      color: var(--text);
      line-height: 1.5;
      background: var(--card-subtle);
      padding: 8px 10px;
      border-radius: 4px;
    }

    .no-events {
      text-align: center;
      padding: 32px 16px;
      color: var(--text-muted);
      font-size: 13px;
    }

    /* Footer */
    footer {
      border-top: 1px solid var(--border);
      background: var(--card-bg);
      padding: 12px 24px;
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: auto;
    }
  </style>
</head>
<body>

<header>
  <div class="brand">
    <span class="brand-title">ThermalGuard</span>
    <span class="brand-badge">VAD</span>
    <span class="brand-sub">Infrared Video Anomaly Detection System</span>
  </div>
  <div class="header-controls">
    <span id="status-badge" class="status-badge idle">
      <span class="status-dot"></span>
      <span id="status-text">Ready</span>
    </span>
    <select id="llm-select" title="LLM Incident Summary Model">
      <option value="rule">Rule-Based (Offline)</option>
      <option value="ollama">Ollama (Local LLM)</option>
      <option value="openai">OpenAI GPT</option>
      <option value="google">Google Gemini</option>
    </select>
    <button id="start-btn" class="btn btn-primary" onclick="startDemo()">Start Demo</button>
    <button id="stop-btn" class="btn btn-danger" onclick="stopDemo()" style="display:none">Stop</button>
    <a href="/report" target="_blank" class="btn btn-secondary">Report</a>
    <button id="theme-btn" class="btn btn-icon" onclick="toggleTheme()" title="Light/Dark Theme">🌙</button>
  </div>
</header>

<main>
  <!-- LEFT COLUMN -->
  <div class="column-left">

    <!-- Live Video Stream -->
    <div class="card" id="stream-card">
      <div class="card-header">
        <span class="card-title">Live Thermal Stream</span>
        <span id="fps-badge" class="card-subtitle">0.0 FPS</span>
      </div>
      <div class="stream-container">
        <div id="stream-placeholder" class="stream-placeholder">
          <p>Thermal video stream will appear here when the demo starts.</p>
        </div>
        <img id="stream-img" src="" alt="Thermal Stream" style="display:none">
      </div>
    </div>

    <!-- Anomaly Score Graph -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Anomaly Score Stream</span>
        <span class="card-subtitle">Last 300 frames</span>
      </div>
      <div class="card-body">
        <canvas id="score-canvas"></canvas>
        <div class="score-meter">
          <div class="score-meter-header">
            <span>Current Score: <span id="score-label" class="score-value">0.00000</span></span>
            <span>Threshold: 0.20000</span>
          </div>
          <div class="score-bar">
            <div id="score-bar-fill" class="score-bar-fill"></div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <!-- RIGHT COLUMN -->
  <div class="column-right">

    <!-- System Metrics -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">System Metrics</span>
      </div>
      <div class="card-body">
        <div class="metrics-grid">
          <div class="metric-box">
            <div class="metric-label">Processed Frames</div>
            <div class="metric-number" id="m-frames">0</div>
            <div class="metric-unit">frames</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Anomaly Frames</div>
            <div class="metric-number" id="m-anomalies">0</div>
            <div class="metric-unit">detected</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Event Count</div>
            <div class="metric-number" id="m-events">0</div>
            <div class="metric-unit">alerts</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Processing Speed</div>
            <div class="metric-number" id="m-fps">0.0</div>
            <div class="metric-unit">fps</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Event Log -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Incident Log</span>
        <span id="event-count-badge" class="card-subtitle">0 events</span>
      </div>
      <div class="card-body" style="padding: 12px;">
        <div class="alert-feed" id="alert-feed">
          <div class="no-events">No anomaly events recorded yet.</div>
        </div>
      </div>
    </div>

  </div>
</main>

<footer>
  <span>ThermalGuard-VAD &bull; Perimeter Anomaly Detection System</span>
  <span>Spatio-Temporal Autoencoder + ConvLSTM</span>
</footer>

<script>
  let scoreHistory = [];
  let knownEvents = 0;
  let pollInterval = null;
  let streamRetry = 0;
  let currentThreshold = 0.20;

  function toggleTheme() {
    const cur = document.documentElement.getAttribute('data-theme') || 'light';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('tg_theme', next);
    const btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = next === 'dark' ? '☀️' : '🌙';
    drawScoreGraph(currentThreshold);
  }

  (function initTheme() {
    const saved = localStorage.getItem('tg_theme') || 'light';
    document.documentElement.setAttribute('data-theme', saved);
    window.addEventListener('DOMContentLoaded', () => {
      const btn = document.getElementById('theme-btn');
      if (btn) btn.textContent = saved === 'dark' ? '☀️' : '🌙';
      drawScoreGraph(currentThreshold);
    });
  })();

  function startDemo() {
    const llm = document.getElementById('llm-select').value;
    fetch('/api/run_demo?llm=' + llm, { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        if (d.ok) {
          document.getElementById('start-btn').disabled = true;
          document.getElementById('stop-btn').style.display = '';
          startStream();
          startPolling();
        } else {
          alert('Error: ' + (d.error || 'Unknown'));
        }
      });
  }

  function stopDemo() {
    fetch('/api/stop', { method: 'POST' });
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').style.display = 'none';
    stopPolling();
  }

  function startStream() {
    const img = document.getElementById('stream-img');
    const placeholder = document.getElementById('stream-placeholder');
    img.src = '/stream?t=' + Date.now();
    img.style.display = 'block';
    placeholder.style.display = 'none';
    img.onerror = () => {
      if (streamRetry < 10) {
        streamRetry++;
        setTimeout(startStream, 800);
      }
    };
  }

  function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(poll, 500);
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
        stopPolling();
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').style.display = 'none';
      }
    } catch (e) { /* network delay */ }
  }

  function updateStatus(s) {
    document.getElementById('m-frames').textContent = s.frame_count;
    document.getElementById('m-anomalies').textContent = s.anomaly_count;
    document.getElementById('m-events').textContent = s.event_count;
    document.getElementById('m-fps').textContent = s.fps.toFixed(1);
    document.getElementById('fps-badge').textContent = s.fps.toFixed(1) + ' FPS';

    currentThreshold = s.threshold || 0.20;
    const score = s.latest_score;
    const pct = Math.min((score / (currentThreshold * 2)) * 100, 100);
    const fill = document.getElementById('score-bar-fill');
    fill.style.width = pct + '%';
    fill.style.background = score >= currentThreshold ? 'var(--accent-red)' : 'var(--accent-blue)';
    document.getElementById('score-label').textContent = score.toFixed(5);

    const badge = document.getElementById('status-badge');
    const statusMap = {
      idle: ['idle', 'Ready'],
      running: ['running', 'Running'],
      done: ['done', 'Completed'],
      error: ['error', 'Error'],
    };
    const [cls, txt] = statusMap[s.status] || ['idle', 'Ready'];
    badge.className = 'status-badge ' + cls;
    document.getElementById('status-text').textContent = txt;

    fetch('/api/score_history').then(r => r.json()).then(h => {
      scoreHistory = h;
      drawScoreGraph(currentThreshold);
    }).catch(() => {});
  }

  function updateEvents(events) {
    document.getElementById('event-count-badge').textContent = events.length + ' events';
    document.getElementById('m-events').textContent = events.length;
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
          <span class="alert-badge">ALERT #${ev.event_id}</span>
          <span class="alert-time">${startTime}</span>
        </div>
        <div class="alert-meta">
          <span>Peak Score: <strong>${ev.peak_score.toFixed(4)}</strong></span>
          <span>Duration: <strong>${ev.duration_frames} frames</strong></span>
          <span>Type: <strong>${ev.anomaly_type || 'Unknown'}</strong></span>
        </div>
        <div class="alert-summary">${ev.llm_summary || '(Generating LLM summary...)'}</div>
      `;
      feed.prepend(card);
    }
    knownEvents = events.length;
  }

  function drawScoreGraph(threshold) {
    const canvas = document.getElementById('score-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    if (W === 0 || H === 0) return;
    canvas.width = W;
    canvas.height = H;

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const bgColor = isDark ? '#1e293b' : '#ffffff';
    const textColor = isDark ? '#94a3b8' : '#64748b';
    const gridColor = isDark ? '#334155' : '#f1f5f9';

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, W, H);

    if (!scoreHistory || scoreHistory.length < 2) {
      ctx.fillStyle = textColor;
      ctx.font = '12px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for data...', W / 2, H / 2);
      return;
    }

    const pad = { l: 40, r: 16, t: 14, b: 20 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const scores = scoreHistory.map(s => s.score);
    const maxY = Math.max(threshold * 1.8, ...scores, 0.4);

    // Grid lines
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let s = 0; s <= 3; s++) {
      const yVal = (maxY / 3) * s;
      const yPos = pad.t + ph * (1 - yVal / maxY);
      ctx.moveTo(pad.l, yPos);
      ctx.lineTo(pad.l + pw, yPos);
    }
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = textColor;
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let s = 0; s <= 3; s++) {
      const yVal = (maxY / 3) * s;
      const yPos = pad.t + ph * (1 - yVal / maxY);
      ctx.fillText(yVal.toFixed(2), pad.l - 6, yPos + 3);
    }

    // Threshold line
    const ty = pad.t + ph * (1 - threshold / maxY);
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.l, ty);
    ctx.lineTo(pad.l + pw, ty);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#dc2626';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Threshold (' + threshold.toFixed(2) + ')', pad.l + 6, ty - 4);

    // Score line
    ctx.lineWidth = 1.8;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    for (let i = 1; i < scoreHistory.length; i++) {
      const x1 = pad.l + ((i - 1) / (scoreHistory.length - 1)) * pw;
      const y1 = pad.t + ph * (1 - Math.min(scoreHistory[i - 1].score, maxY) / maxY);
      const x2 = pad.l + (i / (scoreHistory.length - 1)) * pw;
      const y2 = pad.t + ph * (1 - Math.min(scoreHistory[i].score, maxY) / maxY);

      const isAnomaly = scoreHistory[i].score >= threshold;
      ctx.strokeStyle = isAnomaly ? '#dc2626' : (isDark ? '#60a5fa' : '#2563eb');
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Current point
    const last = scoreHistory[scoreHistory.length - 1];
    const lx = pad.l + pw;
    const ly = pad.t + ph * (1 - Math.min(last.score, maxY) / maxY);
    ctx.fillStyle = last.score >= threshold ? '#dc2626' : (isDark ? '#60a5fa' : '#2563eb');
    ctx.beginPath();
    ctx.arc(lx, ly, 3, 0, Math.PI * 2);
    ctx.fill();
  }
</script>
</body>
</html>"""


REPORT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ThermalGuard-VAD | Anomaly Report</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background: #ffffff;
      color: #0f172a;
      margin: 0;
      padding: 32px 40px;
      line-height: 1.5;
    }
    .header {
      border-bottom: 2px solid #0f172a;
      padding-bottom: 16px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
    }
    h1 { font-size: 20px; font-weight: 700; margin: 0 0 4px 0; }
    .sub { color: #64748b; font-size: 12px; margin: 0; }
    .btn-print {
      padding: 6px 14px;
      background: #0f172a;
      color: #fff;
      border: none;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 28px;
    }
    .stat {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 12px 16px;
    }
    .stat-l { font-size: 11px; color: #64748b; margin-bottom: 4px; }
    .stat-v { font-size: 22px; font-weight: 700; color: #0f172a; font-family: monospace; }
    .section-title {
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 12px;
      color: #0f172a;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .event {
      border: 1px solid #e2e8f0;
      border-left: 4px solid #dc2626;
      border-radius: 6px;
      padding: 14px 16px;
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
      background: #fee2e2;
      border: 1px solid #fecaca;
      color: #dc2626;
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 4px;
    }
    .ev-time { font-size: 12px; color: #64748b; font-family: monospace; }
    .ev-scores {
      display: flex;
      gap: 16px;
      font-size: 12px;
      color: #64748b;
      margin-bottom: 8px;
    }
    .ev-scores strong { color: #0f172a; }
    .ev-summary {
      font-size: 13px;
      color: #334155;
      background: #f8fafc;
      padding: 10px 12px;
      border-radius: 4px;
      line-height: 1.5;
    }
    .no-events {
      padding: 40px;
      text-align: center;
      color: #64748b;
      background: #f8fafc;
      border: 1px dashed #cbd5e1;
      border-radius: 6px;
    }
    footer {
      border-top: 1px solid #e2e8f0;
      margin-top: 40px;
      padding-top: 16px;
      font-size: 11px;
      color: #94a3b8;
      display: flex;
      justify-content: space-between;
    }
    @media print {
      .btn-print { display: none; }
      body { padding: 0; }
    }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>ThermalGuard-VAD — Anomaly Detection Report</h1>
      <p class="sub">Generated: {{ generated_at }} &bull; Perimeter Surveillance AI Evaluation Document</p>
    </div>
    <button class="btn-print" onclick="window.print()">Print / Save as PDF</button>
  </div>

  <div class="stats">
    <div class="stat"><div class="stat-l">Processed Frames</div><div class="stat-v">{{ total_frames }}</div></div>
    <div class="stat"><div class="stat-l">Anomaly Events</div><div class="stat-v">{{ event_count }}</div></div>
    <div class="stat"><div class="stat-l">Anomaly Frames</div><div class="stat-v" style="color:#dc2626">{{ anomaly_frames }}</div></div>
    <div class="stat"><div class="stat-l">Average Speed</div><div class="stat-v">{{ avg_fps }} FPS</div></div>
  </div>

  <div class="section-title">Detected Incidents</div>

  {% if events %}
    {% for ev in events %}
    <div class="event">
      <div class="ev-h">
        <span class="ev-badge">ALERT #{{ ev.event_id }}</span>
        <span class="ev-time">{{ ev.start_time }}</span>
      </div>
      <div class="ev-scores">
        <div>Frame Range: <strong>{{ ev.start_frame }} &ndash; {{ ev.end_frame }}</strong></div>
        <div>Duration: <strong>{{ ev.duration_frames }} frames</strong></div>
        <div>Peak Score: <strong>{{ "%.4f"|format(ev.peak_score) }}</strong></div>
        <div>Type: <strong>{{ ev.anomaly_type }}</strong></div>
      </div>
      <div class="ev-summary">{{ ev.llm_summary or "(No LLM summary generated)" }}</div>
    </div>
    {% endfor %}
  {% else %}
    <div class="no-events">No recorded anomaly events found.</div>
  {% endif %}

  <footer>
    <span>ThermalGuard-VAD &bull; Infrared Video Anomaly Detection System</span>
    <span>Spatio-Temporal Autoencoder + ConvLSTM</span>
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
    """MJPEG video stream."""
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
            time.sleep(0.05)

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/run_demo", methods=["POST"])
def run_demo():
    if state.is_running:
        return jsonify({"ok": False, "error": "Demo is already running"})
    from flask import request
    llm = request.args.get("llm", "rule")
    state.is_running = True
    t = threading.Thread(target=run_demo_thread, args=(llm,), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop_demo():
    state.is_running = False
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
#  Giriş Noktası
# =========================================================================== #
if __name__ == "__main__":
    print("=" * 65)
    print("  ThermalGuard-VAD Web Dashboard")
    print("  http://localhost:5000")
    print("=" * 65)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
