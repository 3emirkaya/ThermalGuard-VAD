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
        self.threshold: float = 0.11
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

        scorer = AnomalyScorer(AnomalyScoreConfig(percentile=99.2, temporal_smoothing=0.6))
        engine = StreamingInferenceEngine(
            model=model, preprocessor=preprocessor, scorer=scorer,
            clip_len=8, threshold=state.threshold, device=device_str,
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
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="ThermalGuard-VAD — Kızılötesi Anomali Tespit Sistemi Canlı Dashboard">
  <title>ThermalGuard-VAD | Termal Anomali İzleme</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0a0c12;
      --bg-secondary: #111420;
      --bg-card: #161b2e;
      --bg-card-hover: #1c2240;
      --border: #1e2d4a;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-red: #ef4444;
      --accent-orange: #f59e0b;
      --accent-purple: #8b5cf6;
      --text-primary: #e2e8f0;
      --text-secondary: #94a3b8;
      --text-muted: #475569;
      --glow-blue: 0 0 20px rgba(59, 130, 246, 0.3);
      --glow-red: 0 0 20px rgba(239, 68, 68, 0.4);
      --glow-green: 0 0 20px rgba(16, 185, 129, 0.3);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* ---- Animated background grid ---- */
    body::before {
      content: '';
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background-image:
        linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px);
      background-size: 40px 40px;
      pointer-events: none;
      z-index: 0;
    }

    /* ---- Header ---- */
    header {
      position: relative;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 28px;
      background: rgba(17, 20, 32, 0.95);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(10px);
    }

    .header-logo {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo-icon {
      width: 38px; height: 38px;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px;
      box-shadow: var(--glow-blue);
    }

    .logo-text h1 {
      font-size: 18px;
      font-weight: 700;
      background: linear-gradient(90deg, #60a5fa, #22d3ee);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .logo-text p {
      font-size: 11px;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: 0.05em;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    /* ---- Status Badge ---- */
    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 99px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.05em;
      border: 1px solid;
      transition: all 0.3s ease;
    }

    .status-badge.idle { border-color: var(--text-muted); color: var(--text-muted); }
    .status-badge.running { border-color: var(--accent-green); color: var(--accent-green); box-shadow: var(--glow-green); }
    .status-badge.done { border-color: var(--accent-blue); color: var(--accent-blue); }
    .status-badge.error { border-color: var(--accent-red); color: var(--accent-red); }

    .status-dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: currentColor;
    }
    .status-badge.running .status-dot {
      animation: pulse 1.2s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.7); }
    }

    /* ---- Buttons ---- */
    .btn {
      padding: 8px 18px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      border: none;
      outline: none;
      letter-spacing: 0.02em;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
      color: white;
    }
    .btn-primary:hover { opacity: 0.85; transform: translateY(-1px); box-shadow: var(--glow-blue); }
    .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    .btn-danger {
      background: rgba(239, 68, 68, 0.15);
      color: var(--accent-red);
      border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .btn-danger:hover { background: rgba(239, 68, 68, 0.25); }

    /* ---- Main Layout ---- */
    main {
      position: relative;
      z-index: 1;
      max-width: 1600px;
      margin: 0 auto;
      padding: 20px 24px;
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 20px;
    }

    .left-column { display: flex; flex-direction: column; gap: 18px; }
    .right-column { display: flex; flex-direction: column; gap: 18px; }

    /* ---- Cards ---- */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      transition: border-color 0.3s ease;
    }
    .card:hover { border-color: rgba(59, 130, 246, 0.3); }

    .card-header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(255,255,255,0.02);
    }

    .card-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-secondary);
      letter-spacing: 0.06em;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .card-title .icon {
      font-size: 16px;
    }

    .card-body { padding: 16px 18px; }

    /* ---- Stream ---- */
    .stream-container {
      position: relative;
      width: 100%;
      background: #000;
      border-radius: 0 0 13px 13px;
      min-height: 380px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    #stream-img {
      max-width: 100%;
      max-height: 540px;
      object-fit: contain;
      display: block;
    }

    .stream-placeholder {
      text-align: center;
      color: var(--text-muted);
      padding: 60px 20px;
    }
    .stream-placeholder .icon { font-size: 48px; margin-bottom: 12px; }
    .stream-placeholder p { font-size: 14px; }

    /* ---- Metrics ---- */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }

    .metric-card {
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      transition: all 0.2s ease;
    }

    .metric-card:hover {
      background: rgba(255,255,255,0.05);
      transform: translateY(-1px);
    }

    .metric-label {
      font-size: 11px;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }

    .metric-value {
      font-size: 24px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      background: linear-gradient(90deg, #60a5fa, #22d3ee);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .metric-unit {
      font-size: 12px;
      color: var(--text-secondary);
      margin-top: 3px;
    }

    /* ---- Score Graph (canvas) ---- */
    #score-canvas {
      width: 100%;
      height: 140px;
      background: rgba(0,0,0,0.3);
      border-radius: 8px;
      display: block;
    }

    /* ---- Alert Feed ---- */
    .alert-feed {
      max-height: 380px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .alert-feed::-webkit-scrollbar { width: 4px; }
    .alert-feed::-webkit-scrollbar-track { background: transparent; }
    .alert-feed::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    .alert-item {
      background: rgba(239, 68, 68, 0.08);
      border: 1px solid rgba(239, 68, 68, 0.2);
      border-radius: 10px;
      padding: 12px 14px;
      animation: slideIn 0.4s ease;
    }

    @keyframes slideIn {
      from { opacity: 0; transform: translateX(20px); }
      to { opacity: 1; transform: translateX(0); }
    }

    .alert-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 6px;
    }

    .alert-badge {
      background: var(--accent-red);
      color: white;
      font-size: 10px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 99px;
      letter-spacing: 0.05em;
    }

    .alert-time {
      font-size: 11px;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }

    .alert-score {
      font-size: 13px;
      font-weight: 600;
      color: var(--accent-orange);
      margin-bottom: 6px;
      font-family: 'JetBrains Mono', monospace;
    }

    .alert-summary {
      font-size: 12px;
      color: var(--text-secondary);
      line-height: 1.5;
    }

    .no-events {
      text-align: center;
      padding: 40px 20px;
      color: var(--text-muted);
      font-size: 13px;
    }
    .no-events .icon { font-size: 32px; margin-bottom: 8px; }

    /* ---- Score bar ---- */
    .score-bar-wrap {
      margin-top: 10px;
    }

    .score-bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-secondary);
      margin-bottom: 6px;
    }

    .score-bar {
      height: 8px;
      background: rgba(255,255,255,0.07);
      border-radius: 4px;
      overflow: hidden;
    }

    .score-bar-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.3s ease, background 0.3s ease;
    }

    /* ---- Alarm flash ---- */
    .alarm-flash {
      animation: alarmFlash 0.6s ease 3;
    }

    @keyframes alarmFlash {
      0%, 100% { box-shadow: none; }
      50% { box-shadow: 0 0 30px rgba(239,68,68,0.6), inset 0 0 20px rgba(239,68,68,0.1); }
    }

    /* ---- LLM select ---- */
    select {
      background: var(--bg-secondary);
      color: var(--text-primary);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 5px 10px;
      font-size: 12px;
      font-family: 'Inter', sans-serif;
      cursor: pointer;
      outline: none;
    }
    select:hover { border-color: var(--accent-blue); }

    /* ---- Footer ---- */
    footer {
      text-align: center;
      padding: 16px;
      color: var(--text-muted);
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
      border-top: 1px solid var(--border);
      position: relative;
      z-index: 1;
    }
  </style>
</head>
<body>

<header>
  <div class="header-logo">
    <div class="logo-icon">🌡️</div>
    <div class="logo-text">
      <h1>ThermalGuard-VAD</h1>
      <p>INFRARED ANOMALY DETECTION SYSTEM v2.0</p>
    </div>
  </div>
  <div class="header-actions">
    <span id="status-badge" class="status-badge idle">
      <span class="status-dot"></span>
      <span id="status-text">HAZIR</span>
    </span>
    <select id="llm-select" title="LLM Özet Arka Ucu">
      <option value="rule">🔵 Kural Tabanlı (Offline)</option>
      <option value="ollama">🟣 Ollama (Lokal LLM)</option>
      <option value="openai">🟢 OpenAI GPT</option>
      <option value="google">🟡 Google Gemini</option>
    </select>
    <button id="start-btn" class="btn btn-primary" onclick="startDemo()">▶ Demo Başlat</button>
    <button id="stop-btn" class="btn btn-danger" onclick="stopDemo()" style="display:none">⏹ Durdur</button>
    <a href="/report" target="_blank" class="btn btn-primary" style="text-decoration:none; background: linear-gradient(135deg,#8b5cf6,#6366f1)">📄 Rapor</a>
  </div>
</header>

<main>
  <!-- SOL KOLON -->
  <div class="left-column">

    <!-- Canlı Video Akışı -->
    <div class="card" id="stream-card">
      <div class="card-header">
        <span class="card-title"><span class="icon">📡</span> CANLI TERMAL AKIŞ</span>
        <span id="fps-badge" style="font-size:12px;color:var(--accent-cyan);font-family:'JetBrains Mono',monospace">0.0 FPS</span>
      </div>
      <div class="stream-container">
        <div id="stream-placeholder" class="stream-placeholder">
          <div class="icon">🌡️</div>
          <p>Demo başlatıldığında canlı termal akış burada görünecek.</p>
        </div>
        <img id="stream-img" src="" alt="Termal Akış" style="display:none">
      </div>
    </div>

    <!-- Anlık Skor Grafiği (Canvas) -->
    <div class="card">
      <div class="card-header">
        <span class="card-title"><span class="icon">📈</span> ANOMALİ SKORU AKIŞI</span>
        <span style="font-size:11px;color:var(--text-muted)">Son 300 kare</span>
      </div>
      <div class="card-body">
        <canvas id="score-canvas"></canvas>
        <div class="score-bar-wrap" style="margin-top:12px">
          <div class="score-bar-label">
            <span>Anlık Skor</span>
            <span id="score-label" style="font-family:'JetBrains Mono',monospace;font-weight:600">0.00000</span>
          </div>
          <div class="score-bar">
            <div id="score-bar-fill" class="score-bar-fill" style="width:0%;background:var(--accent-green)"></div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <!-- SAĞ KOLON -->
  <div class="right-column">

    <!-- Metrikler -->
    <div class="card">
      <div class="card-header">
        <span class="card-title"><span class="icon">⚡</span> SİSTEM METRİKLERİ</span>
      </div>
      <div class="card-body">
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">İşlenen Kare</div>
            <div class="metric-value" id="m-frames">0</div>
            <div class="metric-unit">kare</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Anomali Karesi</div>
            <div class="metric-value" id="m-anomalies" style="background:linear-gradient(90deg,#ef4444,#f97316);-webkit-background-clip:text">0</div>
            <div class="metric-unit">tespit</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Olay Sayısı</div>
            <div class="metric-value" id="m-events" style="background:linear-gradient(90deg,#8b5cf6,#ec4899);-webkit-background-clip:text">0</div>
            <div class="metric-unit">alarm</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">FPS</div>
            <div class="metric-value" id="m-fps">0.0</div>
            <div class="metric-unit">kare/sn</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Alarm Akışı -->
    <div class="card">
      <div class="card-header">
        <span class="card-title"><span class="icon">🚨</span> ANLİK ALARM AKIŞI</span>
        <span id="event-count-badge" style="background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);color:var(--accent-red);font-size:11px;padding:2px 8px;border-radius:99px;font-weight:600">0 olay</span>
      </div>
      <div class="card-body" style="padding:10px 14px">
        <div class="alert-feed" id="alert-feed">
          <div class="no-events">
            <div class="icon">🛡️</div>
            <p>Henüz anomali tespit edilmedi.</p>
          </div>
        </div>
      </div>
    </div>

  </div>
</main>

<footer>
  ThermalGuard-VAD &nbsp;|&nbsp; Spatio-Temporal Autoencoder (STAE) + ConvLSTM &nbsp;|&nbsp; KAIST LWIR Dataset Compatible
</footer>

<script>
  // ------------------------------------------------------------------ //
  //  Global durum
  // ------------------------------------------------------------------ //
  let scoreHistory = [];
  let knownEvents = 0;
  let pollInterval = null;
  let streamRetry = 0;

  // ------------------------------------------------------------------ //
  //  Demo Kontrol
  // ------------------------------------------------------------------ //
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
          alert('Hata: ' + (d.error || 'Bilinmeyen'));
        }
      });
  }

  function stopDemo() {
    fetch('/api/stop', { method: 'POST' });
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').style.display = 'none';
    stopPolling();
  }

  // ------------------------------------------------------------------ //
  //  MJPEG Akış
  // ------------------------------------------------------------------ //
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

  // ------------------------------------------------------------------ //
  //  Durum Polling
  // ------------------------------------------------------------------ //
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
    } catch (e) { /* network hatası */ }
  }

  function updateStatus(s) {
    // Metrikler
    document.getElementById('m-frames').textContent = s.frame_count;
    document.getElementById('m-anomalies').textContent = s.anomaly_count;
    document.getElementById('m-events').textContent = s.event_count;
    document.getElementById('m-fps').textContent = s.fps.toFixed(1);
    document.getElementById('fps-badge').textContent = s.fps.toFixed(1) + ' FPS';

    // Skor bar
    const score = s.latest_score;
    const pct = Math.min(score / (s.threshold * 2) * 100, 100);
    const fill = document.getElementById('score-bar-fill');
    fill.style.width = pct + '%';
    fill.style.background = score >= s.threshold
      ? 'linear-gradient(90deg,#ef4444,#f97316)'
      : 'linear-gradient(90deg,#10b981,#06b6d4)';
    document.getElementById('score-label').textContent = score.toFixed(5);

    // Status badge
    const badge = document.getElementById('status-badge');
    const statusMap = {
      idle: ['idle','HAZIR'],
      running: ['running','ÇALIŞIYOR'],
      done: ['done','TAMAMLANDI'],
      error: ['error','HATA'],
    };
    const [cls, txt] = statusMap[s.status] || ['idle','HAZIR'];
    badge.className = 'status-badge ' + cls;
    document.getElementById('status-text').textContent = txt;

    // Score history — grafik için
    fetch('/api/score_history').then(r => r.json()).then(h => {
      scoreHistory = h;
      drawScoreGraph(s.threshold);
    }).catch(() => {});
  }

  function updateEvents(events) {
    document.getElementById('event-count-badge').textContent = events.length + ' olay';
    document.getElementById('m-events').textContent = events.length;
    if (events.length === knownEvents) return;

    // Yeni olaylar var → ekle
    const feed = document.getElementById('alert-feed');
    if (events.length === 1 && knownEvents === 0) {
      feed.innerHTML = ''; // "henüz anomali yok" mesajını kaldır
    }

    for (let i = knownEvents; i < events.length; i++) {
      const ev = events[i];
      const card = document.createElement('div');
      card.className = 'alert-item';
      const startTime = ev.start_time ? ev.start_time.replace('T', ' ').substring(0, 19) : '-';
      card.innerHTML = `
        <div class="alert-header">
          <span class="alert-badge">ALARM #${ev.event_id}</span>
          <span class="alert-time">${startTime}</span>
        </div>
        <div class="alert-score">
          Zirve: ${ev.peak_score.toFixed(4)} &nbsp;|&nbsp; ${ev.duration_frames} kare
        </div>
        <div class="alert-summary">${ev.llm_summary || '(LLM özeti bekleniyor...)'}</div>
      `;
      feed.prepend(card);

      // Stream kartına flash efekti
      document.getElementById('stream-card').classList.add('alarm-flash');
      setTimeout(() => document.getElementById('stream-card').classList.remove('alarm-flash'), 2000);
    }
    knownEvents = events.length;
  }

  // ------------------------------------------------------------------ //
  //  Canvas Skor Grafiği
  // ------------------------------------------------------------------ //
  function drawScoreGraph(threshold) {
    const canvas = document.getElementById('score-canvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    canvas.width = W;
    canvas.height = H;

    ctx.clearRect(0, 0, W, H);

    if (!scoreHistory || scoreHistory.length < 2) {
      ctx.fillStyle = '#475569';
      ctx.font = '12px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('Veri bekleniyor...', W/2, H/2);
      return;
    }

    const pad = { l: 10, r: 10, t: 10, b: 10 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const scores = scoreHistory.map(s => s.score);
    const maxY = Math.max(threshold * 2, ...scores, 0.5);

    // Threshold line
    const ty = pad.t + ph * (1 - threshold / maxY);
    ctx.strokeStyle = 'rgba(239,68,68,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.l, ty);
    ctx.lineTo(pad.l + pw, ty);
    ctx.stroke();
    ctx.setLineDash([]);

    // Score line
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';

    for (let i = 1; i < scoreHistory.length; i++) {
      const x1 = pad.l + ((i - 1) / (scoreHistory.length - 1)) * pw;
      const y1 = pad.t + ph * (1 - Math.min(scoreHistory[i-1].score, maxY) / maxY);
      const x2 = pad.l + (i / (scoreHistory.length - 1)) * pw;
      const y2 = pad.t + ph * (1 - Math.min(scoreHistory[i].score, maxY) / maxY);

      const isAnomaly = scoreHistory[i].score >= threshold;
      ctx.strokeStyle = isAnomaly ? '#ef4444' : '#10b981';
      ctx.lineWidth = isAnomaly ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Last point dot
    const last = scoreHistory[scoreHistory.length - 1];
    const lx = pad.l + pw;
    const ly = pad.t + ph * (1 - Math.min(last.score, maxY) / maxY);
    ctx.fillStyle = last.score >= threshold ? '#ef4444' : '#10b981';
    ctx.beginPath();
    ctx.arc(lx, ly, 4, 0, Math.PI * 2);
    ctx.fill();
  }
</script>
</body>
</html>"""


REPORT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <title>ThermalGuard-VAD | Anomali Raporu</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter',sans-serif; background:#0f1117; color:#e2e8f0; margin:0; padding:32px; }
    h1 { font-size:24px; font-weight:700; color:#60a5fa; margin-bottom:4px; }
    .sub { color:#64748b; font-size:13px; margin-bottom:32px; }
    .stats { display:flex; gap:16px; margin-bottom:32px; }
    .stat { background:#161b2e; border:1px solid #1e2d4a; border-radius:10px; padding:16px 20px; min-width:120px; }
    .stat-v { font-size:28px; font-weight:700; color:#60a5fa; }
    .stat-l { font-size:11px; color:#64748b; margin-top:4px; }
    .event { background:#161b2e; border:1px solid rgba(239,68,68,0.2); border-radius:12px; padding:18px; margin-bottom:14px; }
    .ev-h { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
    .ev-badge { background:#ef4444; color:white; font-size:11px; font-weight:700; padding:3px 10px; border-radius:99px; }
    .ev-time { font-size:12px; color:#64748b; font-family:monospace; }
    .ev-scores { display:flex; gap:20px; margin-bottom:10px; }
    .ev-s { font-size:13px; font-family:monospace; }
    .ev-s span { color:#f59e0b; font-weight:600; }
    .ev-summary { font-size:14px; color:#94a3b8; line-height:1.6; border-left:3px solid #3b82f6; padding-left:12px; }
    .no-events { text-align:center; padding:60px; color:#475569; }
    footer { text-align:center; margin-top:48px; color:#475569; font-size:12px; }
  </style>
</head>
<body>
  <h1>🌡️ ThermalGuard-VAD — Anomali Tespit Raporu</h1>
  <p class="sub">Oluşturulma: {{ generated_at }}</p>

  <div class="stats">
    <div class="stat"><div class="stat-v">{{ total_frames }}</div><div class="stat-l">İşlenen Kare</div></div>
    <div class="stat"><div class="stat-v">{{ event_count }}</div><div class="stat-l">Anomali Olayı</div></div>
    <div class="stat"><div class="stat-v" style="color:#ef4444">{{ anomaly_frames }}</div><div class="stat-l">Anomali Karesi</div></div>
    <div class="stat"><div class="stat-v">{{ avg_fps }}</div><div class="stat-l">Ort. FPS</div></div>
  </div>

  {% if events %}
    {% for ev in events %}
    <div class="event">
      <div class="ev-h">
        <span class="ev-badge">ALARM #{{ ev.event_id }}</span>
        <span class="ev-time">{{ ev.start_time }}</span>
      </div>
      <div class="ev-scores">
        <div class="ev-s">Kare: <span>{{ ev.start_frame }} – {{ ev.end_frame }}</span></div>
        <div class="ev-s">Süre: <span>{{ ev.duration_frames }} kare</span></div>
        <div class="ev-s">Zirve Skor: <span>{{ "%.4f"|format(ev.peak_score) }}</span></div>
        <div class="ev-s">Tür: <span>{{ ev.anomaly_type }}</span></div>
      </div>
      <div class="ev-summary">{{ ev.llm_summary or "(LLM özeti henüz oluşturulmadı)" }}</div>
    </div>
    {% endfor %}
  {% else %}
    <div class="no-events">📡 Henüz kayıtlı anomali olayı bulunmuyor.</div>
  {% endif %}

  <footer>ThermalGuard-VAD &nbsp;|&nbsp; STAE + ConvLSTM Anomali Tespit Motoru</footer>
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
    """MJPEG video akışı."""
    def generate():
        blank = np.full((380, 640, 3), 22, dtype=np.uint8)
        cv2.putText(blank, "Akis Bekleniyor...", (200, 190),
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
        return jsonify({"ok": False, "error": "Demo zaten çalışıyor"})
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
