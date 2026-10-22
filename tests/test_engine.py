import numpy as np
import pytest
import torch

from thermalguard.data.synthetic import AnomalyEvent, ThermalPerimeterSimulator
from thermalguard.engine.anomaly import AnomalyScoreConfig, AnomalyScorer
from thermalguard.engine.eval import calibrate_normal_threshold, compute_frame_metrics
from thermalguard.engine.infer import StreamingInferenceEngine
from thermalguard.models.stae import build_model
from thermalguard.preprocessing.thermal import ThermalPreprocessConfig, ThermalPreprocessor
from thermalguard.visualization.overlay import DashboardVisualizer


def test_anomaly_scorer():
    scorer = AnomalyScorer(AnomalyScoreConfig(percentile=99.0))
    orig = np.zeros((128, 160), dtype=np.float32)
    recon = np.zeros((128, 160), dtype=np.float32)

    # Sıcak anomali hedefi yerleştir
    orig[50:65, 70:85] = 0.85

    error_map = scorer.compute_error_map(orig, recon)
    assert error_map.shape == (128, 160)
    assert error_map.max() > 0.5

    score = scorer.compute_frame_score(error_map)
    assert score > 0.4

    heatmap = scorer.generate_heatmap(error_map)
    assert heatmap.shape == (128, 160, 3)


def test_streaming_inference_pipeline():
    model = build_model({"widths": (16, 32), "latent": 16, "lstm_hidden": (16,)})
    prep = ThermalPreprocessor(ThermalPreprocessConfig(output_size=(64, 48)))
    scorer = AnomalyScorer()

    engine = StreamingInferenceEngine(model, prep, scorer, clip_len=4, threshold=0.3, device="cpu")

    raw_frame = np.random.randint(40, 160, (240, 320), dtype=np.uint8)

    # 5 ardışık kare gönder (buffer dolumu ve çıkarım testi)
    for _ in range(5):
        res = engine.step(raw_frame)
        assert res.preprocessed.shape == (48, 64)
        assert res.reconstructed.shape == (48, 64)
        assert res.error_map.shape == (48, 64)
        assert isinstance(res.anomaly_score, float)
        assert res.fps > 0.0


def test_perimeter_simulator_and_hud_rendering():
    sim = ThermalPerimeterSimulator(width=80, height=60, num_frames=20, seed=1)
    anom = [AnomalyEvent(start_frame=5, end_frame=15, size=(6, 12))]
    frames, labels, masks = sim.generate_sequence(anomalies=anom)

    assert len(frames) == 20
    assert labels[10] == 1
    assert labels[0] == 0
    assert masks[10].sum() > 0
    assert masks[0].sum() == 0

    # Dashboard görselleştirici testi
    viz = DashboardVisualizer(width=320, height=260, graph_height=80)
    model = build_model({"widths": (16, 32), "latent": 16, "lstm_hidden": (16,)})
    prep = ThermalPreprocessor(ThermalPreprocessConfig(output_size=(80, 60)))
    engine = StreamingInferenceEngine(model, prep, clip_len=3, device="cpu")

    res = engine.step(frames[0])
    hud = viz.render_frame(res)
    assert hud.shape == (260, 320, 3)
