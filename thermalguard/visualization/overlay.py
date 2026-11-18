"""
Tactical Defense Dashboard and Video Visualizer.

Capabilities:
-------------
* Top Panel: Real-time scrolling "Anomaly Score" time-series graph with threshold line,
  tactical status colors, and telemetry.
* Bottom Panel: Dual-stream visualization with raw LWIR White-Hot thermal stream on the left,
  and colorized Ironbow fusion with reconstruction error highlighting and tactical bounding box on the right.
* Export: Directly exports optimized web demo.gif and tactical demo.mp4 files.
"""
from __future__ import annotations

import collections
import os
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from ..engine.infer import InferenceResult


class DashboardVisualizer:
    """
    Renders military-grade tactical HUD video frames featuring top scrolling
    anomaly graph and bottom dual-feed infrared thermal fusion.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 560,
        graph_height: int = 180,
        history_len: int = 100,
        threshold: float = 0.20,
        title: str = "ThermalGuard-VAD | Perimeter Anomaly Detection",
    ):
        self.width = width
        self.height = height
        self.graph_height = graph_height
        self.video_height = height - graph_height
        self.history_len = history_len
        self.threshold = threshold
        self.title = title

        self.scores: Deque[float] = collections.deque(maxlen=history_len)
        self.timestamps: Deque[int] = collections.deque(maxlen=history_len)

    def reset(self) -> None:
        self.scores.clear()
        self.timestamps.clear()

    def _render_score_graph(
        self,
        current_score: float,
        current_frame: int,
        w: int,
        h: int,
    ) -> np.ndarray:
        """
        Top panel: OpenCV-optimized scrolling time-series anomaly score graph.
        """
        graph = np.full((h, w, 3), 22, dtype=np.uint8)  # Dark tactical slate background

        # Margins and plot dimensions
        pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 25
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b

        # Plot background canvas
        cv2.rectangle(graph, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), (32, 35, 40), -1)
        cv2.rectangle(graph, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), (65, 75, 85), 1)

        # Y-axis scaling (0.0 to at least 0.8 or max_score * 1.25)
        max_y = max(0.8, self.threshold * 1.8, max(self.scores) * 1.25 if self.scores else 0.8)

        # Y-axis grid lines and labels
        for val in [0.2, 0.4, 0.6, 0.8]:
            if val < max_y:
                y_px = int(pad_t + plot_h * (1.0 - val / max_y))
                cv2.line(graph, (pad_l, y_px), (pad_l + plot_w, y_px), (45, 50, 58), 1, cv2.LINE_AA)
                cv2.putText(graph, f"{val:.1f}", (pad_l - 42, y_px + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 150, 160), 1, cv2.LINE_AA)

        # Threshold reference line
        thresh_y = int(pad_t + plot_h * (1.0 - self.threshold / max_y))
        if pad_t <= thresh_y <= pad_t + plot_h:
            dash_w = 8
            for x in range(pad_l, pad_l + plot_w, dash_w * 2):
                cv2.line(graph, (x, thresh_y), (min(x + dash_w, pad_l + plot_w), thresh_y),
                         (60, 70, 240), 2, cv2.LINE_AA)
            cv2.putText(graph, f"THRESHOLD: {self.threshold:.2f}",
                        (pad_l + plot_w - 140, thresh_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 90, 245), 1, cv2.LINE_AA)

        # Plot scrolling score trajectory
        if len(self.scores) >= 2:
            n_pts = len(self.scores)
            pts = []
            for i, sc in enumerate(self.scores):
                x_px = int(pad_l + (i / max(1, self.history_len - 1)) * plot_w)
                y_px = int(pad_t + plot_h * (1.0 - np.clip(sc, 0.0, max_y) / max_y))
                pts.append((x_px, y_px))

            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                val = self.scores[i + 1]
                # Electric alert red or bright neon green
                color = (40, 50, 245) if val >= self.threshold else (40, 240, 50)
                thickness = 2 if val < self.threshold else 3
                cv2.line(graph, p1, p2, color, thickness, cv2.LINE_AA)

            last_pt = pts[-1]
            cv2.circle(graph, last_pt, 4,
                       (40, 50, 255) if current_score >= self.threshold else (40, 255, 60), -1)

        # Graph Header and Status Value (No overlap coordinates)
        is_alert = current_score >= self.threshold
        status_text = "ANOMALY DETECTED [ALARM]" if is_alert else "PERIMETER SECURE"
        status_color = (40, 60, 255) if is_alert else (40, 245, 60)

        cv2.putText(graph, "ANOMALY SCORE STREAM", (pad_l, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 210, 220), 1, cv2.LINE_AA)

        cv2.putText(graph, f"SCORE: {current_score:.4f}", (pad_l + 230, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)

        cv2.putText(graph, f"[{status_text}]", (pad_l + plot_w - 200, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)

        return graph

    def render_frame(self, result: InferenceResult) -> np.ndarray:
        """
        Render combined HUD frame: score chart on top, dual thermal view on bottom, telemetry bar.
        """
        self.scores.append(result.anomaly_score)
        self.timestamps.append(result.frame_idx)

        # 1. Top Panel: Graph
        graph_panel = self._render_score_graph(
            current_score=result.anomaly_score,
            current_frame=result.frame_idx,
            w=self.width,
            h=self.graph_height,
        )

        # 2. Bottom Panel: Video & Heatmap
        sub_w = self.width // 2
        sub_h = self.video_height - 35

        # Left: Preprocessed Thermal Input (White-Hot Grayscale)
        left_u8 = (result.preprocessed * 255.0).astype(np.uint8)
        left_bgr = cv2.cvtColor(left_u8, cv2.COLOR_GRAY2BGR)
        left_resized = cv2.resize(left_bgr, (sub_w, sub_h), interpolation=cv2.INTER_LINEAR)
        # Black stroke + crisp white text for high contrast on any background
        cv2.putText(left_resized, "LWIR THERMAL INPUT (WHITE-HOT RAW)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(left_resized, "LWIR THERMAL INPUT (WHITE-HOT RAW)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)

        # Right: Authentic Thermal False-Color (Ironbow / Inferno Palette)
        thermal_color = cv2.applyColorMap(left_u8, cv2.COLORMAP_INFERNO)
        right_resized = cv2.resize(thermal_color, (sub_w, sub_h), interpolation=cv2.INTER_LINEAR)

        if result.is_anomaly:
            cv2.rectangle(right_resized, (0, 0), (sub_w - 1, sub_h - 1), (30, 40, 245), 3)
            # High-visibility tactical yellow with black stroke shadow
            cv2.putText(right_resized, "! ANOMALOUS TARGET DETECTED !", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(right_resized, "! ANOMALOUS TARGET DETECTED !", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (40, 220, 255), 1, cv2.LINE_AA)

            # Detect target blob and draw tactical bounding box on primary intruder
            if getattr(result, "error_map", None) is not None:
                err = result.error_map
                p_thresh = max(float(np.percentile(err, 98.2)), 0.08)
                blob_mask = (err >= p_thresh).astype(np.uint8)
                # Morphological close bridges vehicle wheels, engine, and body into a single coherent entity
                blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, np.ones((7, 15), np.uint8))
                contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                scale_x = sub_w / max(1, err.shape[1])
                scale_y = sub_h / max(1, err.shape[0])

                valid_cnts = [c for c in contours if cv2.contourArea(c) >= 12]
                if valid_cnts:
                    # Pick largest contour or merge contiguous target blobs
                    main_cnt = max(valid_cnts, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(main_cnt)

                    # Merge any nearby contours (e.g. trailing exhaust plume or secondary wheel)
                    for other_c in valid_cnts:
                        if other_c is main_cnt:
                            continue
                        ox, oy, ow, oh = cv2.boundingRect(other_c)
                        # Check horizontal and vertical proximity
                        if abs(ox - (bx + bw // 2)) < 40 and abs(oy - by) < 18:
                            nx1 = min(bx, ox)
                            ny1 = min(by, oy)
                            nx2 = max(bx + bw, ox + ow)
                            ny2 = max(by + bh, oy + oh)
                            bx, by, bw, bh = nx1, ny1, nx2 - nx1, ny2 - ny1

                    # Ignore artifacts in top banner area
                    if by * scale_y > 28:
                        rx1 = max(2, int((bx - 4) * scale_x))
                        ry1 = max(2, int((by - 4) * scale_y))
                        rx2 = min(sub_w - 3, int((bx + bw + 4) * scale_x))
                        ry2 = min(sub_h - 3, int((by + bh + 4) * scale_y))

                        cv2.rectangle(right_resized, (rx1, ry1), (rx2, ry2), (30, 40, 245), 1)
                        cl = 6
                        cv2.line(right_resized, (rx1, ry1), (rx1 + cl, ry1), (40, 60, 255), 2)
                        cv2.line(right_resized, (rx1, ry1), (rx1, ry1 + cl), (40, 60, 255), 2)
                        cv2.line(right_resized, (rx2, ry1), (rx2 - cl, ry1), (40, 60, 255), 2)
                        cv2.line(right_resized, (rx2, ry1), (rx2, ry1 + cl), (40, 60, 255), 2)
                        cv2.line(right_resized, (rx1, ry2), (rx1 + cl, ry2), (40, 60, 255), 2)
                        cv2.line(right_resized, (rx1, ry2), (rx1, ry2 - cl), (40, 60, 255), 2)
                        cv2.line(right_resized, (rx2, ry2), (rx2 - cl, ry2), (40, 60, 255), 2)
                        cv2.line(right_resized, (rx2, ry2), (rx2 - cl, ry2), (40, 60, 255), 2)

                        target_label = getattr(result, "target_label", None)
                        if not target_label:
                            aspect = (rx2 - rx1) / max(1, (ry2 - ry1))
                            box_w = rx2 - rx1
                            box_h = ry2 - ry1
                            if box_w >= 36 and box_h >= 16 and aspect > 1.35:
                                target_label = "VEHICLE [ALERT]"
                            elif aspect > 1.8 and box_h <= 18:
                                target_label = "SABOTEUR [ALERT]"
                            else:
                                target_label = "INTRUDER [ALERT]"

                        cv2.putText(right_resized, target_label, (rx1, max(35, ry1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (50, 220, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(right_resized, "LWIR THERMAL FUSION (IRONBOW PALETTE)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(right_resized, "LWIR THERMAL FUSION (IRONBOW PALETTE)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 255, 120), 1, cv2.LINE_AA)

        # Combine side by side
        video_strip = np.hstack([left_resized, right_resized])

        # Bottom Telemetry Bar (No overlap coordinates)
        hud_bar = np.full((35, self.width, 3), 18, dtype=np.uint8)
        is_cuda = "cuda" in str(getattr(result, "error_map", "")).lower()
        dev_name = "GPU (CUDA)" if is_cuda else "CPU"
        telemetry = (
            f"FRAME: {result.frame_idx:04d} | "
            f"LATENCY: {result.latency_ms:.1f}ms | "
            f"FPS: {result.fps:.1f} | "
            f"DEV: {dev_name}"
        )
        cv2.putText(hud_bar, telemetry, (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 190, 200), 1, cv2.LINE_AA)

        state_color = (40, 50, 255) if result.is_anomaly else (40, 245, 60)
        state_label = "ALARM: ANOMALY DETECTED" if result.is_anomaly else "STATUS: PERIMETER SECURE"
        cv2.putText(hud_bar, state_label, (self.width - 220, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, state_color, 1, cv2.LINE_AA)

        # Vertically stack all panels
        full_frame = np.vstack([graph_panel, video_strip, hud_bar])
        return full_frame


def save_demo_gif(
    frames: List[np.ndarray],
    output_path: str = "demo.gif",
    fps: float = 15.0,
    colors: int = 96,
) -> None:
    """
    Saves frame sequence as a highly-optimized web GIF.
    Applies adaptive color quantization and delta compression to minimize file size.
    """
    if not frames:
        return
    try:
        from PIL import Image
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        # Adaptive palette quantization for fast web loading and high contrast
        quantized = [f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT) for f in pil_frames]
        duration = int(1000.0 / fps)
        quantized[0].save(
            output_path,
            save_all=True,
            append_images=quantized[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0.0
        print(f"[✓] Demo GIF successfully created: {output_path} ({file_size_mb:.2f} MB, {fps:.1f} FPS)")
    except Exception as e:
        print(f"[!] PIL optimization warning ({e}), falling back to imageio...")
        try:
            import imageio
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            imageio.mimsave(output_path, rgb_frames, fps=fps, loop=0)
            print(f"[✓] Demo GIF successfully created: {output_path}")
        except ImportError:
            print("[!] imageio not found, falling back to MP4 export...")
            save_demo_video(frames, output_path.replace(".gif", ".mp4"), fps=fps)


def save_demo_video(
    frames: List[np.ndarray],
    output_path: str = "demo.mp4",
    fps: float = 20.0,
) -> None:
    """
    Saves frame sequence as an MP4 video file.
    """
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"[✓] Demo video successfully saved: {output_path}")
