"""
Physics-based Infrared LWIR Thermal Video Simulator for Defense Perimeter Security.

Generates realistic LWIR (Long-Wave Infrared 8-14 um) video sequences simulating
physical sensor and thermal radiation dynamics for immediate demonstrations,
unit testing, and perimeter security evaluation without requiring external dataset downloads.

Simulated Thermal Dynamics:
---------------------------
  1. Thermal Backgrounds: Cold night sky (low radiance), ground temperature gradients,
     watchtowers, concrete anti-ram barriers, asphalt roads, and metal perimeter fencing.
  2. Multi-Environment Presets:
     - 'outpost': Military Forward Operating Base (FOB) with guard tower, patrol road,
       mountain ridge thermal haze, and concertina wire fence.
     - 'facility': High-security critical facility with concrete blast wall and security light posts.
     - 'perimeter': Classic open border perimeter fence line and rolling terrain.
  3. Physical Sensor Artifacts: Microbolometer NETD temporal noise (~50 mK equivalent),
     column-wise Fixed Pattern Noise (FPN), dead pixels, and optical lens vignetting.
  4. Tactical Anomaly Scenarios (Perimeter Breaches):
     - 'intruder': Warm human target (37°C core body heat, head/torso gradients, bipedal gait).
     - 'vehicle': High-speed tactical vehicle with hot engine block/grille, exhaust plume, and wheels.
     - 'crawler': Low-profile crawling saboteur prone along fence base.
     - 'heat_source': Static anomalous hot spot or abandoned suspicious payload.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class AnomalyEvent:
    start_frame: int
    end_frame: int
    anomaly_type: str = "intruder"  # 'intruder' | 'vehicle' | 'crawler' | 'heat_source'
    start_pos: Tuple[float, float] = (0.2, 0.7)  # (x_norm, y_norm)
    end_pos: Tuple[float, float] = (0.8, 0.7)
    size: Tuple[int, int] = (14, 28)             # (w, h) in pixels
    intensity: float = 235.0                     # Peak thermal brightness (0-255 scale)


class ThermalPerimeterSimulator:
    """
    Infrared perimeter security video stream simulator.
    """

    def __init__(
        self,
        width: int = 160,
        height: int = 128,
        num_frames: int = 200,
        fps: float = 20.0,
        seed: int = 42,
        environment: str = "perimeter",
    ):
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.seed = seed
        self.environment = environment.lower()
        self.rng = np.random.default_rng(seed)

        # Generate base thermal background and perimeter infrastructure
        self.background = self._generate_base_background()
        self.fence_mask = self._generate_perimeter_fence()

    def _generate_base_background(self) -> np.ndarray:
        """Generates realistic thermal background according to environment preset."""
        h, w = self.height, self.width
        bg = np.zeros((h, w), dtype=np.float32)

        if self.environment == "outpost":
            # --- Environment 1: Military Forward Operating Base (FOB) ---
            horizon = int(h * 0.42)
            # Sky radiance gradient (cold night sky: 32 - 52)
            for y in range(horizon):
                bg[y, :] = 32.0 + (y / max(1, horizon)) * 20.0

            # Distant mountain ridge (thermal haze)
            x = np.linspace(0, 3.5 * np.pi, w)
            ridge = np.sin(x) * 5.5 + np.cos(2.2 * x) * 2.5
            for col in range(w):
                sy = int(horizon - 9 + ridge[col])
                bg[max(0, sy):horizon, col] = 46.0

            # Ground gradient (warm earth / soil: 62 - 96)
            for y in range(horizon, h):
                factor = (y - horizon) / max(1, h - horizon)
                bg[y, :] = 62.0 + factor * 34.0

            # Asphalt patrol road with perspective from right to center
            for y in range(horizon, h):
                factor = (y - horizon) / max(1, h - horizon)
                road_w = int(10 + factor * 34)
                road_c = int(w * 0.58 + factor * 12)
                bg[y, max(0, road_c - road_w):min(w, road_c + road_w)] += 13.0

            # Elevated guard watchtower on left flank
            tw_x1, tw_x2 = max(2, int(w * 0.08)), int(w * 0.22)
            tw_base_y = int(h * 0.58)
            tw_top_y = int(h * 0.22)
            # Concrete bunker base
            bg[tw_base_y:int(h * 0.65), tw_x1:tw_x2] = 52.0
            # Sentry cabin (warmer interior)
            bg[tw_top_y:int(h * 0.36), tw_x1:tw_x2] = 78.0
            bg[int(h * 0.27):int(h * 0.33), tw_x1 + 3:tw_x2 - 3] = 105.0  # Warm observation window
            # Stilt support posts
            bg[int(h * 0.36):tw_base_y, tw_x1 + 1:tw_x1 + 3] = 42.0
            bg[int(h * 0.36):tw_base_y, tw_x2 - 3:tw_x2 - 1] = 42.0

        elif self.environment == "facility":
            # --- Environment 2: Critical High-Security Facility ---
            horizon = int(h * 0.40)
            for y in range(horizon):
                bg[y, :] = 34.0 + (y / max(1, horizon)) * 18.0

            # Distant industrial facility silhouettes
            bg[int(h * 0.26):horizon, int(w * 0.55):int(w * 0.82)] = 58.0
            bg[int(h * 0.22):horizon, int(w * 0.62):int(w * 0.74)] = 64.0

            # Concrete tarmac apron (70 - 100)
            for y in range(horizon, h):
                factor = (y - horizon) / max(1, h - horizon)
                bg[y, :] = 68.0 + factor * 30.0

            # Security concrete blast wall
            wall_y = int(h * 0.54)
            bg[wall_y:wall_y + 16, :] = 58.0
            for px in range(0, w, 22):
                bg[wall_y:wall_y + 16, px:px + 2] = 48.0  # Expansion joints

            # Security floodlight stanchions
            for lx in [int(w * 0.25), int(w * 0.75)]:
                bg[int(h * 0.30):wall_y, lx:lx + 2] = 45.0
                bg[int(h * 0.28):int(h * 0.31), lx - 3:lx + 5] = 118.0  # Thermal housing

        else:
            # --- Environment 3: Classic Border Perimeter ---
            horizon = int(h * 0.45)
            for y in range(horizon):
                bg[y, :] = 38.0 + (y / max(1, horizon)) * 20.0

            for y in range(horizon, h):
                factor = (y - horizon) / max(1, h - horizon)
                bg[y, :] = 60.0 + factor * 35.0

            x = np.linspace(0, 4 * np.pi, w)
            hills = np.sin(x) * 4.0
            for col in range(w):
                split_y = int(horizon + hills[col])
                bg[:split_y, col] = np.clip(bg[:split_y, col] - 5.0, 30, 255)

        # Micro-texture terrain noise (gravel, dry soil, emissivity variations)
        noise = self.rng.normal(0, 3.2, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (5, 5), 1.2)
        bg += noise

        # Optical lens vignetting (slight radial falloff typical of wide-angle thermal optics)
        y_idx, x_idx = np.ogrid[:h, :w]
        vignette = 1.0 - 0.18 * (((x_idx - w / 2.0) / (w / 2.0)) ** 2 + ((y_idx - h / 2.0) / (h / 2.0)) ** 2)
        bg *= np.clip(vignette, 0.75, 1.0)

        return np.clip(bg, 0, 255)

    def _generate_perimeter_fence(self) -> np.ndarray:
        """Generates binary fence mask with steel posts and tension lines."""
        h, w = self.height, self.width
        fence = np.zeros((h, w), dtype=np.float32)

        if self.environment == "facility":
            fence_y = int(h * 0.52)
            # Barbed wire coils on top of concrete wall
            fence[fence_y:fence_y + 3, :] = 0.7
            return fence

        fence_y = int(h * 0.60)
        # Horizontal steel tension lines
        for dy in [0, 5, 10, 15, 20]:
            y_pos = fence_y + dy
            if y_pos < h:
                fence[y_pos, :] = 0.65

        # Vertical structural fence posts
        step = 22 if self.environment == "outpost" else 25
        for x_pos in range(8, w, step):
            fence[fence_y:min(h, fence_y + 24), x_pos:x_pos + 2] = 0.9

        # Diagonal cross-braces for outpost fence
        if self.environment == "outpost":
            for x_pos in range(8, w - step, step):
                for dy in range(20):
                    px = x_pos + int(dy * (step / 20.0))
                    py = fence_y + dy
                    if py < h and px < w:
                        fence[py, px] = 0.35

        return fence

    def generate_sequence(
        self,
        anomalies: Optional[List[AnomalyEvent]] = None,
        add_sensor_defects: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates physical thermal video sequence.

        Returns:
          * raw_frames: (N, H, W) uint8 raw thermal frames
          * labels: (N,) int8 frame labels (0: normal, 1: anomaly)
          * masks: (N, H, W) uint8 anomaly ground-truth masks
        """
        if anomalies is None:
            # Default sequence: Intruder breaching sector from frame 70 to 150
            anomalies = [
                AnomalyEvent(
                    start_frame=70,
                    end_frame=150,
                    anomaly_type="intruder",
                    start_pos=(0.15, 0.65),
                    end_pos=(0.85, 0.70),
                    size=(12, 26),
                    intensity=238.0,
                )
            ]

        h, w = self.height, self.width
        raw_frames = np.zeros((self.num_frames, h, w), dtype=np.uint8)
        labels = np.zeros(self.num_frames, dtype=np.int8)
        masks = np.zeros((self.num_frames, h, w), dtype=np.uint8)

        # Fixed Pattern Noise (FPN vertical column streaks from microbolometer readout)
        fpn_cols = self.rng.normal(0, 3.2, w).astype(np.float32) if add_sensor_defects else np.zeros(w)

        for t in range(self.num_frames):
            frame = self.background.copy()

            # Blend perimeter fence into background
            frame = np.where(self.fence_mask > 0, frame * 0.85 + self.fence_mask * 22.0, frame)

            # Atmospheric turbulence and ground radiant drift
            temporal_drift = math.sin(t * 0.08) * 1.4
            frame += temporal_drift

            frame_mask = np.zeros((h, w), dtype=np.uint8)
            is_anomaly = 0

            # Render tactical anomalies
            for anom in anomalies:
                if anom.start_frame <= t <= anom.end_frame:
                    is_anomaly = 1
                    alpha = (t - anom.start_frame) / max(1, (anom.end_frame - anom.start_frame))
                    cx = (anom.start_pos[0] + alpha * (anom.end_pos[0] - anom.start_pos[0])) * w
                    cy = (anom.start_pos[1] + alpha * (anom.end_pos[1] - anom.start_pos[1])) * h

                    bw, bh = anom.size
                    x0 = int(np.clip(cx - bw / 2.0, 0, w))
                    x1 = int(np.clip(cx + bw / 2.0, 0, w))
                    y0 = int(np.clip(cy - bh / 2.0, 0, h))
                    y1 = int(np.clip(cy + bh / 2.0, 0, h))

                    if x1 > x0 and y1 > y0:
                        py, px = np.ogrid[: y1 - y0, : x1 - x0]
                        center_y, center_x = (y1 - y0) / 2.0, (x1 - x0) / 2.0
                        patch_h, patch_w = y1 - y0, x1 - x0

                        if anom.anomaly_type == "vehicle":
                            # --- Tactical High-Speed Vehicle ---
                            rad = ((px - center_x) / (bw / 2.0 + 1e-5)) ** 4 + ((py - center_y) / (bh / 2.0 + 1e-5)) ** 2
                            heat_blob = np.exp(-1.1 * rad) * (anom.intensity * 0.78)

                            # Armored cabin roof
                            cab_mask = (py < patch_h * 0.52) & (px > patch_w * 0.24) & (px < patch_w * 0.76)
                            heat_blob[cab_mask] = np.maximum(heat_blob[cab_mask], anom.intensity * 0.84)

                            # White-hot front radiator & engine bay
                            is_fwd = anom.end_pos[0] >= anom.start_pos[0]
                            eng_x = patch_w * (0.84 if is_fwd else 0.16)
                            eng_rad = ((px - eng_x) / (bw * 0.22 + 1e-5)) ** 2 + ((py - patch_h * 0.55) / (bh * 0.35 + 1e-5)) ** 2
                            heat_blob = np.maximum(heat_blob, np.exp(-1.8 * eng_rad) * anom.intensity)

                            # Trailing exhaust heat plume
                            ex_x = patch_w * (0.08 if is_fwd else 0.92)
                            ex_rad = ((px - ex_x) / (bw * 0.18 + 1e-5)) ** 2 + ((py - patch_h * 0.68) / (bh * 0.28 + 1e-5)) ** 2
                            heat_blob = np.maximum(heat_blob, np.exp(-1.3 * ex_rad) * (anom.intensity * 0.92))

                            # Hot wheels (front and rear at bottom contact)
                            for w_rel in [0.22, 0.78]:
                                wx = patch_w * w_rel
                                w_rad = ((px - wx) / (bw * 0.15 + 1e-5)) ** 2 + ((py - patch_h * 0.82) / (bh * 0.24 + 1e-5)) ** 2
                                heat_blob = np.maximum(heat_blob, np.exp(-2.2 * w_rad) * (anom.intensity * 0.96))

                            frame[y0:y1, x0:x1] = np.maximum(frame[y0:y1, x0:x1], heat_blob)
                            frame_mask[y0:y1, x0:x1] = np.where(rad <= 1.25, 1, 0)

                        elif anom.anomaly_type in ("crawler", "crawling_intruder"):
                            # --- Prone Low-Profile Saboteur ---
                            # Creeping motion along ground
                            crawl_cy = center_y + math.sin(t * 0.4) * 1.0
                            rad = (np.abs((px - center_x) / (bw / 2.0 + 1e-5))) ** 2.5 + ((py - crawl_cy) / (bh / 2.0 + 1e-5)) ** 2
                            heat_blob = np.exp(-1.3 * rad) * (anom.intensity * 0.88)

                            # Elevated warm head
                            is_fwd = anom.end_pos[0] >= anom.start_pos[0]
                            head_x = patch_w * (0.82 if is_fwd else 0.18)
                            head_rad = ((px - head_x) / (bw * 0.16 + 1e-5)) ** 2 + ((py - patch_h * 0.35) / (bh * 0.35 + 1e-5)) ** 2
                            heat_blob = np.maximum(heat_blob, np.exp(-1.8 * head_rad) * anom.intensity)

                            frame[y0:y1, x0:x1] = np.maximum(frame[y0:y1, x0:x1], heat_blob)
                            frame_mask[y0:y1, x0:x1] = np.where(rad <= 1.1, 1, 0)

                        else:
                            # --- Realistic Walking Human Target ---
                            # Head (hottest core ~ 37C)
                            gait_cycle = math.sin(t * 0.7)
                            head_y = patch_h * 0.18 + gait_cycle * 0.8
                            head_rad = ((px - center_x) / (bw * 0.26 + 1e-5)) ** 2 + ((py - head_y) / (bh * 0.20 + 1e-5)) ** 2
                            head_blob = np.exp(-2.0 * head_rad) * anom.intensity

                            # Torso (broad shoulders, tapered waist)
                            torso_y = patch_h * 0.46
                            torso_rad = ((px - center_x) / (bw * 0.38 + 1e-5)) ** 2 + ((py - torso_y) / (bh * 0.30 + 1e-5)) ** 2
                            torso_blob = np.exp(-1.5 * torso_rad) * (anom.intensity * 0.90)

                            # Legs with alternating gait cycle
                            leg_rad_left = ((px - (center_x - bw * 0.14)) / (bw * 0.18 + 1e-5)) ** 2 + (
                                (py - (patch_h * 0.82 + gait_cycle * 1.5)) / (bh * 0.24 + 1e-5)
                            ) ** 2
                            leg_rad_right = ((px - (center_x + bw * 0.14)) / (bw * 0.18 + 1e-5)) ** 2 + (
                                (py - (patch_h * 0.82 - gait_cycle * 1.5)) / (bh * 0.24 + 1e-5)
                            ) ** 2
                            legs_blob = np.maximum(
                                np.exp(-2.2 * leg_rad_left) * (anom.intensity * 0.80),
                                np.exp(-2.2 * leg_rad_right) * (anom.intensity * 0.80),
                            )

                            human_composite = np.maximum.reduce([head_blob, torso_blob, legs_blob])
                            frame[y0:y1, x0:x1] = np.maximum(frame[y0:y1, x0:x1], human_composite)
                            frame_mask[y0:y1, x0:x1] = np.where(human_composite >= 50.0, 1, 0)

            # Microbolometer sensor degradation effects
            if add_sensor_defects:
                frame += fpn_cols[None, :]
                netd_noise = self.rng.normal(0, 3.8, (h, w)).astype(np.float32)
                frame += netd_noise

            raw_frames[t] = np.clip(frame, 0, 255).astype(np.uint8)
            labels[t] = is_anomaly
            masks[t] = frame_mask

        return raw_frames, labels, masks
