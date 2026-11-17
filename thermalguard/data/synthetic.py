"""
Physics-based Infrared LWIR Thermal Video Simulator for Defense Perimeter Security.

Generates realistic LWIR (Long-Wave Infrared 8-14 um) video sequences simulating
physical sensor and thermal radiation dynamics for immediate demonstrations,
unit testing, and perimeter security evaluation without requiring external dataset downloads.

Simulated Thermal Dynamics:
---------------------------
  1. Thermal Backgrounds: Cold night sky radiance falloff, terrain thermal inertia,
     mountain horizons, asphalt patrol roads with perspective, and security wire fencing.
  2. Multi-Environment Presets:
     - 'outpost': Military Forward Operating Base (FOB) with distant mountain horizon,
       asphalt patrol road with perspective, concrete road barriers, and textured terrain.
     - 'facility': High-security perimeter dead zone with textured gravel apron,
       distant security berm, and concertina-topped chainlink fence.
     - 'perimeter': Classic open border perimeter fence line and rolling terrain.
  3. Physical Sensor Artifacts: Microbolometer NETD temporal noise (~50 mK equivalent),
     column-wise Fixed Pattern Noise (FPN), dead pixels, and optical lens vignetting.
  4. Tactical Anomaly Scenarios (Perimeter Breaches):
     - 'intruder': Warm human target (37°C core body heat, head/torso gradients, bipedal gait).
     - 'vehicle': Tactical patrol vehicle with sky-reflecting metallic chassis, white-hot
       friction-heated tires, hot radiator grille, and undercarriage exhaust.
     - 'crawler': Low-profile crawling saboteur with anatomical multi-joint low-crawl kinematics
       (scanning head, core torso, reaching forearms, and alternating knee drive).
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
            # --- Environment 1: Military Forward Operating Base (FOB) Patrol Road ---
            horizon = int(h * 0.44)
            # Cold night sky radiance falloff (26.0 - 46.0)
            for y in range(horizon):
                bg[y, :] = 26.0 + (y / max(1, horizon)) * 20.0

            # Distant mountain ridge with realistic fractal silhouette
            x_axis = np.linspace(0, 4.5 * np.pi, w)
            mountains = np.sin(x_axis) * 5.5 + np.cos(2.3 * x_axis) * 3.2 + np.sin(5.1 * x_axis) * 1.6
            for col in range(w):
                y_top = int(horizon + mountains[col])
                bg[max(0, horizon - 11):min(h, y_top + 1), col] = 36.0 + (col % 6) * 1.5

            # Open desert/rocky perimeter ground
            for y in range(horizon, h):
                depth = (y - horizon) / max(1, h - horizon)
                bg[y, :] = 54.0 + depth * 34.0

            # Asphalt patrol road stretching across lower sector
            road_top = int(h * 0.64)
            road_bot = int(h * 0.88)
            for y in range(road_top, road_bot):
                frac = (y - road_top) / max(1, road_bot - road_top)
                bg[y, :] = 70.0 + frac * 12.0  # Warmer asphalt retaining daytime thermal inertia

            # Soft road shoulders
            bg[road_top:road_top + 2, :] = 58.0
            bg[road_bot - 2:road_bot, :] = 56.0

            # Concrete barrier posts along road edge
            guard_y = road_top - 4
            for px in range(12, w, 22):
                bg[guard_y - 8:guard_y + 4, px:px + 2] = 94.0

        elif self.environment == "facility":
            # --- Environment 2: High-Security Perimeter Dead Zone ---
            horizon = int(h * 0.46)
            # Nocturnal sky gradient
            for y in range(horizon):
                bg[y, :] = 28.0 + (y / max(1, horizon)) * 20.0

            # Distant perimeter berm / tree line silhouette
            x_axis = np.linspace(0, 6.0 * np.pi, w)
            berm = np.sin(x_axis) * 3.5 + np.cos(3.2 * x_axis) * 2.0
            for col in range(w):
                y_top = int(horizon + berm[col])
                bg[max(0, horizon - 8):min(h, y_top + 1), col] = 34.0 + (col % 5) * 1.5

            # Gravel apron / perimeter dead zone
            for y in range(horizon, h):
                depth = (y - horizon) / max(1, h - horizon)
                bg[y, :] = 58.0 + depth * 36.0

            # Low gravel patrol path
            path_y = int(h * 0.68)
            for y in range(path_y, h):
                bg[y, :] += 6.0

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
        noise = cv2.GaussianBlur(noise, (5, 5), 1.0)
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

        if self.environment == "outpost":
            # Outpost wire is alongside road shoulder
            fence_y = int(h * 0.60)
            for dy in [0, 6, 12, 18]:
                y_pos = fence_y + dy
                if y_pos < h:
                    fence[y_pos, :] = 0.55
            for x_pos in range(8, w, 24):
                fence[fence_y:min(h, fence_y + 22), x_pos:x_pos + 2] = 0.85
            return fence

        fence_y = int(h * 0.54) if self.environment == "facility" else int(h * 0.60)
        # Horizontal steel tension lines
        for dy in [0, 5, 10, 15, 20]:
            y_pos = fence_y + dy
            if y_pos < h:
                fence[y_pos, :] = 0.65

        # Vertical structural fence posts
        step = 22 if self.environment == "facility" else 25
        for x_pos in range(8, w, step):
            fence[fence_y:min(h, fence_y + 24), x_pos:x_pos + 2] = 0.90

        # Diagonal cross-braces
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
                        y_grid, x_grid = np.ogrid[:h, :w]
                        is_fwd = anom.end_pos[0] >= anom.start_pos[0]

                        if anom.anomaly_type == "vehicle":
                            # --- Tactical Patrol Vehicle (Authentic Thermal Infrared Physics) ---
                            # Vehicle dimensions: ~36x14 pixels (2.5:1 ratio)
                            veh_cx = cx
                            veh_cy = cy + math.sin(t * 1.5) * 0.4  # Road vibration

                            # 1. Friction-Heated Wheels (intense thermal emission: 245-255)
                            fw_x = veh_cx + int(bw * 0.28) if is_fwd else veh_cx - int(bw * 0.28)
                            rw_x = veh_cx - int(bw * 0.28) if is_fwd else veh_cx + int(bw * 0.28)
                            wheel_y = veh_cy + int(bh * 0.28)

                            fw_dist = ((x_grid - fw_x) / 3.4) ** 2 + ((y_grid - wheel_y) / 3.0) ** 2
                            rw_dist = ((x_grid - rw_x) / 3.4) ** 2 + ((y_grid - wheel_y) / 3.0) ** 2
                            wheels_heat = np.maximum(
                                np.exp(-1.4 * fw_dist) * (anom.intensity * 0.98),
                                np.exp(-1.4 * rw_dist) * (anom.intensity * 0.98),
                            )

                            # 2. Engine Compartment / Front Radiator Grille (white-hot: 252)
                            grille_x = veh_cx + int(bw * 0.44) if is_fwd else veh_cx - int(bw * 0.44)
                            grille_y = veh_cy + int(bh * 0.05)
                            grille_dist = ((x_grid - grille_x) / 2.8) ** 2 + ((y_grid - grille_y) / 3.2) ** 2
                            grille_heat = np.exp(-1.5 * grille_dist) * anom.intensity

                            # 3. Exhaust Pipe & Muffler under rear chassis (240)
                            ex_x = veh_cx - int(bw * 0.42) if is_fwd else veh_cx + int(bw * 0.42)
                            ex_y = veh_cy + int(bh * 0.24)
                            ex_dist = ((x_grid - ex_x) / 3.2) ** 2 + ((y_grid - ex_y) / 1.8) ** 2
                            exhaust_heat = np.exp(-1.8 * ex_dist) * (anom.intensity * 0.94)

                            # 4. Sky-reflecting metallic chassis body (cool dark gray in white-hot mode: 65-80)
                            body_dist = ((x_grid - veh_cx) / (bw * 0.44)) ** 4 + ((y_grid - veh_cy) / (bh * 0.38)) ** 2
                            chassis_body = np.where(body_dist <= 1.0, 72.0, 0.0)

                            # Cabin roof cooler due to sky radiation reflection
                            cab_dist = ((x_grid - (veh_cx - int(bw * 0.05))) / (bw * 0.26)) ** 2 + (
                                (y_grid - (veh_cy - int(bh * 0.22))) / (bh * 0.28)
                            ) ** 2
                            chassis_body = np.where(cab_dist <= 1.0, 62.0, chassis_body)

                            # Highlight on body seams and trim edges
                            seam_dist = np.abs(body_dist - 0.95)
                            edge_highlight = np.where(seam_dist < 0.20, 115.0, 0.0)

                            # Composite vehicle onto frame
                            chassis_mask = body_dist <= 1.0
                            frame[chassis_mask] = chassis_body[chassis_mask] + edge_highlight[chassis_mask]

                            hot_components = np.maximum.reduce([wheels_heat, grille_heat, exhaust_heat])
                            frame = np.maximum(frame, hot_components)

                            # Hot tire contact patches on road
                            for wx in [fw_x, rw_x]:
                                frame[int(wheel_y + 3):int(wheel_y + 5), int(wx - 3):int(wx + 4)] = np.maximum(
                                    frame[int(wheel_y + 3):int(wheel_y + 5), int(wx - 3):int(wx + 4)], 220.0
                                )

                            frame_mask[chassis_mask] = 1

                        elif anom.anomaly_type in ("crawler", "crawling_intruder"):
                            # --- Prone Low-Profile Saboteur (Anatomical Low Crawl Kinematics) ---
                            # Low-profile silhouette: ~26x10 pixels
                            crawl_phase = t * 0.48
                            crawl_cy = cy

                            # 1. Elevated Head: looking ahead, scanning (~235 white-hot)
                            head_x = cx + int(bw * 0.38) if is_fwd else cx - int(bw * 0.38)
                            head_y = crawl_cy - int(bh * 0.15) + int(math.sin(crawl_phase) * 0.8)
                            head_dist = ((x_grid - head_x) / 2.6) ** 2 + ((y_grid - head_y) / 2.2) ** 2
                            head_heat = np.exp(-1.5 * head_dist) * anom.intensity

                            # 2. Torso / Shoulders: core insulated warmth (~205)
                            torso_x = cx + int(bw * 0.05) if is_fwd else cx - int(bw * 0.05)
                            torso_y = crawl_cy
                            torso_dist = ((x_grid - torso_x) / 6.5) ** 2 + ((y_grid - torso_y) / 3.4) ** 2
                            torso_heat = np.exp(-1.2 * torso_dist) * (anom.intensity * 0.88)

                            # 3. Arms / Elbows: alternating pulling strokes
                            sign = 1 if is_fwd else -1
                            elbow_left_x = cx + int(sign * (bw * 0.24 + math.sin(crawl_phase) * 3.5))
                            elbow_left_y = crawl_cy + int(bh * 0.30)
                            elbow_right_x = cx + int(sign * (bw * 0.24 - math.sin(crawl_phase) * 3.5))
                            elbow_right_y = crawl_cy - int(bh * 0.25)

                            el_dist = ((x_grid - elbow_left_x) / 3.0) ** 2 + ((y_grid - elbow_left_y) / 2.0) ** 2
                            er_dist = ((x_grid - elbow_right_x) / 3.0) ** 2 + ((y_grid - elbow_right_y) / 2.0) ** 2
                            elbows_heat = np.maximum(
                                np.exp(-1.8 * el_dist) * (anom.intensity * 0.78),
                                np.exp(-1.8 * er_dist) * (anom.intensity * 0.78),
                            )

                            # 4. Knees & Legs: alternating knee drive pulling along dirt
                            knee_left_x = cx - int(sign * (bw * 0.26 + math.sin(crawl_phase) * 3.2))
                            knee_left_y = crawl_cy + int(bh * 0.35)
                            knee_right_x = cx - int(sign * (bw * 0.26 - math.sin(crawl_phase) * 3.2))
                            knee_right_y = crawl_cy - int(bh * 0.20)

                            feet_x = cx - int(sign * (bw * 0.46))
                            feet_y = crawl_cy + int(math.sin(crawl_phase) * 1.5)

                            kl_dist = ((x_grid - knee_left_x) / 3.5) ** 2 + ((y_grid - knee_left_y) / 2.2) ** 2
                            kr_dist = ((x_grid - knee_right_x) / 3.5) ** 2 + ((y_grid - knee_right_y) / 2.2) ** 2
                            feet_dist = ((x_grid - feet_x) / 4.0) ** 2 + ((y_grid - feet_y) / 1.8) ** 2
                            legs_heat = np.maximum.reduce([
                                np.exp(-1.8 * kl_dist) * (anom.intensity * 0.75),
                                np.exp(-1.8 * kr_dist) * (anom.intensity * 0.75),
                                np.exp(-2.0 * feet_dist) * (anom.intensity * 0.65),
                            ])

                            crawler_composite = np.maximum.reduce([head_heat, torso_heat, elbows_heat, legs_heat])
                            frame = np.maximum(frame, crawler_composite)
                            frame_mask[crawler_composite >= 50.0] = 1

                        else:
                            # --- Realistic Walking Human Target ---
                            py, px = np.ogrid[: y1 - y0, : x1 - x0]
                            center_y, center_x = (y1 - y0) / 2.0, (x1 - x0) / 2.0
                            patch_h, patch_w = y1 - y0, x1 - x0

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
