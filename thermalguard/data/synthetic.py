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
            # --- Environment 1: Military Forward Operating Base (FOB) Perimeter ---
            # Layout: Sentry mast view looking at perimeter fence and horizontal patrol road
            horizon = 48
            for y in range(horizon):
                bg[y, :] = 28.0 + (y / horizon) * 16.0

            # Distant layered mountain ridges
            x_coords = np.linspace(0, 3.5 * np.pi, w)
            ridge1 = np.sin(x_coords) * 5.0 + np.cos(2.2 * x_coords) * 2.5
            for col in range(w):
                sy = int(horizon - 10 + ridge1[col])
                bg[max(0, sy):horizon, col] = 38.0

            # Ground between horizon and road (y=48..84)
            for y in range(horizon, 84):
                f = (y - horizon) / (84 - horizon)
                bg[y, :] = 50.0 + f * 15.0

            # Elevated guard watchtower behind fence on left flank (x=16..36)
            tw_x0, tw_x1 = 16, 36
            tw_base_y, tw_top_y = 70, 24
            bg[tw_top_y:tw_top_y + 14, tw_x0:tw_x1] = 62.0
            bg[tw_top_y - 2:tw_top_y, tw_x0 - 2:tw_x1 + 2] = 46.0
            bg[tw_top_y + 4:tw_top_y + 8, tw_x0 + 3:tw_x1 - 3] = 95.0
            for px in [tw_x0 + 2, tw_x1 - 3]:
                bg[tw_top_y + 14:tw_base_y, px:px + 2] = 42.0

            # Fence line at y=70 behind road
            fence_y = 70
            bg[fence_y, :] = 75.0
            bg[fence_y + 4, :] = 70.0
            for col in range(0, w, 18):
                bg[fence_y - 4:fence_y + 12, col:col + 2] = 88.0

            # CLEAR HORIZONTAL PATROL ROAD (y=84 to y=116)
            road_top, road_bottom = 84, 116
            for y in range(road_top, road_bottom):
                dist_from_c = abs(y - (road_top + road_bottom) / 2.0) / ((road_bottom - road_top) / 2.0)
                bg[y, :] = 78.0 - dist_from_c * 12.0

            # Tire ruts along horizontal road
            bg[92:94, :] -= 5.0
            bg[106:108, :] -= 5.0

            # Foreground embankment below road (y=116..128)
            for y in range(road_bottom, h):
                f = (y - road_bottom) / max(1, h - road_bottom)
                bg[y, :] = 60.0 + f * 8.0

        elif self.environment == "facility":
            # --- Environment 2: Critical High-Security Facility Perimeter Wall ---
            horizon = 48
            for y in range(horizon):
                bg[y, :] = 30.0 + (y / horizon) * 16.0

            # Distant facility roofline behind wall
            bg[int(h * 0.28):horizon, int(w * 0.40):int(w * 0.75)] = 54.0

            # Concrete perimeter wall (y=48 to y=76)
            wall_top, wall_base = 48, 76
            bg[wall_top:wall_base, :] = 58.0
            bg[wall_top:wall_top + 3, :] = 50.0  # Coping cap
            bg[wall_base - 2:wall_base + 1, :] = 66.0  # Footing

            # Thin expansion seams every 36px (realistic concrete panels)
            for px in range(12, w, 36):
                bg[wall_top:wall_base, px] = 46.0

            # Concertina razor wire coiled along the top of the wall
            for col in range(w):
                coil_y = int(wall_top - 4 + math.sin(col * 0.75) * 3.0)
                if 0 <= coil_y < h:
                    bg[coil_y, col] = 76.0

            # Crushed gravel security apron in foreground (y=76..128)
            for y in range(wall_base, h):
                f = (y - wall_base) / max(1, h - wall_base)
                bg[y, :] = 68.0 + f * 10.0

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

    def _render_crawler_patch(
        self, t: int, patch_w: int, patch_h: int, intensity: float, is_forward: bool
    ) -> np.ndarray:
        """Renders an unmistakable human soldier in prone commando crawl (leopard low-crawl)."""
        patch = np.zeros((patch_h, patch_w), dtype=np.float32)
        py, px = np.ogrid[:patch_h, :patch_w]

        # Crawl cycle: 16 frames per complete stride
        cycle = (t % 16) / 16.0
        stride_phase = math.sin(cycle * 2 * math.pi)
        stride_cos = math.cos(cycle * 2 * math.pi)

        dir_sign = 1.0 if is_forward else -1.0
        center_x = patch_w * 0.50

        # 1. Prone Torso (Fabric camouflage thermal signature ~145 DN)
        torso_y = patch_h * 0.54 + stride_phase * 0.6
        torso_cx = center_x - dir_sign * patch_w * 0.04
        torso_rad = ((px - torso_cx) / (patch_w * 0.24 + 1e-5)) ** 2 + ((py - torso_y) / (patch_h * 0.22 + 1e-5)) ** 2
        patch = np.maximum(patch, np.exp(-1.3 * torso_rad) * (intensity * 0.62))

        # 2. Elevated Head & Tactical Helmet
        head_cx = center_x + dir_sign * patch_w * 0.32
        head_cy = patch_h * 0.38 + stride_cos * 0.8
        head_rad = ((px - head_cx) / (patch_w * 0.11 + 1e-5)) ** 2 + ((py - head_cy) / (patch_h * 0.20 + 1e-5)) ** 2
        head_heat = np.exp(-1.8 * head_rad) * (intensity * 0.92)
        helm_mask = (py < head_cy) & (px > head_cx - patch_w * 0.08) & (px < head_cx + patch_w * 0.08)
        head_heat[helm_mask] *= 0.82
        patch = np.maximum(patch, head_heat)

        # 3. Arms: One arm reaching forward, one arm planted under chest
        arm_reach = stride_phase * patch_w * 0.14
        lead_elbow_x = center_x + dir_sign * (patch_w * 0.20 + arm_reach)
        lead_elbow_y = patch_h * 0.64 + max(0.0, -stride_phase) * 1.6
        arm1_rad = ((px - lead_elbow_x) / (patch_w * 0.10 + 1e-5)) ** 2 + ((py - lead_elbow_y) / (patch_h * 0.16 + 1e-5)) ** 2
        patch = np.maximum(patch, np.exp(-2.0 * arm1_rad) * (intensity * 0.74))

        trail_elbow_x = center_x + dir_sign * (patch_w * 0.06 - arm_reach * 0.7)
        trail_elbow_y = patch_h * 0.68 + max(0.0, stride_phase) * 1.6
        arm2_rad = ((px - trail_elbow_x) / (patch_w * 0.10 + 1e-5)) ** 2 + ((py - trail_elbow_y) / (patch_h * 0.16 + 1e-5)) ** 2
        patch = np.maximum(patch, np.exp(-2.0 * arm2_rad) * (intensity * 0.70))

        # 4. Pelvis & Low Waist
        pelvis_cx = center_x - dir_sign * patch_w * 0.18
        pelvis_cy = patch_h * 0.58
        pelvis_rad = ((px - pelvis_cx) / (patch_w * 0.16 + 1e-5)) ** 2 + ((py - pelvis_cy) / (patch_h * 0.20 + 1e-5)) ** 2
        patch = np.maximum(patch, np.exp(-1.5 * pelvis_rad) * (intensity * 0.59))

        # 5. Distinct Bent Knee (V-Shape outward crawl kick) & Trailing Boot
        knee_push = stride_cos * patch_w * 0.09
        knee_x = center_x - dir_sign * (patch_w * 0.30 - knee_push)
        knee_y = patch_h * 0.70 + stride_phase * 1.4
        knee_rad = ((px - knee_x) / (patch_w * 0.12 + 1e-5)) ** 2 + ((py - knee_y) / (patch_h * 0.17 + 1e-5)) ** 2
        patch = np.maximum(patch, np.exp(-2.0 * knee_rad) * (intensity * 0.68))

        boot_x = center_x - dir_sign * (patch_w * 0.44 + stride_cos * patch_w * 0.06)
        boot_y = patch_h * 0.66 - stride_phase * 1.0
        boot_rad = ((px - boot_x) / (patch_w * 0.10 + 1e-5)) ** 2 + ((py - boot_y) / (patch_h * 0.14 + 1e-5)) ** 2
        patch = np.maximum(patch, np.exp(-2.2 * boot_rad) * (intensity * 0.64))

        return cv2.GaussianBlur(patch, (3, 3), 0.7)

    def _render_vehicle_patch(
        self, t: int, patch_w: int, patch_h: int, intensity: float, is_forward: bool
    ) -> np.ndarray:
        """Renders an authentic, clearly visible military tactical patrol truck / Humvee in LWIR."""
        patch = np.zeros((patch_h, patch_w), dtype=np.float32)
        py, px = np.ogrid[:patch_h, :patch_w]

        # 1. Main Truck Body / Hull (Solid, clearly defined military profile ~145 DN)
        hull_mask = (py >= patch_h * 0.35) & (py <= patch_h * 0.72) & (px >= patch_w * 0.06) & (px <= patch_w * 0.94)
        patch[hull_mask] = intensity * 0.58

        # 2. Raised Armored Crew Cabin (Central / Rear)
        cab_x0 = int(patch_w * (0.24 if is_forward else 0.30))
        cab_x1 = int(patch_w * (0.68 if is_forward else 0.76))
        cab_mask = (py >= patch_h * 0.16) & (py <= patch_h * 0.42) & (px >= cab_x0) & (px <= cab_x1)
        patch[cab_mask] = intensity * 0.54

        # Windshield (sloped glass, cooler)
        ws_x0 = int(patch_w * (0.62 if is_forward else 0.30))
        ws_x1 = int(patch_w * (0.70 if is_forward else 0.38))
        ws_mask = (py >= patch_h * 0.18) & (py <= patch_h * 0.38) & (px >= ws_x0) & (px <= ws_x1)
        patch[ws_mask] = intensity * 0.38

        # Side window
        win_x0 = int(patch_w * (0.34 if is_forward else 0.44))
        win_x1 = int(patch_w * (0.58 if is_forward else 0.66))
        win_mask = (py >= patch_h * 0.20) & (py <= patch_h * 0.36) & (px >= win_x0) & (px <= win_x1)
        patch[win_mask] = intensity * 0.42

        # 3. Front Engine Bay & White-Hot Radiator Grill (235-255 DN)
        eng_x0 = int(patch_w * (0.84 if is_forward else 0.06))
        eng_x1 = int(patch_w * (0.95 if is_forward else 0.18))
        patch[int(patch_h * 0.38):int(patch_h * 0.68), eng_x0:eng_x1] = intensity * 0.95

        for gx in range(eng_x0, eng_x1, 2):
            if 0 <= gx < patch_w:
                patch[int(patch_h * 0.42):int(patch_h * 0.64), gx] = intensity

        hood_x0 = int(patch_w * (0.68 if is_forward else 0.18))
        hood_x1 = int(patch_w * (0.84 if is_forward else 0.34))
        patch[int(patch_h * 0.34):int(patch_h * 0.40), hood_x0:hood_x1] = intensity * 0.75

        # 4. Wheels & Wheel Arches
        w_front = patch_w * (0.78 if is_forward else 0.22)
        w_rear = patch_w * (0.24 if is_forward else 0.78)
        wheel_y = patch_h * 0.76

        for wx in [w_front, w_rear]:
            arch_rad = ((px - wx) / (patch_w * 0.14 + 1e-5)) ** 2 + ((py - (wheel_y - 2)) / (patch_h * 0.22 + 1e-5)) ** 2
            patch[arch_rad <= 1.0] = np.minimum(patch[arch_rad <= 1.0], 55.0)

            tire_rad = ((px - wx) / (patch_w * 0.11 + 1e-5)) ** 2 + ((py - wheel_y) / (patch_h * 0.18 + 1e-5)) ** 2
            tire_mask = (tire_rad <= 1.0) & (tire_rad >= 0.28)
            patch[tire_mask] = intensity * 0.78

            hub_rad = ((px - wx) / (patch_w * 0.05 + 1e-5)) ** 2 + ((py - wheel_y) / (patch_h * 0.09 + 1e-5)) ** 2
            patch = np.maximum(patch, np.exp(-2.2 * hub_rad) * (intensity * 0.98))

        # 5. Hot Exhaust System & Plume
        ex_x = patch_w * (0.05 if is_forward else 0.95)
        ex_y = patch_h * 0.68
        ex_rad = ((px - ex_cx if (ex_cx := ex_x) else 0) / (patch_w * 0.06 + 1e-5)) ** 2 + ((py - ex_y) / (patch_h * 0.09 + 1e-5)) ** 2
        patch = np.maximum(patch, np.exp(-1.8 * ex_rad) * intensity)

        plume_dir = -1.0 if is_forward else 1.0
        for step in range(1, 12):
            px_c = ex_x + plume_dir * step * (patch_w * 0.038)
            py_c = ex_y - step * 0.5 + math.sin(t * 0.9 + step * 0.8) * 1.4
            p_rad = ((px - px_c) / (patch_w * 0.05 + step * 0.8 + 1e-5)) ** 2 + ((py - py_c) / (patch_h * 0.08 + step * 0.5 + 1e-5)) ** 2
            fade = max(0.0, 1.0 - step / 11.0) ** 1.3
            patch = np.maximum(patch, np.exp(-1.5 * p_rad) * (intensity * 0.85 * fade))

        return cv2.GaussianBlur(patch, (3, 3), 0.7)

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

                        is_fwd = anom.end_pos[0] >= anom.start_pos[0]
                        if anom.anomaly_type == "vehicle":
                            # --- Tactical Mechanized Patrol Vehicle (Humvee / Technical) ---
                            full_patch = self._render_vehicle_patch(t, bw, bh, anom.intensity, is_fwd)
                            px0 = x0 - int(cx - bw / 2.0)
                            px1 = px0 + (x1 - x0)
                            py0 = y0 - int(cy - bh / 2.0)
                            py1 = py0 + (y1 - y0)
                            heat_patch = full_patch[py0:py1, px0:px1]

                            frame[y0:y1, x0:x1] = np.maximum(frame[y0:y1, x0:x1], heat_patch)
                            frame_mask[y0:y1, x0:x1] = np.where(heat_patch >= 60.0, 1, 0)

                        elif anom.anomaly_type in ("crawler", "crawling_intruder"):
                            # --- Tactical Prone Crawling Saboteur (Commando Low-Crawl) ---
                            full_patch = self._render_crawler_patch(t, bw, bh, anom.intensity, is_fwd)
                            px0 = x0 - int(cx - bw / 2.0)
                            px1 = px0 + (x1 - x0)
                            py0 = y0 - int(cy - bh / 2.0)
                            py1 = py0 + (y1 - y0)
                            heat_patch = full_patch[py0:py1, px0:px1]

                            frame[y0:y1, x0:x1] = np.maximum(frame[y0:y1, x0:x1], heat_patch)
                            frame_mask[y0:y1, x0:x1] = np.where(heat_patch >= 50.0, 1, 0)

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
