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
            # --- Environment 2: Tactical Perimeter Breach Sector (Crawl Cover -> Open Sprint) ---
            # Left Sector (x=0..72): Elevated razor-wire & sensor beam barrier with low crawl gap below
            # Right Sector (x=74..160): Open approach corridor leading to facility entrance gate
            horizon = 46
            for y in range(horizon):
                bg[y, :] = 32.0 + (y / horizon) * 16.0

            # Rolling distant ridge across horizon (natural LWIR terrain profile)
            x_coords = np.linspace(0, 3.2 * np.pi, w)
            ridge = np.sin(x_coords + 0.6) * 3.5 + np.cos(2.0 * x_coords) * 1.8
            for col in range(w):
                sy = int(horizon - 6 + ridge[col])
                bg[max(0, sy):horizon, col] = 44.0

            # Ground thermal gradient from horizon to foreground (y=46..128: 56 -> 88 DN)
            for y in range(horizon, h):
                f = (y - horizon) / max(1, h - horizon)
                bg[y, :] = 56.0 + f * 30.0

            # Background perimeter fence line running across mid-ground (y=62)
            bg[62, :128] = 68.0
            bg[66, :128] = 65.0
            for px in range(10, 128, 22):
                bg[56:70, px:px + 2] = 74.0

            # Right Flank: High-Security Facility Gatehouse & Perimeter Compound (x=120..159)
            # Wall body matched to mid-frame thermal median (~65 DN) to prevent FPN destripe column halos
            bg[38:70, 124:158] = 65.0           # Main concrete gatehouse facade
            bg[35:38, 122:159] = 53.0           # Roof overhang cornice (3px horizontal trim)
            bg[70:72, 124:158] = 74.0           # Concrete foundation plinth
            bg[46:72, 118:121] = 74.0           # Entrance gate pillar
            bg[43:48, 130:145] = 104.0          # Horizontal guardhouse observation window (5px tall)
            bg[52:70, 133:142] = 71.0           # Steel personnel entry door
            # Warm exterior transformer / HVAC unit on building flank (anchors AGC smoothly)
            for ry in range(52, 66):
                for rx in range(147, 156):
                    dx = (rx - 151.5) / 4.0
                    dy = (ry - 59.0) / 6.0
                    bg[ry, rx] = 68.0 + 82.0 * math.exp(-0.85 * (dx * dx + dy * dy))

            # Horizontal compacted dirt/asphalt infiltration trail (y=88..108)
            for y in range(88, 108):
                dist_c = abs(y - 98.0) / 10.0
                bg[y, :] = 76.0 - dist_c * 7.0
            bg[94:96, :] -= 3.0
            bg[102:104, :] -= 3.0

            # Left Sector (x=0..65): Elevated Low Razor-Wire & Pipe Barrier above Crawl Gap!
            # Barrier occupies y=72..81, leaving a clear low crawl space at y=83..104 underneath
            bg[73:76, 0:66] = 86.0             # Upper horizontal security pipe / beam
            bg[80:82, 0:66] = 79.0             # Lower razor-wire tension cable
            for col in range(0, 66):
                coil_y = int(77 + math.sin(col * 0.85) * 2.8)
                bg[coil_y, col] = 94.0
            # Structural support stanchions for left barrier (ending at x=64 breach point!)
            for px in [6, 25, 44, 64]:
                bg[68:90, px:px + 2] = 90.0
            # Small warm infrared sensor emitter mounted on the end stanchion (x=63..67, y=71..75)
            bg[71:75, 63:67] = 132.0

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
        terrain_std = 1.4 if self.environment == "facility" else 3.2
        noise = self.rng.normal(0, terrain_std, (h, w)).astype(np.float32)
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
            # Concertina wire is already rendered cleanly on the wall crest
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
        """
        Renders a physical LWIR human infiltrator executing a 3-phase tactical breach:
          - Phase 1 (t <= 108): Low prone commando crawl under the outer barrier
          - Phase 2 (108 < t < 128): Dynamic hands-and-knee push-up & stand-up transition
          - Phase 3 (t >= 128): Full upright forward-leaning athletic sprint
        Uses smooth Gaussian capsule distance fields for authentic LWIR thermal radiation.
        """
        patch = np.zeros((patch_h, patch_w), dtype=np.float32)
        py, px = np.ogrid[:patch_h, :patch_w]
        px_f = px.astype(np.float32)
        py_f = py.astype(np.float32)

        # Posture transition parameter p in [0.0 (prone crawl) .. 1.0 (upright sprint)]
        if t <= 108:
            p = 0.0
        elif t >= 128:
            p = 1.0
        else:
            u = (t - 108.0) / 20.0
            p = u * u * (3.0 - 2.0 * u)  # Smoothstep transition
        kneel = 4.0 * p * (1.0 - p)      # Peaks at p=0.5 during one-knee push-up

        dir_s = 1.0 if is_forward else -1.0
        cx0 = patch_w * 0.50
        gy = patch_h * 0.88  # Fixed ground contact plane inside patch

        # Phase oscillators for crawl (period=18) and sprint (period=11)
        phi_c = (t / 18.0) * 2.0 * math.pi
        phi_r = ((t - 118.0) / 11.0) * 2.0 * math.pi

        def lerp2(a: Tuple[float, float], b: Tuple[float, float], w_p: float) -> Tuple[float, float]:
            return (a[0] + w_p * (b[0] - a[0]), a[1] + w_p * (b[1] - a[1]))

        def to_canvas(rel_x: float, abs_y: float) -> Tuple[float, float]:
            return (cx0 + dir_s * rel_x, abs_y)

        def add_capsule(
            p1: Tuple[float, float], p2: Tuple[float, float], radius: float, temp: float
        ) -> None:
            vx = p2[0] - p1[0]
            vy = p2[1] - p1[1]
            l2 = vx * vx + vy * vy + 1e-6
            u_proj = np.clip(((px_f - p1[0]) * vx + (py_f - p1[1]) * vy) / l2, 0.0, 1.0)
            dx = px_f - (p1[0] + u_proj * vx)
            dy = py_f - (p1[1] + u_proj * vy)
            d2 = (dx * dx + dy * dy) / (radius * radius + 1e-6)
            np.maximum(patch, np.exp(-1.45 * d2) * temp, out=patch)

        # --- 1. TORSO & HEAD KINEMATICS ---
        # Crawl core coords (relative x, absolute y)
        hip_c = (-3.4, gy - 3.6)
        sh_c = (4.6, gy - 5.2 - 0.4 * abs(math.sin(phi_c)))
        head_c = (9.6, gy - 7.3 + 0.35 * math.sin(2.0 * phi_c))

        # Sprint core coords (athletic forward lean + natural running bounce)
        bounce = 1.15 * abs(math.sin(phi_r))
        hip_r = (-1.0, gy - 12.2 - bounce)
        sh_r = (1.6, gy - 19.6 - bounce)
        head_r = (2.8, gy - 24.8 - bounce)

        hip_rel = lerp2(hip_c, hip_r, p)
        sh_rel = lerp2(sh_c, sh_r, p)
        # During push-up (kneel), chest rises slightly ahead of hips
        sh_rel = (sh_rel[0], sh_rel[1] - 1.8 * kneel)
        head_rel = lerp2(head_c, head_r, p)
        head_rel = (head_rel[0], head_rel[1] - 1.6 * kneel)

        hip = to_canvas(*hip_rel)
        shoulder = to_canvas(*sh_rel)
        head = to_canvas(*head_rel)
        neck = lerp2(shoulder, head, 0.48)

        # --- 2. LEGS (2-Segment Hip -> Knee -> Ankle -> Foot for Near & Far Legs) ---
        for leg_idx, phase_off, heat_scale in [(0, math.pi, 0.75), (1, 0.0, 0.88)]:
            # Crawl leg state (distinct bent-knee crawl drive vs extended trail leg)
            c_kick = 0.5 * (1.0 + math.cos(phi_c + phase_off))
            y_sep = -1.2 if leg_idx == 0 else 0.4
            knee_c = (-8.0 + 4.8 * c_kick, gy - 1.5 - 3.2 * c_kick + y_sep)
            ank_c = (-14.8 + 4.0 * c_kick, gy - 0.7 - 0.5 * c_kick + y_sep * 0.5)

            # Sprint leg state (full dynamic stride swing + backward knee flexion)
            th_r = phi_r + phase_off
            a_thigh = 0.72 * math.sin(th_r) + 0.14
            knee_flex = 0.48 + 0.65 * max(0.0, -math.cos(th_r - 0.35))
            a_shin = a_thigh - knee_flex
            l_thigh, l_shin = 6.3, 6.0
            kx_r = hip_r[0] + l_thigh * math.sin(a_thigh)
            ky_r = hip_r[1] + l_thigh * math.cos(a_thigh)
            ax_r = kx_r + l_shin * math.sin(a_shin)
            ay_r = min(gy, ky_r + l_shin * math.cos(a_shin))

            knee_rel = lerp2(knee_c, (kx_r, ky_r), p)
            ank_rel = lerp2(ank_c, (ax_r, ay_r), p)

            # During stand-up transition, lead leg (leg_idx==1) steps forward under hip to push up
            if leg_idx == 1:
                knee_rel = (knee_rel[0] + 4.5 * kneel, knee_rel[1] - 1.8 * kneel)
                ank_rel = (ank_rel[0] + 5.8 * kneel, min(gy, ank_rel[1] + 0.5 * kneel))

            knee = to_canvas(*knee_rel)
            ankle = to_canvas(*ank_rel)
            foot = to_canvas(ank_rel[0] + 1.7, min(gy + 0.3, ank_rel[1] + 0.3))

            add_capsule(hip, knee, 1.65, intensity * heat_scale)
            add_capsule(knee, ankle, 1.32, intensity * (heat_scale - 0.04))
            add_capsule(ankle, foot, 1.15, intensity * (heat_scale - 0.08))

        # --- 3. ARMS (2-Segment Shoulder -> Elbow -> Hand for Near & Far Arms) ---
        for arm_idx, phase_off, heat_scale in [(0, math.pi, 0.76), (1, 0.0, 0.88)]:
            # Crawl arm state (reaching out flat in front of head & pulling back)
            c_reach = 0.5 * (1.0 + math.sin(phi_c + phase_off))
            elb_c = (5.4 + 4.2 * c_reach, gy - 2.0 - 0.8 * c_reach)
            hnd_c = (9.4 + 6.8 * c_reach, gy - 0.8 - 1.0 * max(0.0, math.cos(phi_c + phase_off)))

            # Sprint arm state (vigorous counter-phase bent-elbow pumping)
            th_a = phi_r + phase_off + math.pi
            a_upper = 0.68 * math.sin(th_a) - 0.10
            a_fore = a_upper + 1.35
            ex_r = sh_r[0] + 4.4 * math.sin(a_upper)
            ey_r = sh_r[1] + 4.4 * math.cos(a_upper)
            hx_r = ex_r + 4.1 * math.sin(a_fore)
            hy_r = ey_r + 4.1 * math.cos(a_fore)

            elb_rel = lerp2(elb_c, (ex_r, ey_r), p)
            hnd_rel = lerp2(hnd_c, (hx_r, hy_r), p)
            # Hands push off ground during early stand-up
            hnd_rel = (hnd_rel[0] - 1.5 * kneel, hnd_rel[1] + 1.8 * kneel)

            elbow = to_canvas(*elb_rel)
            hand = to_canvas(*hnd_rel)

            add_capsule(shoulder, elbow, 1.28, intensity * heat_scale)
            add_capsule(elbow, hand, 1.15, intensity * (heat_scale + 0.04))

        # --- 4. CORE TORSO, DISTINCT NECK & ROUND HUMAN HEAD ---
        torso_r = 2.45 + 0.55 * p
        add_capsule(hip, shoulder, torso_r, intensity * 0.92)
        add_capsule(shoulder, neck, 1.20, intensity * 0.84)  # Narrower neck creates clear head separation
        # Distinct round Gaussian human head (matches Perimeter Wire Human Crossing realism)
        head_rad2 = ((px_f - head[0]) / (2.35 + 0.25 * p)) ** 2 + ((py_f - head[1]) / (2.25 + 0.35 * p)) ** 2
        np.maximum(patch, np.exp(-1.55 * head_rad2) * intensity, out=patch)

        return patch

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
                    span = max(1, (anom.end_frame - anom.start_frame))
                    alpha = (t - anom.start_frame) / span
                    if anom.anomaly_type in ("crawler", "crawling_intruder"):
                        # 3-Phase Velocity Profile:
                        # Phase 1 (t <= 108): Slow pulsed low-crawl under left barrier (alpha -> 0.0..0.44, clears x=64 post)
                        # Phase 2 (108 < t < 128): Push-up & stand-up in clear zone x=74..85 (alpha -> 0.44..0.51)
                        # Phase 3 (t >= 128): Rapid upright tactical sprint across courtyard (alpha -> 0.51..1.0)
                        if t <= 108:
                            u_c = max(0.0, (t - anom.start_frame) / max(1.0, 108.0 - anom.start_frame))
                            surge = (0.35 / (2.0 * math.pi * 8.0)) * math.sin(2.0 * math.pi * 8.0 * u_c)
                            alpha = 0.44 * np.clip(u_c - surge, 0.0, 1.0)
                        elif t < 128:
                            u_t = (t - 108.0) / 20.0
                            alpha = 0.44 + 0.07 * (0.35 * u_t + 0.65 * (u_t * u_t))
                        else:
                            u_r = min(1.0, (t - 128.0) / max(1.0, anom.end_frame - 128.0))
                            alpha = 0.51 + 0.49 * u_r
                        alpha = float(np.clip(alpha, 0.0, 1.0))
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
                fpn_scale = 0.55 if self.environment == "facility" else 1.0
                netd_std = 2.2 if self.environment == "facility" else 3.8
                frame += fpn_cols[None, :] * fpn_scale
                netd_noise = self.rng.normal(0, netd_std, (h, w)).astype(np.float32)
                frame += netd_noise

            raw_frames[t] = np.clip(frame, 0, 255).astype(np.uint8)
            labels[t] = is_anomaly
            masks[t] = frame_mask

        return raw_frames, labels, masks
