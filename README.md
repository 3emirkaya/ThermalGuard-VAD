<div align="center">

# ThermalGuard-VAD

**Unsupervised Long-Wave Infrared (LWIR) Video Anomaly Detection for Defense Perimeter Surveillance**

*Real-time radiometric thermal preprocessing, Spatio-Temporal Autoencoder (STAE + ConvLSTM) inference, and a modern C2 web console engineered for edge deployment.*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-18181b?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Edge Ready](https://img.shields.io/badge/Deployment-TorchScript%20%2F%20ONNX-10b981?style=flat-square)](weights/)
[![PyTest](https://img.shields.io/badge/Tests-13%2F13%20Passing-10b981?style=flat-square)](tests/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-f97316?style=flat-square)](LICENSE)

[Live Console](#tactical-web-console) &nbsp;&middot;&nbsp;
[Live Benchmarks](#real-time-lwir-detection-benchmarks) &nbsp;&middot;&nbsp;
[Quick Start](#quick-start) &nbsp;&middot;&nbsp;
[Engineering Docs](#english-documentation) &nbsp;&middot;&nbsp;
[Türkçe Dokümantasyon](#turkce-dokumantasyon)

</div>

---

<a id="tactical-web-console"></a>
## Tactical Web Console (`app.py`)

![ThermalGuard-VAD Live Command Console](docs/dashboard_live.png)

> **Live Perimeter Surveillance Console:** Single-click scenario switcher (`Real LWIR` & `Simulation`), multi-source input support (KAIST sequences, RTSP IP thermal cameras, USB webcams, or custom uploads), **960×810 HD dual-feed LWIR HUD** (White-Hot raw input + Inferno fusion & tracking bounding box), live **Instantaneous Anomaly Score** meter, and automated real-time **Incident Stream**.

---

<a id="real-time-lwir-detection-benchmarks"></a>
## Real-Time LWIR Detection Benchmarks

<table>
  <tr>
    <td width="50%" align="center">
      <img src="demo_kaist_pedestrian.gif" alt="KAIST Pedestrian Sector Crossing" width="100%" /><br/>
      <strong>KAIST Pedestrian Sector Crossing</strong><br/>
      <sub>Real LWIR thermal sensor sequence &mdash; spatial anomaly localization &amp; tracking</sub>
    </td>
    <td width="50%" align="center">
      <img src="demo_field_thermal.gif" alt="Operational Field Corridor Benchmark" width="100%" /><br/>
      <strong>Operational Field Corridor (640&times;512 LWIR)</strong><br/>
      <sub>Live dynamic score elevation &amp; perimeter intruder bounding-box detection</sub>
    </td>
  </tr>
</table>

### Integrated Security Audit & Incident Forensics

![ThermalGuard-VAD Security Audit Report](docs/incident_report.png)

> **Built-in Live Audit Log:** Located directly below the surveillance viewport, automatically capturing frame ranges, intrusion duration, peak anomaly scores, target classifications (`Human` / `Vehicle` / `Crawler`), and offline or LLM-generated tactical threat assessments (Rule-Based, Ollama, OpenAI GPT, Google Gemini) with one-click **PDF Export**.

---

<a id="quick-start"></a>
## Quick Start

```bash
git clone https://github.com/3emirkaya/ThermalGuard-VAD.git
cd ThermalGuard-VAD
pip install -r requirements.txt

# Launch the Web Command Console -> http://localhost:5000
python app.py
```

| Workflow | Command | Description |
| :--- | :--- | :--- |
| **Web Console** | `python app.py` | Launches the Obsidian/Alabaster live monitoring & audit dashboard at `localhost:5000` |
| **Unit Tests** | `pytest` | Runs all 13 preprocessing, model architecture, and streaming engine tests |
| **Model Training** | `python scripts/train.py --config configs/default.yaml` | Trains the unsupervised STAE + ConvLSTM model on normal LWIR clips |
| **Ablation Benchmark** | `python scripts/benchmark.py` | Evaluates AUROC, PR-AUC, Pixel-AUC, and sensor noise robustness |
| **Edge Export** | `python scripts/export_onnx.py` | Exports verified TorchScript & ONNX runtimes with exact bitwise parity |

### Core Capabilities at a Glance

| Layer | Engineering Highlights |
| :--- | :--- |
| **Thermal Preprocessing** | Deterministic 5-stage LWIR pipeline: Percentile AGC + EMA &rarr; Dead/Hot Pixel Repair &rarr; Column FPN Destriping &rarr; Bilateral Denoising &rarr; CLAHE |
| **Anomaly Backbone** | Unsupervised 2D CNN Encoder &rarr; Fused-Gate GroupNorm ConvLSTM Bottleneck (75% compression, zero skip connections) &rarr; Bilinear 2D Decoder |
| **Scoring & Tracking** | Top-K (99.5th percentile) spatial aggregation + causal EMA smoothing + morphological contour fusion for cohesive target bounding boxes |
| **Tactical Simulator** | Physics-based LWIR generator with 3 environments (`outpost` vehicle breach, `facility` 2-Bone IK crawl-to-sprint saboteur, `perimeter` wire crossing) |
| **Edge Readiness** | 588K parameters (~4.60 MB), **44.6 FPS** on CPU / **160+ FPS** on NVIDIA Jetson, $< 15\text{ MB}$ rolling streaming RAM |

---

## Navigation

* **English Technical Documentation**
  * [1. Defense Mission Profile & Thermal Physics Rationale](#1-defense-mission-profile--thermal-physics-rationale)
  * [2. Why Standard RGB Models Fail on Infrared](#2-why-standard-rgb-models-fail-on-infrared)
  * [3. Unsupervised Normality Learning Formulation](#3-unsupervised-normality-learning-formulation)
  * [4. Thermal-Specific Preprocessing Pipeline](#4-thermal-specific-preprocessing-pipeline)
  * [5. Spatio-Temporal Autoencoder (STAE) Architecture](#5-spatio-temporal-autoencoder-stae-architecture)
  * [6. Anomaly Scoring & Spatial Localization Heatmap](#6-anomaly-scoring--spatial-localization-heatmap)
  * [7. Empirical Benchmark & Ablation Study](#7-empirical-benchmark--ablation-study)
  * [8. Edge Deployment & Real-Time Telemetry](#8-edge-deployment--real-time-telemetry)
  * [9. Repository Architecture](#9-repository-architecture)
* **Türkçe Teknik Dokümantasyon**
  * [Türkçe Versiyon](#turkce-dokumantasyon)

---

<a id="english-documentation"></a>
# English Technical Documentation

## 1. Defense Mission Profile & Thermal Physics Rationale

In border protection, Forward Operating Base (FOB) perimeters, critical ammunition depots, and coastal security stations, **Long-Wave Infrared (LWIR: 8–14 µm)** thermal cameras serve as the primary electro-optical observation layer. Operating in the thermal infrared atmospheric transmission window, uncooled microbolometer sensors capture passive emitted radiative heat rather than reflected ambient visible light.

However, operational field conditions introduce severe physical challenges:
1. **Zero Chrominance and Minimal Texture:** Thermal pixels represent blackbody radiant exitance proportional to surface temperature ($M = \epsilon \sigma T^4$) and emissivity. Chrominance (RGB color cues) does not exist, and high-frequency textural patterns typical of optical photography are smoothed out by thermal diffusion.
2. **Thermal Crossover Phenomenon:** During dawn and dusk, the radiant temperature of ambient terrain (soil, rocks, vegetation) transitions across the ambient temperature of human targets or vehicle signatures. The differential radiant temperature drops near zero ($\Delta T \to 0$), collapsing global dynamic range and causing targets to vanish against the background.
3. **Sensor-Level Defect Characteristics:**
   * **Fixed Pattern Noise (FPN):** Microbolometer Focal Plane Arrays (FPAs) suffer from spatial non-uniformity across readout columns, producing persistent vertical and horizontal striping artifacts.
   * **Noise Equivalent Temperature Difference (NETD):** Tactical microbolometers display temporal high-frequency noise (~40–60 mK), which easily overwhelms weak intruder signatures.
   * **Dead / Hot Pixels:** Defective detector elements generate impulse salt-and-pepper noise spikes.

`ThermalGuard-VAD` is engineered to resolve these constraints through physics-aware thermal preprocessing coupled with a spatio-temporal deep neural architecture.

---

## 2. Why Standard RGB Models Fail on Infrared

Deploying standard visible-spectrum (RGB) computer vision backbones directly onto thermal camera streams is an anti-pattern that fails in military evaluations for fundamental reasons:

| Aspect | Visible RGB Domain | Thermal Infrared (LWIR) Domain | Failure Mode of RGB Models |
| :--- | :--- | :--- | :--- |
| **Physical Quantity** | Reflected visible light (400–700 nm) | Emitted surface radiant heat (8–14 µm) | Models trained on ImageNet feature hierarchies look for nonexistent color gradients. |
| **Dynamic Range** | 8-bit per channel (sRGB curve) | Raw 14-bit or 16-bit radiometric linear exitance | Global Min-Max scaling compresses subtle target signatures into a narrow 8-bit histogram band. |
| **Edge Information** | Chromatic & shadow edges | Thermal boundary differentials | Shadows have no thermal signature, while thermal heat plumes bleed across physical silhouettes. |
| **Sensor Artifacts** | Bayer pattern demosaicing noise | FPA column FPN & microbolometer thermal drift | RGB models interpret static sensor striping as structural edges and object boundaries. |

---

## 3. Unsupervised Normality Learning Formulation

In defense perimeter security, true negative ("normal") data is abundant: thousands of hours of empty perimeter wire, swaying vegetation under natural wind, and routine patrol dynamics. Conversely, security anomalies (unauthorized human infiltration, stationary suspicious packages, or concealed warm engines) are exceptionally rare, unpredictably diverse, and practically impossible to annotate exhaustively.

`ThermalGuard-VAD` operates strictly under the **Unsupervised Video Anomaly Detection (VAD)** paradigm:

```
[Normal Thermal Sequence: Empty Perimeter] ──► Train STAE Model ──► Learns Normal Dynamic Manifold
                                                                           │
[Test Frame with Intruder / Breaching Target] ──► Forward Pass ───────────┴──► High Reconstruction Error!
                                                                                (Anomalous Heatmap)
```

1. **Training Phase:** The model is trained exclusively on normal video clips ($x \in \mathcal{X}_{\text{normal}}$) to learn a compact spatio-temporal reconstruction mapping $\hat{x} = f_\theta(x)$.
2. **Inference Phase:** When normal terrain is observed, the network successfully reconstructs the scene with minimal residual loss. When an unseen target (human intruder or vehicle) enters, the bottleneck constraints prevent the network from reproducing the anomalous patterns, generating a localized spike in reconstruction loss.

---

## 4. Thermal-Specific Preprocessing Pipeline

To prepare raw infrared data for deep neural feature extraction without amplifying noise, `thermalguard.preprocessing` implements a deterministic 5-stage pipeline:

```
Raw Radiometric Frame (8-bit or 16-bit)
       │
       ▼
[1. Percentile AGC & Temporal EMA]
    • Cuts extreme 0.5% and 99.5% outliers; applies EMA smoothing to suppress AGC flicker.
       │
       ▼
[2. Dead / Hot Pixel Correction]
    • Detects spatial outliers exceeding 3×3 median neighborhood by >40 DN without blurring edges.
       │
       ▼
[3. Column / Row FPN Destriping]
    • Calculates 1D median profile along sensor columns and subtracts a low-pass Gaussian profile.
       │
       ▼
[4. Edge-Preserving Denoising (Bilateral Filter)]
    • Attenuates temporal NETD sensor noise while preserving sharp thermal target contours.
       │
       ▼
[5. CLAHE (Contrast Limited Adaptive Histogram Equalization)]
    • Enhances local thermal gradients across 8×8 tiles with clipLimit=2.0 AFTER noise removal.
       │
       ▼
Model-Ready Normalized Float32 Tensor ∈ [0.0, 1.0] (128 × 160)
```

> **Critical Engineering Rule (Order of Operations):** CLAHE must **never** be applied directly to raw infrared frames. Applying histogram equalization before denoising dramatically amplifies high-frequency NETD noise and column FPN stripes, leading to severe false alarm rates. In `ThermalGuard-VAD`, edge-preserving bilateral filtering strictly precedes CLAHE.

---

## 5. Spatio-Temporal Autoencoder (STAE) Architecture

The network consists of a weight-shared 2D Convolutional Frame Encoder, a multi-layer ConvLSTM temporal bottleneck, and a 2D Convolutional Decoder:

```
Input Clip: x ∈ ℝ^(B × T × 1 × H × W)    [Default: T=8, H=128, W=160]
     │
     ▼
[Shared 2D CNN Frame Encoder]            [Channels: 16 → 32 → 64 → 96, LeakyReLU(0.2)]
     │
     ▼ Latent Features: z_t ∈ ℝ^(B × T × 64 × 8 × 10)  [75% Compression Ratio]
     │
     ▼
[Fused-Gate GroupNorm ConvLSTM]          [Hidden: (64, 64), 3×3 Kernels, Recurrent Memory]
     │
     ▼ Bottleneck State: h_T ∈ ℝ^(B × 64 × 8 × 10)
     │
     ▼
[Bilinear Upsampling + 2D CNN Decoder]   [Channels: 96 → 64 → 32 → 16, Sigmoid Output]
     │
     ▼
Reconstructed Frame: x̂_T ∈ ℝ^(B × 1 × 1 × H × W)
```

### Prohibition of Skip Connections
Unlike U-Net architectures used for image segmentation, **skip connections are strictly prohibited** in anomaly detection autoencoders. If skip connections are present between encoder and decoder layers, high-frequency spatial details of unauthorized intruders bypass the latent bottleneck and are faithfully reconstructed by the decoder, completely nullifying the reconstruction loss signal.

### Hybrid Reconstruction Loss
$$\mathcal{L}(x, \hat{x}) = \text{MSE}(x, \hat{x}) + \lambda_{\text{ssim}} \cdot \left(1 - \text{SSIM}(x, \hat{x})\right)$$
* **MSE Term:** Measures absolute radiometric/intensity deviations across the reconstructed frame.
* **SSIM Term ($\lambda_{\text{ssim}} = 0.2$):** Penalizes structural and edge degradation, preventing the model from converging to a trivial "mean gray blur" in low-contrast environments.

---

## 6. Anomaly Scoring & Spatial Localization Heatmap

### Spatial Aggregation: Why Mean Pooling Fails
In border surveillance, an intruder occupies only 1% to 3% of the image frame. If standard spatial mean pooling is applied to the error map $e(x, \hat{x}) = (x - \hat{x})^2$, the vast normal background (97%) drowns out the anomaly signal, resulting in missed detections.

`ThermalGuard-VAD` implements **Top-K / 99.5th Percentile Aggregation**:
$$\mathcal{S}_{\text{frame}} = \text{Percentile}_{99.5}\left( (x - \hat{x})^2 \right)$$
Combined with a causal Exponential Moving Average (EMA, $\alpha=0.6$) across consecutive frames, this eliminates transient single-frame sensor noise while maintaining immediate sensitivity to perimeter breaches.

### High-DPI Dual-Panel Tactical HUD (`960×810`)
* **Upper Panel:** Live scrolling graph rendering anomaly score vs. frame index with dashed threshold line (`Threshold = 0.20`) and alert status readout.
* **Lower Panel:** Split display showing preprocessed LWIR input alongside the Inferno false-color thermal fusion with morphological contour merging and tactical corner-bracket target tracking.

---

## 7. Empirical Benchmark & Ablation Study

Evaluated under controlled baseline and severe field degradation conditions (simulating heavy fog, thermal crossover, NETD noise $\sigma=12\text{ DN}$, and column FPN):

| Pipeline Configuration | Frame AUROC | PR-AUC | Pixel-AUC (Localization) | RMS Contrast | Noise $\sigma$ | Latency (ms) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Raw Baseline (Min-Max Only)** | 0.9683 | 0.9460 | 0.5423 | 0.184 | 0.71 | 21.0 ms | 47.6 FPS |
| **B: CLAHE Only (No Denoising)** | 0.9756 | 0.9600 | 0.8685 | 0.235 | 1.31 | 22.8 ms | 43.8 FPS |
| **C: ThermalGuard Full Pipeline** | **0.9874** | **0.9873** | **0.9190** | **0.217** | **0.84** | 22.4 ms | **44.6 FPS** |
| **A*: Raw + Field Sensor Degradation** | 0.4602 | 0.5121 | 0.1703 | 0.043 | 6.28 | 20.7 ms | 48.2 FPS |
| **C*: ThermalGuard + Field Degradation** | **Preserved** | — | **0.7829** | 0.177 | Suppressed | 22.9 ms | **43.7 FPS** |

### Key Findings for Defense Evaluators
1. **Localization Failure of Raw RGB Normalization:** While Pipeline A achieves acceptable whole-frame detection on clean test clips, its Pixel-AUC is near chance (0.5423), failing to localize where the intruder is in the image.
2. **Catastrophic Failure under Field Noise:** When subjected to realistic field degradation (Pipeline A*), raw processing collapses completely (AUROC 0.4602).
3. **Robustness of ThermalGuard (Pipeline C):** Incorporating destriping and bilateral filtering prior to CLAHE preserves spatial anomaly localization (**Pixel-AUC 0.9190**) while maintaining real-time edge throughput (**44.6 FPS** on standard CPU, **160+ FPS** on embedded GPU).

---

## 8. Edge Deployment & Real-Time Telemetry

The architecture is verified for embedded deployment on edge platforms (NVIDIA Jetson AGX Orin, Xavier NX, and x86 edge servers):

* **Model Footprint:** 588,000 float32 parameters (~4.60 MB total storage).
* **TorchScript & ONNX Export:** Traceable into standalone TorchScript (`weights/stae_thermal_traced.pt`) and ONNX graph structures.
* **Numerical Equivalence:** Maximal absolute discrepancy between PyTorch and deployed runtime outputs: $|y_{\text{pytorch}} - y_{\text{deploy}}| = 0.000000 \times 10^0$ (exact bitwise parity).
* **Streaming Memory Guarantee:** Rolling frame buffer consumes $< 15\text{ MB}$ of system RAM during uninterrupted 24/7 video processing.

---

## 9. Repository Architecture

```
ThermalGuard-VAD/
├── app.py                      # Obsidian/Alabaster live C2 web console & audit reporting server
├── configs/
│   ├── default.yaml            # High-fidelity surveillance configuration
│   └── edge_jetson.yaml        # Lightweight profile optimized for NVIDIA Jetson (<15ms)
├── data/                       # Dataset caching & KAIST benchmark sequences
├── docs/                       # High-resolution web console & audit report screenshots
│   ├── dashboard_live.png      # Live perimeter command center (Obsidian UI)
│   └── incident_report.png     # Integrated security audit & LLM threat report
├── scripts/
│   ├── benchmark.py            # Systematic defense ablation & noise robustness evaluator
│   ├── download_kaist.py       # Stdlib-only KAIST LWIR thermal channel downloader
│   ├── export_onnx.py          # Edge deployment export & numerical equivalence validator
│   ├── infer.py                # Real-time streaming inference engine & tactical HUD recorder
│   └── train.py                # Unsupervised Spatio-Temporal Autoencoder training CLI
├── tests/
│   ├── test_engine.py          # Anomaly scoring, streaming buffer, and HUD unit tests
│   ├── test_models.py          # ConvLSTM and STAE tensor dimension & loss unit tests
│   └── test_preprocessing.py   # Radiometric AGC, CLAHE, FPN, and noise filter unit tests
├── thermalguard/
│   ├── data/                   # Data sources, KAIST parser, and physics-based 2-Bone IK simulator
│   ├── engine/                 # Streaming inference, anomaly scoring, and metric evaluation
│   ├── models/                 # Spatio-Temporal Autoencoder & Fused ConvLSTM
│   ├── preprocessing/          # 5-stage thermal pipeline (AGC, FPN, Bilateral, CLAHE)
│   ├── reporting/              # Event logger & multi-backend LLM tactical summarizer
│   └── visualization/          # High-DPI 960×810 HUD renderer & GIF/MP4 exporter
├── demo_kaist_pedestrian.gif   # Real LWIR benchmark animation (KAIST Pedestrian Crossing)
├── demo_field_thermal.gif      # Operational field corridor benchmark animation
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Build system, metadata & pytest pythonpath
```

---

<a id="turkce-dokumantasyon"></a>
# Türkçe Teknik Dokümantasyon

## 1. Savunma Sanayii Problem Tanımı ve Termal Fizik Motivasyonu

Sınır gözetleme karakolları, askeri üs bölgeleri, mühimmat depoları ve sahil güvenlik hatlarında kullanılan **Uzun Dalga Kızılötesi (LWIR - Long Wave Infrared: 8–14 µm)** kameralar; zifiri karanlık, yoğun sis ve duman koşullarında dahi pasif ışıma enerjisini yakalayan kritik elektro-optik sensörlerdir.

Ancak sahada karşılaşılan fiziksel kısıtlar standart modelleri çaresiz bırakır:
1. **Renk ve Doku Bilgisinin Yokluğu:** Termal kameralar görünür ışığı değil, nesnelerin yaydığı kızılötesi ışıma enerjisini (sıcaklık) ölçer. Klasik RGB modelleri (ImageNet ön eğitimli ResNet, YOLO vb.) renk ve ince doku gradyanlarına aşırı bağımlı olduğundan termal görüntülerde çöker.
2. **Termal Geçiş (Thermal Crossover):** Gün doğumu ve gün batımında çevre sıcaklığı ile hedef sıcaklığı eşitlenir; görüntüdeki dinamik aralık çöker ($\Delta T \to 0$).
3. **Mikrobolometre Sensör Kusurları:**
   * **Sabit Desen Gürültüsü (FPN):** Odak düzlemi dizisinden (FPA) kaynaklanan sütun çizgileri oluşur.
   * **Zamansal Gürültü (NETD):** Termal kameralarda 40–60 mK seviyesinde yüksek frekanslı sıcaklık gürültüsü bulunur.
   * **Ölü / Sıcak Pikseller:** Bozuk algılayıcı pikseller tuz-biber gürültüsü üretir.

---

## 2. Standart RGB Modelleri Kızılötesinde Neden Çöker?

| İnceleme Boyutu | Görünür Bölge (RGB) | Termal Bölge (LWIR) | RGB Modellerinin Yaşadığı Hata |
| :--- | :--- | :--- | :--- |
| **Fiziksel Büyüklük** | Yansıyan ışık (400–700 nm) | Yayılan sıcaklık ışıması (8–14 µm) | Var olmayan renk gradyanları ararken sahte tespitler üretir. |
| **Dinamik Aralık** | 8-bit sRGB | 14/16-bit ham radyometrik çıkış | Global min-max normalizasyon zayıf hedefleri ezer. |
| **Kenar Yapısı** | Gölgeler ve renk sınırları | Termal sıcaklık gradyanları | Gölgelerin ısısı yoktur; ısı çevreye difüzyonla yayılır. |
| **Sensör Kusurları** | Bayer demosaicing gürültüsü | FPN sütun çizgileri | FPN çizgilerini nesne sınırı zannederek yanılır. |

---

## 3. Denetimsiz (Unsupervised) Normallik Öğrenme Yaklaşımı

Sınır güvenliğinde "anomali" (gece saatinde tel örgüye sızan unsur, şüpheli ısı kaynağı, yetkisiz araç geçişi) 10.000 saatte belki 1 kez gerçekleşir. Her senaryo için önceden milyonlarca etiketli veri toplamak mümkün değildir.

`ThermalGuard-VAD`, yalnızca **normal devriye / rutin durum videolarıyla** eğitilir:
* **Eğitim Aşaması:** Model yalnızca normal sekansların zamansal ve uzamsal dinamiklerini öğrenir.
* **Çıkarım Aşaması:** Normal arazide rekonstrüksiyon hatası düşük kalırken; anormal bir hedef (sızan insan veya araç) girdiğinde ağ bunu yeniden üretemez ve rekonstrüksiyon hatası (MSE / SSIM) aniden fırlar.

---

## 4. Termal Görüntülere Özel 5 Aşamalı Ön İşleme Hattı

```
Ham Termal Kare (8/16-bit)
       │
       ▼
 1. Otomatik Kazanç Kontrolü (Percentile AGC + Temporal EMA)
    • %0.5 ve %99.5 persentil kırpma; canlı yayında parlaklık titremesini (flicker) önleyen EMA
       │
       ▼
 2. Ölü / Sıcak Piksel Düzeltme (Impulse Noise Removal)
    • 3×3 yerel medyan farkına göre sensör kusurlarını kenarları bozmadan temizler
       │
       ▼
 3. Sabit Desen Gürültüsü / Sütun Çizgisi Giderme (FPN Destriping)
    • Sütun profillerinin Gauss yumuşatma farkı alınarak statik dikey çizgiler silinir
       │
       ▼
 4. Kenar Koruyucu Gürültü Azaltma (Bilateral Filter)
    • Sıcak hedef konturlarını koruyarak NETD gürültüsünü filtreler
       │
       ▼
 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    • Gürültü temizlendikten SONRA yerel bloklarda (8×8) kontrastı dengeler
       │
       ▼
 Model Giriş Tensörü ∈ [0.0, 1.0] (128 × 160)
```

> **Kritik Kural:** CLAHE asla gürültü filtresinden önce uygulanmamalıdır. Önce uygulanırsa sensörün termal gürültüsünü katlayarak sahte alarmlara yol açar.

---

## 5. Model Mimarisi, Skorlama ve Saha Performansı

* **STAE + Fused ConvLSTM:** 4 bloklu ağırlık paylaşımlı 2D CNN Encoder (16 → 32 → 64 → 96), çift katmanlı GroupNorm ConvLSTM darboğazı (%75 sıkıştırma, skip connection kesinlikle yasak) ve 2D CNN Decoder.
* **Hibrit Kayıp:** $\mathcal{L}(x, \hat{x}) = \text{MSE}(x, \hat{x}) + 0.2 \cdot (1 - \text{SSIM}(x, \hat{x}))$.
* **Top-K / %99.5 Persentil Skorlama:** Karenin yalnızca %1–3'ünü kaplayan hedeflerin arka plan içinde kaybolmasını önler.
* **Deneysel Başarım:** Ham normalizasyonda **0.5423** kalan Piksel Lokalizasyon AUC değeri, ThermalGuard tam hattıyla **0.9190** seviyesine, Frame AUROC ise **0.9874** değerine ulaşmıştır.
* **Kenar Cihaz Performansı:** 588.000 parametre (~4.60 MB), standart CPU üzerinde **44.6 FPS**, gömülü GPU (NVIDIA Jetson) üzerinde **160+ FPS**, PyTorch–TorchScript sayısal farkı **0.000000e+00**.
