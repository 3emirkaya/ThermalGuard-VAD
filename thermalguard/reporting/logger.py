"""
ThermalGuard Olay Kaydı (Event Logger) — Zaman Damgalı Anomali Kayıt Sistemi.

Başvuru kapsamı gereksinimleri:
  ✅ Anomali olaylarının zaman damgasıyla kayıt altına alınması
  ✅ Yanlış alarm önleme (debounce / persistence filtresi)
  ✅ JSON ve CSV çıktı formatları
  ✅ Kullanıcıya anlık bildirim için callback desteği

Kayıt Mantığı (Debounce):
--------------------------
Tek başına yüksek gelen kare kayıt edilmez; anomali ``persist_frames``
ardışık kare boyunca devam ederse gerçek "olay" olarak kabul edilir.
Bu, tek karelik sensör gürültüsü / yanlış alarmları bastırır.
"""
from __future__ import annotations

import csv
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional


# --------------------------------------------------------------------------- #
#  Veri Yapıları
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyEvent:
    """Tek bir anomali olayının tüm meta verisi."""

    event_id: int
    start_frame: int
    end_frame: int
    start_time: str            # ISO 8601 zaman damgası
    end_time: Optional[str]
    duration_frames: int
    anomaly_type: str          # 'unknown' | simülatör tipi
    peak_score: float
    mean_score: float
    scores: List[float]        # Olay boyunca kare bazlı skor listesi
    llm_summary: Optional[str] = None   # LLM tarafından üretilen metin özeti

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_brief_dict(self) -> Dict:
        """Web arayüzü / API için hafif sözlük (scores listesi hariç)."""
        d = self.to_dict()
        d.pop("scores", None)
        return d


# --------------------------------------------------------------------------- #
#  Olay Kaydı
# --------------------------------------------------------------------------- #
class EventLogger:
    """
    Kare bazlı anomali sinyallerini debounce ederek anlamlı olaylar üretir,
    bunları JSON ve CSV dosyalarına yazar, callback'lerle dış sistemlere bildirir.

    Örnek kullanım::

        logger = EventLogger(log_dir="logs/", persist_frames=3)
        logger.on_event(lambda ev: print(f"ALARM: {ev.llm_summary}"))

        for res in engine:
            logger.update(res.frame_idx, res.is_anomaly, res.anomaly_score)
        logger.finalize()
    """

    def __init__(
        self,
        log_dir: str = "logs",
        session_name: Optional[str] = None,
        persist_frames: int = 3,
        max_gap_frames: int = 15,
        on_event_callback: Optional[Callable[[AnomalyEvent], None]] = None,
    ):
        """
        Parameters
        ----------
        log_dir         : Kayıt klasörü (otomatik oluşturulur).
        session_name    : JSON/CSV dosya adı öneki. None → timestamp.
        persist_frames  : Olay başlatmak için gereken ardışık anomali kare sayısı.
        max_gap_frames  : Normal kare aralığı bu değeri geçerse olay kapanır.
        on_event_callback : Olay kapandığında çağrılan fonksiyon (AnomalyEvent alır).
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.persist_frames = persist_frames
        self.max_gap_frames = max_gap_frames
        self._callback = on_event_callback

        ts = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._json_path = self.log_dir / f"{ts}_events.json"
        self._csv_path = self.log_dir / f"{ts}_events.csv"
        self._all_path = self.log_dir / f"{ts}_all_frames.csv"

        # Durum değişkenleri
        self._event_id = 0
        self._events: List[AnomalyEvent] = []
        self._lock = threading.Lock()

        # Devam eden olay takibi
        self._in_event: bool = False
        self._pending_count: int = 0      # Olay başlatma için sayaç
        self._gap_count: int = 0          # Kapanma için normal kare sayacı
        self._current_event: Optional[Dict] = None

        # Tüm kare logu (frame_idx, score, is_anomaly)
        self._all_frames: List[tuple] = []

        # CSV başlıklarını oluştur
        self._init_csv()

    # ------------------------------------------------------------------ #
    def _init_csv(self) -> None:
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "event_id", "start_frame", "end_frame",
                "start_time", "end_time", "duration_frames",
                "anomaly_type", "peak_score", "mean_score", "llm_summary",
            ])
            writer.writeheader()

        with open(self._all_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["frame_idx", "anomaly_score", "is_anomaly"])

    # ------------------------------------------------------------------ #
    def update(
        self,
        frame_idx: int,
        is_anomaly: bool,
        anomaly_score: float,
        anomaly_type: str = "unknown",
    ) -> Optional[AnomalyEvent]:
        """
        Her kare için çağrılır. Olay kapandığında ``AnomalyEvent`` döner, aksi halde None.
        """
        with self._lock:
            # Tüm kare logu
            self._all_frames.append((frame_idx, anomaly_score, int(is_anomaly)))
            with open(self._all_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([frame_idx, f"{anomaly_score:.6f}", int(is_anomaly)])

            closed_event: Optional[AnomalyEvent] = None

            if is_anomaly:
                self._gap_count = 0
                if not self._in_event:
                    self._pending_count += 1
                    if self._pending_count >= self.persist_frames:
                        # Olay başlat
                        self._in_event = True
                        start_f = frame_idx - self._pending_count + 1
                        self._current_event = {
                            "start_frame": start_f,
                            "start_time": datetime.now(timezone.utc).isoformat(),
                            "anomaly_type": anomaly_type,
                            "scores": [],
                        }
                else:
                    # Devam eden olay
                    pass

                if self._in_event and self._current_event:
                    self._current_event["scores"].append(anomaly_score)

            else:
                if self._in_event:
                    self._gap_count += 1
                    if self._current_event:
                        self._current_event["scores"].append(anomaly_score)
                    if self._gap_count >= self.max_gap_frames:
                        closed_event = self._close_event(frame_idx, anomaly_type)
                else:
                    self._pending_count = 0

            return closed_event

    # ------------------------------------------------------------------ #
    def _close_event(self, frame_idx: int, anomaly_type: str) -> AnomalyEvent:
        """Mevcut olayı kapat ve kaydet."""
        ce = self._current_event
        scores = ce.get("scores", [])
        end_time = datetime.now(timezone.utc).isoformat()

        event = AnomalyEvent(
            event_id=self._event_id,
            start_frame=ce["start_frame"],
            end_frame=frame_idx,
            start_time=ce["start_time"],
            end_time=end_time,
            duration_frames=frame_idx - ce["start_frame"] + 1,
            anomaly_type=anomaly_type,
            peak_score=float(max(scores)) if scores else 0.0,
            mean_score=float(sum(scores) / len(scores)) if scores else 0.0,
            scores=scores,
        )

        self._event_id += 1
        self._in_event = False
        self._pending_count = 0
        self._gap_count = 0
        self._current_event = None
        self._events.append(event)

        # Dosyalara yaz
        self._append_event_to_files(event)

        # Callback
        if self._callback:
            threading.Thread(target=self._callback, args=(event,), daemon=True).start()

        return event

    # ------------------------------------------------------------------ #
    def _append_event_to_files(self, event: AnomalyEvent) -> None:
        # JSON: tüm olay listesini yeniden yaz (küçük dosya → no-problem)
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._events], f, ensure_ascii=False, indent=2)

        # CSV: sadece satır ekle
        brief = event.to_brief_dict()
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "event_id", "start_frame", "end_frame",
                "start_time", "end_time", "duration_frames",
                "anomaly_type", "peak_score", "mean_score", "llm_summary",
            ])
            writer.writerow({k: brief.get(k, "") for k in writer.fieldnames})

    # ------------------------------------------------------------------ #
    def finalize(self) -> List[AnomalyEvent]:
        """
        Video bitiminde açık kalan olayı kapat ve son JSON/CSV dosyalarını yaz.
        Tüm olayların listesini döndürür.
        """
        with self._lock:
            if self._in_event and self._current_event:
                ce = self._current_event
                scores = ce.get("scores", [])
                event = AnomalyEvent(
                    event_id=self._event_id,
                    start_frame=ce["start_frame"],
                    end_frame=ce["start_frame"] + len(scores),
                    start_time=ce["start_time"],
                    end_time=datetime.now(timezone.utc).isoformat(),
                    duration_frames=len(scores),
                    anomaly_type=ce.get("anomaly_type", "unknown"),
                    peak_score=float(max(scores)) if scores else 0.0,
                    mean_score=float(sum(scores) / len(scores)) if scores else 0.0,
                    scores=scores,
                )
                self._events.append(event)
                self._event_id += 1
                self._in_event = False
                self._current_event = None
                self._append_event_to_files(event)

        # Son JSON yaz
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._events], f, ensure_ascii=False, indent=2)

        return list(self._events)

    # ------------------------------------------------------------------ #
    @property
    def events(self) -> List[AnomalyEvent]:
        return list(self._events)

    @property
    def json_path(self) -> Path:
        return self._json_path

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def set_event_llm_summary(self, event_id: int, summary: str) -> None:
        """LLM özeti üretildikten sonra ilgili olaya eklenir."""
        with self._lock:
            for ev in self._events:
                if ev.event_id == event_id:
                    ev.llm_summary = summary
                    break
            # JSON'u güncelle
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self._events], f, ensure_ascii=False, indent=2)
