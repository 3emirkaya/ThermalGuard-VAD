"""
ThermalGuard LLM Uyarı Özeti Üretici.

Başvuru kapsamı:
  "Uyarı metinlerinin bir dil modeliyle okunabilir özetlere dönüştürülmesi"

Bu modül, tespit edilen anomali olaylarının ham verilerini (kare numarası,
skor, süre, tür) alarak doğal dil açıklamalarına dönüştürür.

Desteklenen Arka Uçlar (API):
-------------------------------
1. ``ollama``  : Lokal Ollama sunucusu (Llama-3, Mistral, Gemma, vb.)
               → İnternet gerekmez, gizlilik güçlü, ücretsiz
2. ``openai``  : OpenAI API (GPT-4o, GPT-4o-mini)
               → OPENAI_API_KEY çevre değişkeni gerekir
3. ``google``  : Google Gemini API (gemini-1.5-flash)
               → GOOGLE_API_KEY çevre değişkeni gerekir
4. ``rule``    : Kural tabanlı şablon (hiçbir API gerekmez, her zaman çalışır)
               → Offline / fallback modu

Kural tabanlı mod:
------------------
``backend="rule"`` seçildiğinde API çağrısı yapılmaz; anomali türü, skoru
ve süresi üzerinden önceden tanımlı Türkçe şablonlar kullanılır.
"""
from __future__ import annotations

import json
import os
import textwrap
from typing import Optional

from .logger import AnomalyEvent


# --------------------------------------------------------------------------- #
#  Prompt Şablonu
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT_TR = textwrap.dedent("""\
    Sen bir sınır güvenliği kızılötesi kamera sistemi analiz asistanısın.
    Sana verilen termal anomali olay verisini analiz et ve aşağıdaki kurallara göre
    kısa, okunabilir Türkçe bir güvenlik uyarısı raporu yaz:

    - İlk cümle: Olayı kısaca özetle (ne tespit edildi, ne zaman, kaç kare sürdü).
    - İkinci cümle: Risk seviyesini değerlendir (düşük / orta / yüksek).
    - Üçüncü cümle: Öneri (gözlemci uyarısı, kayıt başlatma, doğrulama vb.).

    Maksimum 3 cümle, toplam 80 kelime. Teknik jargondan kaçın.
""")


def _build_user_prompt(event: AnomalyEvent, fps: float = 20.0) -> str:
    duration_s = event.duration_frames / max(fps, 1.0)
    anomaly_name_map = {
        "intruder": "sıcak insan hedefi (olası sızma)",
        "vehicle": "hızlı araç ısı izi",
        "heat_source": "statik ısı kaynağı / bırakılan nesne",
        "unknown": "tanımlanamayan anomali",
    }
    anom_name = anomaly_name_map.get(event.anomaly_type, event.anomaly_type)

    return textwrap.dedent(f"""\
        Olay Verisi:
        - Tespit Türü      : {anom_name}
        - Kare Aralığı     : {event.start_frame} – {event.end_frame}
        - Süre             : {event.duration_frames} kare ({duration_s:.1f} sn)
        - Zirve Anomali Skoru: {event.peak_score:.4f}
        - Ortalama Skor    : {event.mean_score:.4f}
        - Başlangıç Zamanı : {event.start_time}

        Yukarıdaki veriye dayanarak kısa güvenlik uyarısı raporu yaz.
    """)


# --------------------------------------------------------------------------- #
#  Kural Tabanlı (Rule-Based) Şablon Üretici
# --------------------------------------------------------------------------- #
def _rule_based_summary(event: AnomalyEvent, fps: float = 20.0) -> str:
    """API gerektirmeyen şablon tabanlı Türkçe özet üretici."""
    duration_s = event.duration_frames / max(fps, 1.0)

    type_desc = {
        "intruder": "kızılötesi görüntüde sıcak insan silueti tespit edildi — olası yetkisiz giriş",
        "vehicle": "sınır hattına hızla yaklaşan araç ısı izi tespit edildi",
        "heat_source": "statik bölgede aniden beliren şüpheli ısı kaynağı / nesne tespit edildi",
        "unknown": "termal anomali tespit edildi",
    }
    desc = type_desc.get(event.anomaly_type, "termal anomali tespit edildi")

    if event.peak_score >= 0.5:
        risk = "YÜKSEK"
        action = "Bölgeye derhal gözlemci sevk edilmesi ve kayıt başlatılması önerilir."
    elif event.peak_score >= 0.25:
        risk = "ORTA"
        action = "Bölgenin yakın izlemesi ve bir sonraki kare döngüsünde doğrulama yapılması önerilir."
    else:
        risk = "DÜŞÜK"
        action = "Sistem kaydı incelenmeli; gerekirse saha teyidi yapılabilir."

    return (
        f"Kare {event.start_frame}–{event.end_frame} aralığında ({duration_s:.1f} sn) "
        f"{desc} (zirve skor: {event.peak_score:.3f}). "
        f"Risk seviyesi: {risk} — {action}"
    )


# --------------------------------------------------------------------------- #
#  Ana Sınıf
# --------------------------------------------------------------------------- #
class LLMSummarizer:
    """
    AnomalyEvent nesneleri için Türkçe doğal dil uyarı özeti üretir.

    Örnek::

        summarizer = LLMSummarizer(backend="ollama", model="llama3")
        summary = summarizer.summarize(event)
        print(summary)
    """

    def __init__(
        self,
        backend: str = "rule",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        fps: float = 20.0,
        timeout: float = 15.0,
    ):
        """
        Parameters
        ----------
        backend     : 'rule' | 'ollama' | 'openai' | 'google'
        model       : LLM model adı (backend'e göre varsayılan seçilir)
        api_key     : OpenAI / Google API anahtarı (None → çevre değişkeninden okunur)
        ollama_url  : Ollama API URL
        fps         : Video FPS (süre hesabı için)
        timeout     : API isteği zaman aşımı (saniye)
        """
        self.backend = backend.lower()
        self.fps = fps
        self.timeout = timeout
        self.ollama_url = ollama_url.rstrip("/")

        # Model varsayılanları
        _defaults = {
            "ollama": "llama3",
            "openai": "gpt-4o-mini",
            "google": "gemini-1.5-flash",
        }
        self.model = model or _defaults.get(self.backend, "")
        self.api_key = api_key

    # ------------------------------------------------------------------ #
    def summarize(self, event: AnomalyEvent) -> str:
        """
        Anomali olayı için Türkçe özet üretir.
        Hata durumunda kural tabanlı özete fallback yapılır.
        """
        try:
            if self.backend == "rule":
                return _rule_based_summary(event, self.fps)
            elif self.backend == "ollama":
                return self._ollama(event)
            elif self.backend == "openai":
                return self._openai(event)
            elif self.backend == "google":
                return self._google(event)
            else:
                raise ValueError(f"Bilinmeyen backend: {self.backend}")
        except Exception as exc:
            # Herhangi bir hata durumunda kural tabanlı özete dön
            fallback = _rule_based_summary(event, self.fps)
            return f"[LLM hatası: {exc}] {fallback}"

    # ------------------------------------------------------------------ #
    def _ollama(self, event: AnomalyEvent) -> str:
        """Lokal Ollama sunucusuna istek atar."""
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "prompt": f"{SYSTEM_PROMPT_TR}\n\n{_build_user_prompt(event, self.fps)}",
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()

    # ------------------------------------------------------------------ #
    def _openai(self, event: AnomalyEvent) -> str:
        """OpenAI API ile özet üretir."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai paketi kurulu değil: pip install openai")

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TR},
                {"role": "user", "content": _build_user_prompt(event, self.fps)},
            ],
            max_tokens=200,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    def _google(self, event: AnomalyEvent) -> str:
        """Google Gemini API ile özet üretir."""
        import urllib.request

        key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY çevre değişkeni tanımlı değil")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={key}"
        full_prompt = f"{SYSTEM_PROMPT_TR}\n\n{_build_user_prompt(event, self.fps)}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"maxOutputTokens": 200},
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    # ------------------------------------------------------------------ #
    def summarize_and_attach(self, event: AnomalyEvent, logger=None) -> str:
        """
        Özet üretir, event'e ve (varsa) logger'a geri yazar.
        """
        summary = self.summarize(event)
        event.llm_summary = summary
        if logger is not None:
            logger.set_event_llm_summary(event.event_id, summary)
        return summary
