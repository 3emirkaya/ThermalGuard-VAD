"""
KAIST Multispectral Pedestrian Benchmark — LWIR-only subset downloader.

Sadece termal (LWIR) kanalı ve yaya anotasyonları indirilir; RGB kanalı
bilinçli olarak atlanır (bu projede renk bilgisi kullanılmaz).

Kaynaklar (Hugging Face aynaları — orijinal: SoonminHwang/rgbt-ped-detection):
  * Görüntüler : richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset
  * Anotasyon  : sywise98/kaist-labels   (annotations-xml-new)

Örnek:
    python scripts/download_kaist.py --seqs set00/V000 set03/V001 --stride 1
    python scripts/download_kaist.py --seqs set05 --stride 2 --out data/kaist

Bağımlılık gerektirmez (yalnızca Python stdlib).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMG_REPO = "richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset"
IMG_ROOT = "kaist_train"
ANN_REPO = "sywise98/kaist-labels"
ANN_ROOT = "annotations-xml-new"

HF = "https://huggingface.co"
UA = {"User-Agent": "ThermalGuard-VAD/1.0 (+research)"}


def _get(url: str, retries: int = 4, timeout: int = 30):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), dict(r.headers)
        except Exception as e:  # noqa: BLE001 - ağ hatalarında yeniden dene
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed: {url} ({last})")


def list_tree(repo: str, path: str) -> list[dict]:
    """HF tree API (sayfalı) — bir klasördeki tüm girdileri döndürür."""
    url = f"{HF}/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}"
    out: list[dict] = []
    while url:
        body, headers = _get(url)
        out.extend(json.loads(body))
        link = headers.get("Link") or headers.get("link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return out


def resolve_url(repo: str, path: str) -> str:
    return f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def expand_sequences(specs: list[str]) -> list[str]:
    """'set00' -> tüm V klasörleri, 'set00/V001' -> aynen."""
    seqs: list[str] = []
    for s in specs:
        s = s.strip("/")
        if "/" in s:
            seqs.append(s)
            continue
        entries = list_tree(IMG_REPO, f"{IMG_ROOT}/{s}")
        vs = sorted(e["path"].split("/")[-1] for e in entries
                    if e["type"] == "directory" and re.match(r"V\d{3}$", e["path"].split("/")[-1]))
        if vs:
            seqs += [f"{s}/{v}" for v in vs]
        else:  # bazı aynalarda set altında doğrudan lwir/ bulunur
            seqs.append(s)
    return seqs


def download_file(url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    data, _ = _get(url)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", nargs="+", default=["set00/V000"], help="setXX veya setXX/VYYY")
    ap.add_argument("--out", default="data/kaist", help="çıktı kök klasörü")
    ap.add_argument("--stride", type=int, default=1, help="her N. kareyi indir (1 = hepsi, 20 Hz)")
    ap.add_argument("--max-frames", type=int, default=0, help="sekans başına üst sınır (0 = sınırsız)")
    ap.add_argument("--no-ann", action="store_true", help="anotasyonları indirme")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    seqs = expand_sequences(args.seqs)
    print(f"[i] {len(seqs)} sekans: {', '.join(seqs)}")

    for seq in seqs:
        lwir_dir = f"{IMG_ROOT}/{seq}/lwir"
        files = sorted(e["path"] for e in list_tree(IMG_REPO, lwir_dir) if e["path"].endswith(".jpg"))
        files = files[:: max(1, args.stride)]
        if args.max_frames:
            files = files[: args.max_frames]
        jobs = []
        for f in files:
            name = Path(f).name
            jobs.append((resolve_url(IMG_REPO, f), out / "images" / seq / "lwir" / name))
            if not args.no_ann:
                ann = f"{ANN_ROOT}/{seq}/{Path(name).stem}.xml"
                jobs.append((resolve_url(ANN_REPO, ann), out / "annotations" / seq / f"{Path(name).stem}.xml"))

        t0, n_new, n_err = time.time(), 0, 0
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(download_file, u, d): d for u, d in jobs}
            for i, fu in enumerate(as_completed(futs), 1):
                try:
                    n_new += int(fu.result())
                except Exception as e:  # noqa: BLE001
                    n_err += 1
                    if n_err < 5:
                        print(f"   ! {e}", file=sys.stderr)
                if i % 200 == 0 or i == len(jobs):
                    print(f"\r[{seq}] {i}/{len(jobs)} dosya  ({time.time() - t0:.0f}s)", end="", flush=True)
        print(f"\n[{seq}] tamam: {len(files)} kare, {n_new} yeni dosya, {n_err} hata")


if __name__ == "__main__":
    main()
