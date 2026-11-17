"""
setup_proyek.py  -  Tahap 0 (persiapan lingkungan), satu file.

Cara pakai (jalankan dari folder tempat skrip ini disimpan):
    python setup_proyek.py              # buat folder + requirements.txt + config.py + utils.py
    python setup_proyek.py --instal     # sama seperti di atas, lalu pip install -r requirements.txt
    python setup_proyek.py --cek        # cek versi library + uji package ID Play Store (setelah instal)
    python setup_proyek.py --timpa      # timpa file yang sudah ada (default: dilewati)

Skrip membuat folder 'penelitian-ojol/' di sebelah skrip ini.
Letakkan skrip DI LUAR folder penelitian-ojol agar tidak terjadi folder bersarang.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

NAMA_PROYEK = "penelitian-ojol"
BASE = Path(__file__).resolve().parent / NAMA_PROYEK

FOLDER = [
    "data/raw", "data/interim", "data/processed", "data/splits",
    "outputs/figures", "outputs/tables", "outputs/models",
    "notebooks",
]

# --------------------------------------------------------------------------
# Isi requirements.txt
# --------------------------------------------------------------------------
REQUIREMENTS = r'''# --- Inti ---
pandas
numpy
scikit-learn
scipy
pyarrow
joblib
tqdm
jupyter
# --- Scraping & pra-pemrosesan ---
google-play-scraper
emoji
Sastrawi
# --- Visualisasi ---
matplotlib
seaborn
wordcloud
# --- Topik ---
gensim
bertopic
sentence-transformers
umap-learn
hdbscan
# --- Klasifikasi ---
imbalanced-learn
# --- IndoBERT (dipakai di Colab pada Tahap 10; boleh dilewati di lokal) ---
torch
transformers
datasets
accelerate
'''

# --------------------------------------------------------------------------
# Isi config.py
# --------------------------------------------------------------------------
CONFIG_PY = r'''import os
from pathlib import Path

# ===== Reproducibility =====
SEED = 42

# ===== Path =====
# Di Colab, atur dulu: os.environ["OJOL_BASE_DIR"] = "/content/drive/MyDrive/penelitian-ojol"
BASE_DIR   = Path(os.environ.get("OJOL_BASE_DIR", ".")).resolve()
RAW_DIR    = BASE_DIR / "data" / "raw"
INTERIM_DIR= BASE_DIR / "data" / "interim"
PROC_DIR   = BASE_DIR / "data" / "processed"
SPLIT_DIR  = BASE_DIR / "data" / "splits"
FIG_DIR    = BASE_DIR / "outputs" / "figures"
TAB_DIR    = BASE_DIR / "outputs" / "tables"
MODEL_DIR  = BASE_DIR / "outputs" / "models"

# ===== Scraping =====
# Kunci = nama aplikasi di laporan, nilai = package ID di Google Play.
# PERLU DIVERIFIKASI lewat `python setup_proyek.py --cek` (ID bisa berubah/salah).
APPS = {
    "Gojek": "com.gojek.app",
    "Grab":  "com.grabtaxi.passenger",
    "Maxim": "com.taxsee.taxsee",
}
LANG, COUNTRY = "id", "id"
START_DATE = "2024-01-01"
END_DATE   = "2026-12-31"
TARGET_PER_APP = 10_000
BATCH_SIZE = 200          # jumlah ulasan per permintaan
SLEEP_SECONDS = 1.5       # jeda antarpermintaan (sopan terhadap server)
MAX_RETRY = 5

# ===== Pelabelan =====
# rating 1-2 = negatif, 3 = netral, 4-5 = positif
def rating_ke_label(r: int) -> str:
    if r <= 2:
        return "negatif"
    if r == 3:
        return "netral"
    return "positif"

LABEL2ID = {"negatif": 0, "netral": 1, "positif": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
UKURAN_SAMPEL_VALIDASI = 500

# ===== Pembagian data =====
RASIO_SPLIT = (0.8, 0.1, 0.1)   # latih / validasi / uji
'''

# --------------------------------------------------------------------------
# Isi utils.py
# --------------------------------------------------------------------------
UTILS_PY = r'''import os, random, sys, platform
import numpy as np
from config import *

def set_seed(seed: int = SEED):
    """Kunci seed untuk Python, NumPy, dan PyTorch (jika terpasang)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # efektif penuh hanya bila diset SEBELUM Python dimulai
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def make_dirs():
    """Buat semua folder yang didefinisikan di config."""
    for d in [RAW_DIR, INTERIM_DIR, PROC_DIR, SPLIT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def cek_lingkungan(simpan_ke=None):
    """Cetak dan (opsional) simpan versi library penting untuk bagian Metode."""
    from importlib.metadata import version, PackageNotFoundError
    paket = ["pandas", "numpy", "scikit-learn", "gensim", "bertopic",
             "sentence-transformers", "umap-learn", "hdbscan",
             "google-play-scraper", "torch", "transformers"]
    baris = [f"Python {sys.version.split()[0]} | {platform.platform()}"]
    for p in paket:
        try:
            baris.append(f"{p}: {version(p)}")
        except PackageNotFoundError:
            baris.append(f"{p}: TIDAK TERPASANG")
    teks = "\n".join(baris)
    print(teks)
    if simpan_ke:
        simpan_ke.write_text(teks, encoding="utf-8")
    return teks
'''


def tulis_file(path: Path, isi: str, timpa: bool):
    """Tulis file; lewati bila sudah ada kecuali --timpa."""
    if path.exists() and not timpa:
        print(f"[LEWATI] {path.name} sudah ada (pakai --timpa untuk menimpa)")
        return
    path.write_text(isi, encoding="utf-8")
    print(f"[DIBUAT] {path.relative_to(BASE.parent)}")


def buat_proyek(timpa: bool):
    for f in FOLDER:
        (BASE / f).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Folder proyek siap di: {BASE}")
    tulis_file(BASE / "requirements.txt", REQUIREMENTS, timpa)
    tulis_file(BASE / "config.py", CONFIG_PY, timpa)
    tulis_file(BASE / "utils.py", UTILS_PY, timpa)


def instal():
    req = BASE / "requirements.txt"
    print(f"\n[INSTAL] pip install -r {req} (bisa lama dan berukuran besar)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


def cek():
    """Verifikasi lingkungan + uji package ID Play Store (setara Blok 5)."""
    os.chdir(BASE)
    sys.path.insert(0, str(BASE))
    from config import APPS, LANG, COUNTRY, SEED, TAB_DIR
    from utils import make_dirs, set_seed, cek_lingkungan

    make_dirs()
    set_seed(SEED)
    cek_lingkungan(simpan_ke=TAB_DIR / "versi_lingkungan.txt")

    try:
        import torch
        print("\nGPU tersedia:", torch.cuda.is_available())
    except ImportError:
        print("\ntorch belum terpasang (tidak masalah untuk tahap 1-9)")

    try:
        from google_play_scraper import app, reviews, Sort
    except ImportError:
        print("\ngoogle-play-scraper belum terpasang. Jalankan dulu: python setup_proyek.py --instal")
        return

    print("\n=== Uji package ID ===")
    for nama, pkg in APPS.items():
        try:
            info = app(pkg, lang=LANG, country=COUNTRY)
            print(f"[OK]    {nama:6s} -> {info['title']} | skor {info['score']:.2f} | "
                  f"jumlah ulasan {info['reviews']:,}")
            hasil, _ = reviews(pkg, lang=LANG, country=COUNTRY, sort=Sort.NEWEST, count=3)
            print(f"        sampel ulasan terbaru: {len(hasil)} baris diterima")
        except Exception as e:
            print(f"[GAGAL] {nama:6s} ({pkg}) -> {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Persiapan proyek penelitian ulasan ojol (Tahap 0)")
    ap.add_argument("--instal", action="store_true", help="jalankan pip install -r requirements.txt")
    ap.add_argument("--cek", action="store_true", help="cek versi library dan uji package ID Play Store")
    ap.add_argument("--timpa", action="store_true", help="timpa file yang sudah ada")
    args = ap.parse_args()

    if args.cek:
        cek()
        return
    buat_proyek(args.timpa)
    if args.instal:
        instal()
    print("\nSelesai. Langkah berikutnya: python setup_proyek.py --cek")


if __name__ == "__main__":
    main()
