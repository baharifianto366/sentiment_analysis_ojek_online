import os
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
