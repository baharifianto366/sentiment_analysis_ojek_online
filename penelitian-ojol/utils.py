import os, random, sys, platform
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
