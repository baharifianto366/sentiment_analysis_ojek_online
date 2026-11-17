# -*- coding: utf-8 -*-
"""
Tahap 13: lampiran reproduksibilitas + pemeriksaan kelengkapan bahan Hasil dan Pembahasan.

Pemakaian (PowerShell, dari folder Setup_proyek):
  python tahap13_lampiran.py

Skrip hanya MEMBACA file proyek. Yang ditulis hanya:
  outputs/tables/t13_lampiran_reproduksibilitas.txt
  outputs/tables/t13_daftar_file.csv
"""
import sys, os, json, hashlib, platform, datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = Path(__file__).resolve().parent
PROYEK = HERE if (HERE / "config.py").exists() else HERE / "penelitian-ojol"
os.chdir(PROYEK); sys.path.insert(0, str(PROYEK))
from config import *            # SEED, path, APPS, ...
from utils import set_seed, make_dirs

import pandas as pd
from importlib import metadata

# =====================================================================
# KONFIGURASI
# =====================================================================
TAB, FIG = Path(TAB_DIR), Path(FIG_DIR)
DATA = Path(RAW_DIR).parent
OUT = TAB.parent
PAKET = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn", "gensim", "Sastrawi",
         "emoji", "bertopic", "sentence-transformers", "umap-learn", "hdbscan", "wordcloud",
         "openpyxl", "joblib", "google-play-scraper", "torch", "transformers"]
# angka acuan dari hasil tahap sebelumnya (untuk pemeriksaan konsistensi)
ACUAN_N = {"ulasan_berlabel": 29603, "train": 23682, "val": 2960, "test": 2961}
ACUAN_LABEL = {"negatif": 10963, "netral": 1468, "positif": 17172}
# bahan per bagian laporan (pola relatif terhadap folder proyek). Nama file EDA hanyalah tebakan.
BAHAN = {
    "Data dan pra-pemrosesan": ["data/processed/ulasan_berlabel.csv", "data/splits/split.csv", "data/splits/split_info.json"],
    "Validasi label (Tahap 3)": ["outputs/tables/kappa_validasi.csv", "outputs/tables/ketidaksesuaian_label.csv",
                                 "outputs/tables/contoh_ketidaksesuaian.csv"],
    "EDA (Tahap 4; nama file tebakan)": ["outputs/tables/eda_*.csv", "outputs/figures/eda_*.png"],
    "LDA (Tahap 5)": ["outputs/tables/lda_topik_k12_inklusif.csv", "outputs/tables/lda_penamaan_topik_k12_inklusif.csv",
                      "outputs/tables/lda_topik_dokumen_k12_inklusif.csv"],
    "BERTopic (Tahap 6)": ["outputs/tables/bertopic_topik*.csv", "outputs/tables/bertopic_penamaan*.csv",
                           "outputs/tables/perbandingan_lda_bertopic*.csv"],
    "Klasifikasi (Tahap 8-10)": ["outputs/tables/tfidf_*.csv", "outputs/tables/emb_*.csv", "outputs/tables/indobert_*.csv",
                                  "outputs/predictions/indobert_test.csv"],
    "Evaluasi (Tahap 11)": ["outputs/tables/evaluasi_perbandingan_model.csv", "outputs/tables/evaluasi_uji_berpasangan.csv",
                            "outputs/tables/evaluasi_per_aplikasi.csv", "outputs/tables/evaluasi_ketahanan.csv",
                            "outputs/tables/evaluasi_pasangan_kesalahan.csv", "outputs/tables/evaluasi_ringkasan.txt",
                            "outputs/tables/evaluasi_kesalahan_per_ulasan.csv", "outputs/figures/evaluasi_*.png"],
    "12A sentimen per aplikasi": ["outputs/tables/t12a_sentimen_app_sampel.csv", "outputs/tables/t12a_selisih_antar_app.csv",
                                  "outputs/tables/t12a_sentimen_app_populasi_semua.csv",
                                  "outputs/tables/t12a_sentimen_app_populasi_min3kata_unik.csv",
                                  "outputs/tables/t12a_tren_bulanan_negatif.csv", "outputs/tables/t12a_tren_tahunan.csv",
                                  "outputs/figures/t12a_*.png"],
    "12B sentimen per topik": ["outputs/tables/pemetaan_topik_aspek.csv", "outputs/tables/t12b_lda_sentimen_dalam_topik.csv",
                               "outputs/tables/t12b_lda_komposisi_dalam_negatif.csv",
                               "outputs/tables/t12b_bertopic*_sentimen_dalam_topik.csv",
                               "outputs/tables/t12b_keamanan_kata_kunci.csv", "outputs/figures/t12b_*.png"],
    "12C analisis kesalahan": ["outputs/tables/t12c_kelompok_kesalahan.csv", "outputs/tables/t12c_pola_kesalahan.csv",
                               "outputs/tables/t12c_galat_per_dimensi.csv",
                               "outputs/tables/t12c_semua_salah_menurut_label_rating.csv",
                               "outputs/tables/t12c_rekap_pengodean.csv", "outputs/tables/t12c_contoh_per_kategori.csv"],
}

L = []


def cetak(*a):
    s = " ".join(str(x) for x in a)
    print(s); L.append(s)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def n_baris(p):
    try:
        return len(pd.read_csv(p, dtype=str, keep_default_na=False, usecols=[0], encoding="utf-8-sig"))
    except Exception as e:
        return f"galat: {type(e).__name__}"


def versi(pkg):
    try:
        return metadata.version(pkg)
    except Exception:
        return "tidak terpasang"


def main():
    set_seed(SEED); make_dirs()
    cetak("LAMPIRAN REPRODUKSIBILITAS")
    cetak("dibuat:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    cetak("\n== 1. Lingkungan ==")
    cetak("Python:", platform.python_version(), "|", platform.platform())
    cetak("SEED:", SEED)
    for p in PAKET:
        cetak(f"  {p}: {versi(p)}")

    cetak("\n== 2. Pembagian data (tidak boleh berubah) ==")
    split = DATA / "splits" / "split.csv"
    if split.exists():
        m = md5_file(split)
        cetak("md5 split.csv:", m)
        info = DATA / "splits" / "split_info.json"
        if info.exists():
            cetak("  cocok dengan split_info.json:", "YA" if m in info.read_text(encoding="utf-8", errors="replace") else "TIDAK <- periksa!")
        for pj in sorted(OUT.rglob("*parameter*.json")):
            ok = m in pj.read_text(encoding="utf-8", errors="replace")
            cetak(f"  {pj.name}: md5 {'cocok' if ok else 'TIDAK cocok <- periksa!'}")
    else:
        cetak("split.csv tidak ditemukan.")

    cetak("\n== 3. Jumlah data (dibandingkan dengan angka acuan hasil tahap sebelumnya) ==")
    f_lab = DATA / "processed" / "ulasan_berlabel.csv"
    for nama, p in [("ulasan_berlabel", f_lab), ("train", DATA / "splits" / "train.csv"),
                    ("val", DATA / "splits" / "val.csv"), ("test", DATA / "splits" / "test.csv")]:
        if p.exists():
            n = n_baris(p)
            tanda = "" if n == ACUAN_N[nama] else f"  <- BEDA dari acuan {ACUAN_N[nama]:,}"
            cetak(f"  {nama}: {n:,}{tanda}" if isinstance(n, int) else f"  {nama}: {n}")
        else:
            cetak(f"  {nama}: file tidak ada")
    if f_lab.exists():
        try:
            d = pd.read_csv(f_lab, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=["app", "label_final"])
            vc = d["label_final"].value_counts().to_dict()
            cetak("  label_final:", vc)
            for k, v in ACUAN_LABEL.items():
                if vc.get(k) != v:
                    cetak(f"  <- label '{k}' beda dari acuan {v:,}")
            cetak("  per aplikasi:", d["app"].value_counts().to_dict())
        except Exception as e:
            cetak("  gagal membaca label_final:", type(e).__name__)

    cetak("\n== 4. Kelengkapan bahan per bagian laporan ==")
    lengkap = True
    for bagian, pola in BAHAN.items():
        cetak(f"[{bagian}]")
        for pl in pola:
            cocok = sorted(PROYEK.glob(pl))
            if cocok:
                cetak(f"   OK ({len(cocok)}): {pl}")
            else:
                lengkap = False
                cetak(f"   TIDAK ADA: {pl}")
    cetak("-> Semua bahan ada." if lengkap else "-> Ada bahan yang belum ada; jalankan tahap terkait atau cek nama file sebenarnya.")

    cetak("\n== 5. Isian manual yang menentukan angka laporan ==")
    f_pem = TAB / "pemetaan_topik_aspek.csv"
    if f_pem.exists():
        try:
            p = pd.read_csv(f_pem, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            cetak(f"  pemetaan_topik_aspek.csv: {len(p)} topik; nama_manual kosong: {(p['nama_manual'].str.strip() == '').sum()}; "
                  f"kategori_aspek kosong: {(p['kategori_aspek'].str.strip() == '').sum()}")
        except Exception as e:
            cetak("  pemetaan_topik_aspek.csv tidak terbaca:", type(e).__name__)
    else:
        cetak("  pemetaan_topik_aspek.csv belum ada (jalankan 12B).")
    f_x = TAB / "t12c_templat_pengodean_kesalahan.xlsx"
    if f_x.exists():
        try:
            x = pd.read_excel(f_x, sheet_name="pengodean", dtype=str).fillna("")
            cetak(f"  pengodean kesalahan: kode_1 terisi {(x['kode_1'].str.strip() != '').sum()}/{len(x)}; "
                  f"kode_penilai2 terisi {(x['kode_penilai2'].str.strip() != '').sum()} (minimal 30 untuk kappa)")
        except Exception as e:
            cetak("  templat pengodean tidak terbaca (Excel terbuka?):", type(e).__name__)
    else:
        cetak("  templat pengodean belum ada (jalankan 12C).")

    cetak("\n== 6. Daftar file ==")
    baris = []
    for akar in [DATA, OUT]:
        for p in sorted(akar.rglob("*")):
            if p.is_file():
                st = p.stat()
                baris.append({"path": str(p.relative_to(PROYEK)), "ukuran_kb": round(st.st_size / 1024, 1),
                              "diubah": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")})
    df = pd.DataFrame(baris)
    cetak(f"  {len(df)} file (rincian di t13_daftar_file.csv)")
    try:
        df.to_csv(TAB / "t13_daftar_file.csv", index=False, encoding="utf-8-sig")
        (TAB / "t13_lampiran_reproduksibilitas.txt").write_text("\n".join(L), encoding="utf-8")
    except PermissionError:
        raise SystemExit("Tidak bisa menulis hasil Tahap 13. Tutup file di Excel / tunggu OneDrive / pindahkan proyek ke C:\\riset\\.")
    print("\ntersimpan: t13_lampiran_reproduksibilitas.txt dan t13_daftar_file.csv di", TAB)


if __name__ == "__main__":
    main()
