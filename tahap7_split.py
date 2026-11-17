"""
tahap7_split.py  -  Tahap 7: pembagian data stratified 80:10:10 (latih / validasi / uji).

Satu pembagian dipakai oleh SEMUA model (Tahap 8, 9, 10, 11) agar perbedaan kinerja berasal dari metode,
bukan dari data. Penanganan ketidakseimbangan kelas (class weight/oversampling) dilakukan nanti HANYA
pada data latih.

Letakkan di sebelah folder penelitian-ojol/ atau di dalamnya, lalu jalankan:

    python tahap7_split.py

Opsi:
    --strata label_app      stratifikasi menurut label x aplikasi (default; menjaga proporsi label per aplikasi)
    --strata label          stratifikasi menurut label saja
    --timpa                 izinkan menimpa pembagian yang sudah ada (HATI-HATI: model lama jadi tidak sebanding)
    --izinkan-sementara     izinkan memakai label yang BELUM divalidasi (hasil diberi akhiran _sementara)

Input : data/processed/ulasan_berlabel.csv   (hasil Tahap 3, setelah --hitung selesai)
Output: data/splits/split.csv                (reviewId, split)  <- sumber kebenaran
        data/splits/train.csv, val.csv, test.csv      (kolom penting, siap dipakai model dan diunggah ke Colab)
        data/splits/split_info.json          (parameter, jumlah, dan tanda tangan md5 untuk verifikasi)
        outputs/tables/split_ringkasan.csv, split_proporsi.csv
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
PROYEK = HERE if (HERE / "config.py").exists() else HERE / "penelitian-ojol"
if not (PROYEK / "config.py").exists():
    sys.exit("config.py tidak ditemukan. Jalankan dulu setup_proyek.py, lalu letakkan skrip ini "
             "di sebelah folder penelitian-ojol/ atau di dalamnya.")
os.chdir(PROYEK)
sys.path.insert(0, str(PROYEK))

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")

from config import *            # noqa: F401,F403  (SEED, RASIO_SPLIT, PROC_DIR, SPLIT_DIR, TAB_DIR, LABEL2ID, ...)
from utils import set_seed, make_dirs

FILE_MASUK = PROC_DIR / "ulasan_berlabel.csv"
URUT_LABEL = ["negatif", "netral", "positif"]
KOLOM_DATA = ["reviewId", "app", "tanggal", "rating", "thumbs_up", "label_final", "label_id", "tervalidasi",
              "dikoreksi", "menyebut_bintang", "teks_asli", "teks_ringan", "teks_bersih", "teks_topik"]
MIN_NETRAL_UJI_PER_APP = 30       # peringatan bila kelas netral di data uji per aplikasi lebih sedikit dari ini


def md5_file(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Tahap 7: pembagian data stratified")
    ap.add_argument("--strata", choices=["label_app", "label"], default="label_app")
    ap.add_argument("--timpa", action="store_true")
    ap.add_argument("--izinkan-sementara", action="store_true")
    args = ap.parse_args()

    make_dirs()
    set_seed(SEED)
    pd.set_option("display.width", 200)

    if not FILE_MASUK.exists():
        sys.exit(f"{FILE_MASUK} tidak ada. Selesaikan Tahap 3 lebih dulu.")
    d = pd.read_csv(FILE_MASUK, keep_default_na=False,
                    dtype={"reviewId": str, "teks_asli": str, "teks_ringan": str, "teks_bersih": str,
                           "teks_topik": str})
    perlu = {"reviewId", "app", "label_final", "label_id", "teks_ringan", "teks_bersih"}
    if not perlu <= set(d.columns):
        sys.exit(f"Kolom tidak lengkap di {FILE_MASUK.name}; kurang: {sorted(perlu - set(d.columns))}")
    d["tervalidasi"] = d["tervalidasi"].astype(str).str.lower().eq("true") if "tervalidasi" in d else False

    # ---------- Penjaga: label harus sudah divalidasi ----------
    n_valid = int(d["tervalidasi"].sum())
    sementara = n_valid == 0
    if sementara and not args.izinkan_sementara:
        sys.exit("Belum ada label yang divalidasi manual (kolom 'tervalidasi' semua False): validasi Tahap 3 "
                 "belum selesai. Pembagian sebaiknya memakai label yang sudah dikoreksi. Selesaikan "
                 "'python tahap3_pelabelan.py --hitung', atau pakai --izinkan-sementara untuk uji coba "
                 "(hasil diberi akhiran _sementara dan tidak boleh dipakai untuk hasil akhir).")
    sfx = "_sementara" if sementara else ""
    if sementara:
        print("[PERHATIAN] Label masih SEMENTARA (dari rating). Keluaran diberi akhiran _sementara.\n")

    f_split = SPLIT_DIR / f"split{sfx}.csv"
    if f_split.exists() and not args.timpa:
        sys.exit(f"{f_split.name} sudah ada. Pembagian harus tetap sama untuk semua model; menimpanya membuat "
                 f"model yang sudah dilatih tidak sebanding. Gunakan --timpa hanya bila Anda sengaja mengulang "
                 f"seluruh pelatihan.")

    # ---------- Pemeriksaan data ----------
    assert d["reviewId"].is_unique, "reviewId tidak unik"
    assert set(d["label_final"]) <= set(URUT_LABEL), f"label tak dikenal: {set(d['label_final']) - set(URUT_LABEL)}"
    print(f"Ulasan: {len(d):,} | label tervalidasi manual: {n_valid}")
    print(d["label_final"].value_counts().reindex(URUT_LABEL).to_string())

    # ---------- Pembagian ----------
    r_latih, r_val, r_uji = RASIO_SPLIT
    strata = d["label_final"] if args.strata == "label" else d["label_final"] + "|" + d["app"]
    idx = np.arange(len(d))
    idx_latih, idx_sisa = train_test_split(idx, test_size=1 - r_latih, stratify=strata, random_state=SEED,
                                           shuffle=True)
    idx_val, idx_uji = train_test_split(idx_sisa, test_size=r_uji / (r_val + r_uji),
                                        stratify=strata.iloc[idx_sisa], random_state=SEED, shuffle=True)
    d["split"] = ""
    d.loc[idx_latih, "split"] = "train"
    d.loc[idx_val, "split"] = "val"
    d.loc[idx_uji, "split"] = "test"
    assert (d["split"] != "").all() and len(set(idx_latih) & set(idx_val)) == 0 \
        and len(set(idx_latih) & set(idx_uji)) == 0 and len(set(idx_val) & set(idx_uji)) == 0

    # ---------- Simpan ----------
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    d[["reviewId", "split"]].to_csv(f_split, index=False, encoding="utf-8")
    kolom = [c for c in KOLOM_DATA if c in d.columns]
    for nama in ("train", "val", "test"):
        d[d["split"] == nama][kolom].to_csv(SPLIT_DIR / f"{nama}{sfx}.csv", index=False, encoding="utf-8")

    # ---------- Ringkasan dan pemeriksaan ----------
    urut = ["train", "val", "test"]
    jumlah = pd.crosstab(d["split"], d["label_final"]).reindex(index=urut, columns=URUT_LABEL)
    jumlah["total"] = jumlah.sum(axis=1)
    persen = pd.crosstab(d["split"], d["label_final"], normalize="index").reindex(index=urut, columns=URUT_LABEL) * 100
    persen.loc["SEMUA"] = d["label_final"].value_counts(normalize=True).reindex(URUT_LABEL) * 100
    selisih = (persen.loc[urut] - persen.loc["SEMUA"]).abs().max().max()

    print("\n=== JUMLAH PER SPLIT DAN LABEL ===")
    print(jumlah.to_string())
    print("\n=== PERSEN LABEL PER SPLIT (harus hampir sama) ===")
    print(persen.round(2).to_string())
    print(f"Selisih terbesar terhadap proporsi keseluruhan: {selisih:.2f} poin persen")

    app_label = pd.crosstab([d["split"], d["app"]], d["label_final"]).reindex(columns=URUT_LABEL)
    print("\n=== JUMLAH PER SPLIT x APLIKASI x LABEL ===")
    print(app_label.to_string())
    netral_uji = app_label.xs("test", level="split")["netral"]
    for a, n in netral_uji.items():
        if n < MIN_NETRAL_UJI_PER_APP:
            print(f"[PERHATIAN] kelas netral pada data uji {a} hanya {n} ulasan; metrik per aplikasi untuk "
                  f"kelas ini akan sangat tidak stabil (laporkan selang kepercayaan).")

    n_dup = int(d.duplicated("teks_bersih", keep=False).sum())
    teks_lintas = d.groupby("teks_bersih")["split"].nunique()
    n_bocor = int((teks_lintas > 1).sum())
    print(f"\nTeks_bersih ganda: {n_dup} baris | teks yang muncul di lebih dari satu split: {n_bocor} "
          f"(harus 0; bila >0, ada kebocoran data)")
    valid_split = d[d["tervalidasi"]].groupby("split").size().reindex(urut, fill_value=0)
    print("Ulasan tervalidasi manual per split:", valid_split.to_dict())

    info = {"waktu": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "seed": SEED, "rasio": list(RASIO_SPLIT),
            "strata": args.strata, "n": len(d), "n_per_split": d["split"].value_counts().to_dict(),
            "label_sementara": sementara, "n_tervalidasi": n_valid,
            "md5_split_csv": md5_file(f_split), "selisih_proporsi_maks_poin": round(float(selisih), 3)}
    (SPLIT_DIR / f"split_info{sfx}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    jumlah.to_csv(TAB_DIR / f"split_ringkasan{sfx}.csv", encoding="utf-8")
    persen.round(3).to_csv(TAB_DIR / f"split_proporsi{sfx}.csv", encoding="utf-8")
    app_label.to_csv(TAB_DIR / f"split_app_label{sfx}.csv", encoding="utf-8")

    print(f"\nTersimpan: {f_split}")
    print(f"           {SPLIT_DIR / f'train{sfx}.csv'} | val{sfx}.csv | test{sfx}.csv")
    print(f"Tanda tangan md5 (untuk verifikasi pada Tahap 8-11): {info['md5_split_csv']}")
    print("Hasil ini TIDAK boleh diubah lagi setelah model dilatih. Cadangkan folder data/splits/.")


if __name__ == "__main__":
    main()
