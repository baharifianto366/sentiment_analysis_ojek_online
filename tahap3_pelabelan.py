"""
tahap3_pelabelan.py  -  Tahap 3: pelabelan sentimen, validasi manual, Cohen's kappa, koreksi label.

Karena ada pekerjaan manual oleh manusia, tahap ini dijalankan dalam tiga langkah:

  1) python tahap3_pelabelan.py --buat
       - memberi label otomatis dari rating (1-2 negatif, 3 netral, 4-5 positif)
       - mengambil sampel acak untuk validasi (default: acak berstrata app x label, ~500 ulasan)
       - membuat file Excel untuk dua penilai (tanpa rating dan tanpa label otomatis: penilaian buta)
       - menulis data/processed/ulasan_berlabel.csv (sementara label_final = label otomatis)

  2) Kedua penilai mengisi file masing-masing SECARA MANDIRI (tanpa berdiskusi).

  3) python tahap3_pelabelan.py --hitung
       - menghitung Cohen's kappa (antar-penilai) dan kesesuaian dengan label dari rating
       - bila ada ulasan yang dinilai berbeda oleh kedua penilai, membuat sengketa_penilai.xlsx;
         isi kolom 'label_sepakat' lewat diskusi, lalu jalankan --hitung sekali lagi
       - bila semua beres: koreksi label, estimasi tingkat ketidaksesuaian, tulis hasil akhir

Opsi:
  --strategi acak_sederhana    sampel acak sederhana murni (default: berstrata)
  --timpa                      izinkan --buat menimpa pekerjaan yang sudah ada (HATI-HATI)
  --file1 / --file2            lokasi file penilai bila tidak di folder default

Prasyarat tambahan: pip install openpyxl

Letakkan skrip ini di sebelah folder penelitian-ojol/ atau di dalamnya.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Lokasi proyek
# --------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CWD_AWAL = Path.cwd()               # untuk menafsirkan path relatif pada --file1/--file2
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
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")
try:
    import openpyxl  # noqa: F401
except ImportError:
    sys.exit("Library 'openpyxl' dibutuhkan untuk file Excel penilai. Jalankan: pip install openpyxl")

from config import *            # noqa: F401,F403  (SEED, LABEL2ID, rating_ke_label, ...)
from utils import set_seed, make_dirs

# --------------------------------------------------------------------------
# Konfigurasi tahap 3 (ubah di sini bila perlu)
# --------------------------------------------------------------------------
FILE_MASUK = PROC_DIR / "ulasan_bersih.csv"
FILE_KELUAR = PROC_DIR / "ulasan_berlabel.csv"
DIR_VALIDASI = INTERIM_DIR / "validasi"
FILE_KUNCI = DIR_VALIDASI / "kunci_sampel.csv"          # JANGAN diberikan ke penilai
FILE_PARAM = DIR_VALIDASI / "parameter_validasi.json"
FILE_SENGKETA = DIR_VALIDASI / "sengketa_penilai.xlsx"
FILE_PENILAI = {1: DIR_VALIDASI / "validasi_penilai1.xlsx", 2: DIR_VALIDASI / "validasi_penilai2.xlsx"}
FILE_PILOT = DIR_VALIDASI / "pilot_kalibrasi.xlsx"
FILE_PANDUAN = DIR_VALIDASI / "panduan_pelabelan.txt"

UKURAN_PER_STRATUM = 56         # 9 strata (3 app x 3 label) x 56 = 504 ulasan
UKURAN_ACAK_SEDERHANA = 500     # dipakai bila --strategi acak_sederhana
UKURAN_PILOT = 20               # ulasan latihan kalibrasi (di luar sampel utama)
N_BOOTSTRAP = 2000
LABEL_URUT = ["negatif", "netral", "positif"]

PANDUAN = """PANDUAN PELABELAN SENTIMEN (baca sebelum mulai)

Tujuan: menilai sentimen yang diungkapkan oleh TEKS ulasan, bukan jumlah bintangnya.
Anda tidak diberi rating. Jangan menebak-nebak rating; nilai isi teksnya saja.

Pilih SATU label untuk setiap ulasan:
  positif : pengguna puas, senang, memuji, atau menceritakan pengalaman yang baik.
  negatif : pengguna mengeluh, kecewa, marah, atau menceritakan masalah / pengalaman buruk.
  netral  : pertanyaan, informasi, atau saran tanpa kecenderungan puas/kecewa yang jelas,
            atau pujian dan keluhan yang benar-benar seimbang.

Aturan untuk kasus sulit:
  1. Sarkasme / ironi: nilai sentimen yang SEBENARNYA dimaksud.
     Contoh: "bagus banget nunggu 2 jam" = negatif.
  2. Campuran: pilih sentimen yang dominan atau paling kuat. Bila benar-benar seimbang = netral.
  3. Permintaan atau saran tanpa emosi ("tolong tambah fitur X") = netral.
     Bila disertai kekecewaan ("... dari dulu tidak ada, payah") = negatif.
  4. Slang, typo, campur bahasa: nilai maknanya. Bila tidak dapat dipahami, pilih netral
     dan isi kolom 'Ragu?' dengan 'ya'.
  5. Pujian atau keluhan untuk driver, tarif, promo, fitur, maupun aplikasi sama-sama dihitung.
     Yang dinilai adalah sikap pengguna secara keseluruhan.
  6. Bila teks menyebut jumlah bintang ("kasih bintang 1"), perlakukan sebagai bagian dari teks,
     tetapi tetap nilai keseluruhan isinya.
  7. JANGAN berdiskusi dengan penilai lain sebelum kedua file dikumpulkan.

Kolom 'Ragu?': isi 'ya' bila Anda tidak yakin (tetap pilih satu label).
Kolom 'Catatan' (opsional): mis. 'sarkasme', 'campur bahasa', 'tidak jelas'.
Jangan mengubah kolom 'Kode' dan jangan mengurutkan ulang baris.
"""


# --------------------------------------------------------------------------
# Fungsi bantu
# --------------------------------------------------------------------------
def muat_data():
    if not FILE_MASUK.exists():
        sys.exit(f"{FILE_MASUK} tidak ada. Selesaikan Tahap 2 lebih dulu.")
    df = pd.read_csv(FILE_MASUK, parse_dates=["tanggal"], keep_default_na=False,
                     dtype={"teks_asli": str, "teks_ringan": str, "teks_bersih": str,
                            "teks_topik": str, "reviewId": str})
    perlu = {"app", "reviewId", "rating", "teks_asli"}
    if not perlu <= set(df.columns):
        sys.exit(f"Kolom wajib tidak lengkap di {FILE_MASUK.name}: butuh {sorted(perlu)}")
    df["label_otomatis"] = df["rating"].map(rating_ke_label)
    df["stratum"] = df["app"] + "|" + df["label_otomatis"]
    return df


def simpan_berlabel(df, sampel_final=None):
    """Tulis ulasan_berlabel.csv. sampel_final: DataFrame(reviewId, label_final) hasil validasi."""
    out = df.drop(columns=["stratum"]).copy()
    out["label_final"] = out["label_otomatis"]
    out["tervalidasi"] = False
    out["dikoreksi"] = False
    if sampel_final is not None:
        peta = dict(zip(sampel_final["reviewId"], sampel_final["label_final"]))
        ada = out["reviewId"].isin(peta)
        out.loc[ada, "label_final"] = out.loc[ada, "reviewId"].map(peta)
        out.loc[ada, "tervalidasi"] = True
        out["dikoreksi"] = out["label_final"] != out["label_otomatis"]
    out["label_id"] = out["label_final"].map(LABEL2ID)
    out.to_csv(FILE_KELUAR, index=False, encoding="utf-8")
    return out


def bersihkan_sel(t):
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(t))     # karakter ilegal untuk Excel
    return " " + t if t[:1] in "=+-@" else t                      # cegah dibaca sebagai rumus


def tulis_excel(path, daftar_item):
    """daftar_item: list (kode, teks). Dropdown label agar tidak ada salah ketik."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["Kode", "Teks ulasan", "Label (pilih)", "Ragu? (isi 'ya')", "Catatan (opsional)"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for kode, teks in daftar_item:
        ws.append([kode, bersihkan_sel(teks), None, None, None])
    n = len(daftar_item) + 1
    dv = DataValidation(type="list", formula1='"negatif,netral,positif"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Label tidak valid",
                        error="Pilih: negatif, netral, atau positif")
    ws.add_data_validation(dv)
    dv.add(f"C2:C{n}")
    dv2 = DataValidation(type="list", formula1='"ya"', allow_blank=True)
    ws.add_data_validation(dv2)
    dv2.add(f"D2:D{n}")
    for kol, lebar in zip("ABCDE", [8, 90, 16, 16, 34]):
        ws.column_dimensions[kol].width = lebar
    for row in ws.iter_rows(min_row=2, max_row=n, min_col=1, max_col=5):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    wp = wb.create_sheet("Panduan")
    wp.column_dimensions["A"].width = 120
    for baris in PANDUAN.split("\n"):
        wp.append([baris])
    wb.save(path)


def baca_penilai(path, nama):
    """Baca file penilai; validasi kode dan label. Mengembalikan DataFrame(kode, label, ragu, catatan)."""
    if not path.exists():
        sys.exit(f"File {nama} tidak ditemukan: {path}")
    d = pd.read_excel(path, sheet_name="Data", dtype=str)
    d.columns = ["kode", "teks", "label", "ragu", "catatan"][:len(d.columns)]
    d["label"] = d["label"].fillna("").str.strip().str.lower()
    kosong = d[d["label"] == ""]
    tidak_valid = d[(d["label"] != "") & (~d["label"].isin(LABEL_URUT))]
    if len(kosong) or len(tidak_valid):
        pesan = [f"[{nama}] file belum lengkap/valid: {len(kosong)} label kosong, "
                 f"{len(tidak_valid)} label tidak valid."]
        if len(kosong):
            pesan.append("Kode kosong (maks 15): " + ", ".join(kosong["kode"].head(15)))
        if len(tidak_valid):
            pesan.append("Kode tidak valid (maks 15): " + ", ".join(tidak_valid["kode"].head(15)))
        sys.exit("\n".join(pesan))
    d["ragu"] = d["ragu"].fillna("").str.strip().str.lower().eq("ya")
    d["catatan"] = d["catatan"].fillna("")
    return d[["kode", "label", "ragu", "catatan"]]


def kappa_dan_ci(a, b, bobot=None):
    """Cohen's kappa + selang kepercayaan 95% (bootstrap). a, b: array label teks."""
    a, b = np.asarray(a), np.asarray(b)
    k = cohen_kappa_score(a, b, labels=LABEL_URUT, weights=bobot)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(a), len(a))
        boot.append(cohen_kappa_score(a[i], b[i], labels=LABEL_URUT, weights=bobot))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return k, lo, hi


def interpretasi_kappa(k):
    """Skala Landis & Koch (1977); patokannya bersifat konvensi, bukan aturan mutlak."""
    if k < 0:
        return "lebih buruk dari kebetulan"
    for batas, nama in [(0.20, "sangat rendah"), (0.40, "rendah"), (0.60, "sedang"),
                        (0.80, "substansial")]:
        if k <= batas:
            return nama
    return "hampir sempurna"


def estimasi_proporsi(sampel, populasi, strategi, strata_dipakai=None):
    """Estimasi proporsi 'tak sesuai' pada POPULASI dari sampel (berstrata -> berbobot).
    Mengembalikan (p, se). Variansi memakai koreksi populasi hingga."""
    if strategi == "acak_sederhana":
        x = sampel["tak_sesuai"].astype(float)
        p = x.mean()
        return p, float(np.sqrt(p * (1 - p) / max(len(x) - 1, 1)))
    strata = strata_dipakai or list(populasi.keys())
    N = sum(populasi[s] for s in strata)
    p, var = 0.0, 0.0
    for s in strata:
        g = sampel[sampel["stratum"] == s]
        if len(g) == 0:
            continue
        n_s, N_s = len(g), populasi[s]
        p_s = g["tak_sesuai"].astype(float).mean()
        w = N_s / N
        p += w * p_s
        if n_s > 1:
            var += (w ** 2) * (1 - n_s / N_s) * p_s * (1 - p_s) / (n_s - 1)
    return p, float(np.sqrt(var))


def gambar_matriks(m, judul, sumbu_y, sumbu_x, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(3), LABEL_URUT)
    ax.set_yticks(range(3), LABEL_URUT)
    ax.set_xlabel(sumbu_x)
    ax.set_ylabel(sumbu_y)
    ax.set_title(judul, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Langkah 1: --buat
# --------------------------------------------------------------------------
def langkah_buat(strategi, timpa):
    if FILE_PARAM.exists() and not timpa:
        sys.exit(f"Pekerjaan validasi sudah ada ({FILE_PARAM.name}). Menjalankan --buat lagi akan "
                 f"menimpa file penilai dan hasil koreksi. Bila Anda yakin, tambahkan --timpa.")
    DIR_VALIDASI.mkdir(parents=True, exist_ok=True)
    df = muat_data()
    print(f"Data masuk: {len(df):,} baris")

    populasi = df.groupby("stratum").size().to_dict()
    print("\nLabel otomatis dari rating (populasi):")
    print(pd.crosstab(df["app"], df["label_otomatis"]).reindex(columns=LABEL_URUT).to_string())

    # --- sampel utama ---
    if strategi == "acak_sederhana":
        sampel = df.sample(n=UKURAN_ACAK_SEDERHANA, random_state=SEED)
    else:
        bagian = []
        for s, g in df.groupby("stratum"):
            n = min(UKURAN_PER_STRATUM, len(g))
            if n < UKURAN_PER_STRATUM:
                print(f"[PERHATIAN] stratum {s} hanya {len(g)} ulasan; seluruhnya diambil")
            bagian.append(g.sample(n=n, random_state=SEED))
        sampel = pd.concat(bagian)
    sampel = sampel.sample(frac=1, random_state=SEED).reset_index(drop=True)   # acak urutan
    sampel["kode"] = [f"V{i + 1:03d}" for i in range(len(sampel))]
    print(f"\nSampel validasi: {len(sampel)} ulasan (strategi: {strategi})")
    print(pd.crosstab(sampel["app"], sampel["label_otomatis"]).reindex(columns=LABEL_URUT).to_string())

    # --- kunci (rahasia) ---
    sampel[["kode", "reviewId", "app", "rating", "label_otomatis", "stratum"]] \
        .to_csv(FILE_KUNCI, index=False, encoding="utf-8")

    # --- file penilai: isi sama, urutan acak berbeda, tanpa rating/label otomatis ---
    for no, geser in [(1, 1), (2, 2)]:
        urut = sampel.sample(frac=1, random_state=SEED + geser)
        tulis_excel(FILE_PENILAI[no], list(zip(urut["kode"], urut["teks_asli"])))
        print(f"Dibuat: {FILE_PENILAI[no]}")

    # --- pilot kalibrasi (di luar sampel utama) ---
    sisa = df[~df["reviewId"].isin(sampel["reviewId"])]
    pilot = sisa.sample(n=UKURAN_PILOT, random_state=SEED)
    tulis_excel(FILE_PILOT, [(f"P{i + 1:02d}", t) for i, t in enumerate(pilot["teks_asli"])])
    print(f"Dibuat: {FILE_PILOT}")
    FILE_PANDUAN.write_text(PANDUAN, encoding="utf-8")

    FILE_PARAM.write_text(json.dumps({
        "waktu_buat": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "strategi": strategi, "seed": SEED,
        "ukuran_per_stratum": UKURAN_PER_STRATUM, "n_sampel": len(sampel),
        "n_populasi": len(df), "populasi_strata": {k: int(v) for k, v in populasi.items()},
    }, indent=2), encoding="utf-8")

    simpan_berlabel(df)
    print(f"\nData berlabel sementara (label_final = label dari rating): {FILE_KELUAR}")
    print("""
LANGKAH BERIKUTNYA
  1. Beri tiap penilai SATU file saja: validasi_penilai1.xlsx atau validasi_penilai2.xlsx
     (plus pilot_kalibrasi.xlsx untuk latihan). JANGAN berikan kunci_sampel.csv.
  2. Pilot dulu: kedua penilai mengisi pilot_kalibrasi.xlsx, lalu berdiskusi untuk menyamakan
     pemahaman panduan. Pilot TIDAK dihitung dalam kappa.
  3. Setelah itu, masing-masing mengisi file utamanya secara mandiri, tanpa berdiskusi.
  4. Letakkan kembali kedua file di folder yang sama, lalu jalankan: python tahap3_pelabelan.py --hitung""")


# --------------------------------------------------------------------------
# Langkah 3: --hitung
# --------------------------------------------------------------------------
def langkah_hitung(f1, f2):
    if not (FILE_KUNCI.exists() and FILE_PARAM.exists()):
        sys.exit("Belum ada sampel validasi. Jalankan dulu: python tahap3_pelabelan.py --buat")
    make_dirs()
    param = json.loads(FILE_PARAM.read_text(encoding="utf-8"))
    kunci = pd.read_csv(FILE_KUNCI, dtype={"reviewId": str})
    p1 = baca_penilai(f1, "penilai 1")
    p2 = baca_penilai(f2, "penilai 2")
    if set(p1["kode"]) != set(kunci["kode"]) or set(p2["kode"]) != set(kunci["kode"]):
        sys.exit("Kumpulan 'Kode' di file penilai tidak cocok dengan sampel. Jangan mengubah/menghapus "
                 "baris atau kolom Kode.")
    if p1["kode"].duplicated().any() or p2["kode"].duplicated().any():
        sys.exit("Ada Kode ganda di file penilai.")

    d = (kunci.merge(p1.rename(columns={"label": "p1", "ragu": "ragu1", "catatan": "cat1"}), on="kode")
              .merge(p2.rename(columns={"label": "p2", "ragu": "ragu2", "catatan": "cat2"}), on="kode"))
    n = len(d)
    pd.set_option("display.width", 200)

    # ---------- A. Kesepakatan antar-penilai ----------
    k, lo, hi = kappa_dan_ci(d["p1"], d["p2"])
    kw, lwo, lwh = kappa_dan_ci(d["p1"], d["p2"], bobot="linear")
    sepakat = float((d["p1"] == d["p2"]).mean())
    print(f"\n=== A. KESEPAKATAN ANTAR-PENILAI (n = {n}) ===")
    print(f"Kesepakatan teramati          : {sepakat:.3f}")
    print(f"Cohen's kappa                 : {k:.3f}  (95% CI bootstrap {lo:.3f} - {hi:.3f}) -> {interpretasi_kappa(k)}")
    print(f"Kappa berbobot linear (ordinal): {kw:.3f}  (95% CI {lwo:.3f} - {lwh:.3f})")
    print(f"Ulasan ditandai 'ragu': penilai 1 = {int(d['ragu1'].sum())}, penilai 2 = {int(d['ragu2'].sum())}")
    m12 = confusion_matrix(d["p1"], d["p2"], labels=LABEL_URUT)
    print("\nMatriks penilai 1 (baris) vs penilai 2 (kolom):")
    print(pd.DataFrame(m12, index=LABEL_URUT, columns=LABEL_URUT).to_string())

    # ---------- B. Kesesuaian tiap penilai dengan label dari rating ----------
    print("\n=== B. KESESUAIAN PENILAI DENGAN LABEL DARI RATING ===")
    for no in (1, 2):
        kk, a, b = kappa_dan_ci(d[f"p{no}"], d["label_otomatis"])
        print(f"Penilai {no} vs rating: kesesuaian {float((d[f'p{no}'] == d['label_otomatis']).mean()):.3f}, "
              f"kappa {kk:.3f} (95% CI {a:.3f} - {b:.3f})")

    # ---------- C. Sengketa antar-penilai ----------
    sengketa = d[d["p1"] != d["p2"]].copy()
    print(f"\n=== C. SENGKETA ANTAR-PENILAI: {len(sengketa)} dari {n} ulasan ===")
    teks_asli = pd.read_csv(FILE_KELUAR, usecols=["reviewId", "teks_asli"], dtype=str,
                            keep_default_na=False).set_index("reviewId")["teks_asli"]
    d["label_sepakat"] = np.where(d["p1"] == d["p2"], d["p1"], "")
    if len(sengketa):
        if not FILE_SENGKETA.exists():
            from openpyxl import Workbook
            from openpyxl.styles import Alignment, Font
            from openpyxl.worksheet.datavalidation import DataValidation
            wb = Workbook()
            ws = wb.active
            ws.title = "Sengketa"
            ws.append(["Kode", "Teks ulasan", "Label penilai 1", "Label penilai 2", "label_sepakat (pilih)"])
            for c in ws[1]:
                c.font = Font(bold=True)
            for _, r in sengketa.iterrows():
                ws.append([r["kode"], bersihkan_sel(teks_asli[r["reviewId"]]), r["p1"], r["p2"], None])
            dv = DataValidation(type="list", formula1='"negatif,netral,positif"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"E2:E{len(sengketa) + 1}")
            for kol, lebar in zip("ABCDE", [8, 90, 16, 16, 20]):
                ws.column_dimensions[kol].width = lebar
            for row in ws.iter_rows(min_row=2, max_row=len(sengketa) + 1):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "A2"
            wb.save(FILE_SENGKETA)
            print(f"Dibuat: {FILE_SENGKETA}\nKedua penilai berdiskusi, isi kolom 'label_sepakat' untuk setiap baris, "
                  f"simpan, lalu jalankan --hitung lagi.\n(Hasil kappa di atas tidak berubah oleh diskusi ini.)")
            _simpan_metrik_awal(d, n, k, lo, hi, kw, lwo, lwh, sepakat, len(sengketa))
            return
        sel = pd.read_excel(FILE_SENGKETA, sheet_name="Sengketa", dtype=str)
        sel.columns = ["kode", "teks", "a", "b", "label_sepakat"]
        sel["label_sepakat"] = sel["label_sepakat"].fillna("").str.strip().str.lower()
        if set(sel["kode"]) != set(sengketa["kode"]):
            sys.exit(f"Daftar sengketa di {FILE_SENGKETA.name} tidak cocok dengan file penilai saat ini. "
                     f"Bila file penilai diubah, hapus {FILE_SENGKETA.name} lalu jalankan --hitung lagi.")
        belum = sel[~sel["label_sepakat"].isin(LABEL_URUT)]
        if len(belum):
            sys.exit(f"{len(belum)} sengketa belum diisi/valid pada 'label_sepakat' "
                     f"(kode: {', '.join(belum['kode'].head(15))}).")
        d = d.merge(sel[["kode", "label_sepakat"]].rename(columns={"label_sepakat": "ls2"}), on="kode", how="left")
        d["label_sepakat"] = np.where(d["label_sepakat"] == "", d["ls2"], d["label_sepakat"])
        d = d.drop(columns="ls2")
    d["konsensus"] = d["label_sepakat"]

    # ---------- D. Konsensus vs label dari rating ----------
    d["tak_sesuai"] = d["konsensus"] != d["label_otomatis"]
    print(f"\n=== D. KONSENSUS vs LABEL DARI RATING (n = {n}) ===")
    mk = confusion_matrix(d["label_otomatis"], d["konsensus"], labels=LABEL_URUT)
    print("Baris = label dari rating, kolom = konsensus penilai:")
    print(pd.DataFrame(mk, index=LABEL_URUT, columns=LABEL_URUT).to_string())
    print(f"\nTak sesuai pada SAMPEL: {int(d['tak_sesuai'].sum())} dari {n} ({100 * d['tak_sesuai'].mean():.1f}%)")

    populasi = {k_: int(v) for k_, v in param["populasi_strata"].items()}
    strategi = param["strategi"]
    d["stratum"] = d["stratum"]
    p, se = estimasi_proporsi(d, populasi, strategi)
    ci = (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))
    print(f"ESTIMASI tingkat ketidaksesuaian pada POPULASI ({param['n_populasi']:,} ulasan): "
          f"{100 * p:.1f}% (95% CI {100 * ci[0]:.1f}% - {100 * ci[1]:.1f}%)"
          f"{'  [berbobot menurut strata]' if strategi == 'berstrata' else ''}")

    baris = []
    print("\nEstimasi ketidaksesuaian per label dari rating:")
    for lab in LABEL_URUT:
        strata = [s for s in populasi if s.endswith("|" + lab)]
        pp, ss = estimasi_proporsi(d, populasi, strategi, strata) if strategi == "berstrata" \
            else estimasi_proporsi(d[d["label_otomatis"] == lab], populasi, strategi)
        print(f"  {lab:8s}: {100 * pp:5.1f}%  (95% CI {100 * max(0, pp - 1.96 * ss):.1f}% - "
              f"{100 * min(1, pp + 1.96 * ss):.1f}%)")
        baris.append({"kelompok": f"label_rating={lab}", "estimasi": pp, "ci_bawah": max(0, pp - 1.96 * ss),
                      "ci_atas": min(1, pp + 1.96 * ss)})
    if strategi == "berstrata":
        print("\nEstimasi ketidaksesuaian per aplikasi:")
        for app_ in sorted({s.split("|")[0] for s in populasi}):
            strata = [s for s in populasi if s.startswith(app_ + "|")]
            pp, ss = estimasi_proporsi(d, populasi, strategi, strata)
            print(f"  {app_:6s}: {100 * pp:5.1f}%  (95% CI {100 * max(0, pp - 1.96 * ss):.1f}% - "
                  f"{100 * min(1, pp + 1.96 * ss):.1f}%)")
            baris.append({"kelompok": f"app={app_}", "estimasi": pp, "ci_bawah": max(0, pp - 1.96 * ss),
                          "ci_atas": min(1, pp + 1.96 * ss)})
    baris.insert(0, {"kelompok": "SEMUA", "estimasi": p, "ci_bawah": ci[0], "ci_atas": ci[1]})
    pd.DataFrame(baris).to_csv(TAB_DIR / "ketidaksesuaian_label.csv", index=False, encoding="utf-8")

    # ---------- E. Simpan semua keluaran ----------
    metrik = pd.DataFrame([
        ("n_validasi", n), ("kesepakatan_teramati", sepakat), ("kappa", k), ("kappa_ci_bawah", lo),
        ("kappa_ci_atas", hi), ("kappa_linear", kw), ("kappa_linear_ci_bawah", lwo),
        ("kappa_linear_ci_atas", lwh), ("sengketa_antar_penilai", len(sengketa)),
        ("ragu_penilai1", int(d["ragu1"].sum())), ("ragu_penilai2", int(d["ragu2"].sum())),
        ("tak_sesuai_sampel", float(d["tak_sesuai"].mean())), ("tak_sesuai_estimasi_populasi", p),
        ("tak_sesuai_ci_bawah", ci[0]), ("tak_sesuai_ci_atas", ci[1])], columns=["metrik", "nilai"])
    metrik.to_csv(TAB_DIR / "kappa_validasi.csv", index=False, encoding="utf-8")
    pd.DataFrame(m12, index=LABEL_URUT, columns=LABEL_URUT).to_csv(TAB_DIR / "matriks_penilai1_vs_penilai2.csv")
    pd.DataFrame(mk, index=LABEL_URUT, columns=LABEL_URUT).to_csv(TAB_DIR / "matriks_rating_vs_konsensus.csv")
    gambar_matriks(m12, "Penilai 1 vs Penilai 2", "Penilai 1", "Penilai 2", FIG_DIR / "validasi_penilai1_vs_2.png")
    gambar_matriks(mk, "Label dari rating vs konsensus", "Label dari rating", "Konsensus penilai",
                   FIG_DIR / "validasi_rating_vs_konsensus.png")

    contoh = d[d["tak_sesuai"]].copy()
    contoh["teks_asli"] = contoh["reviewId"].map(teks_asli)
    contoh[["kode", "app", "rating", "label_otomatis", "konsensus", "ragu1", "ragu2", "cat1", "cat2", "teks_asli"]] \
        .to_csv(TAB_DIR / "contoh_ketidaksesuaian.csv", index=False, encoding="utf-8")

    df = muat_data()
    simpan_berlabel(df, d.assign(label_final=d["konsensus"])[["reviewId", "label_final"]])
    akhir = pd.read_csv(FILE_KELUAR, keep_default_na=False)
    print(f"\n=== E. HASIL AKHIR ===\nTersimpan: {FILE_KELUAR}")
    print(f"Label dikoreksi: {int(akhir['dikoreksi'].sum())} ulasan | tervalidasi manual: {int(akhir['tervalidasi'].sum())}")
    print("\nDistribusi label_final:")
    print(pd.crosstab(akhir["app"], akhir["label_final"]).reindex(columns=LABEL_URUT).to_string())
    print(f"\nTabel metrik   : {TAB_DIR / 'kappa_validasi.csv'}")
    print(f"Contoh tak sesuai (untuk analisis sarkasme dsb.): {TAB_DIR / 'contoh_ketidaksesuaian.csv'}")
    print("\n5 contoh ketidaksesuaian:")
    print(contoh[["app", "rating", "label_otomatis", "konsensus", "teks_asli"]].head(5)
          .to_string(index=False, max_colwidth=70))


def _simpan_metrik_awal(d, n, k, lo, hi, kw, lwo, lwh, sepakat, n_sengketa):
    pd.DataFrame([("n_validasi", n), ("kesepakatan_teramati", sepakat), ("kappa", k),
                  ("kappa_ci_bawah", lo), ("kappa_ci_atas", hi), ("kappa_linear", kw),
                  ("kappa_linear_ci_bawah", lwo), ("kappa_linear_ci_atas", lwh),
                  ("sengketa_antar_penilai", n_sengketa)], columns=["metrik", "nilai"]) \
        .to_csv(TAB_DIR / "kappa_validasi.csv", index=False, encoding="utf-8")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Tahap 3: pelabelan, validasi manual, kappa")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--buat", action="store_true", help="label otomatis + sampel validasi + template penilai")
    g.add_argument("--hitung", action="store_true", help="hitung kappa, rekonsiliasi, koreksi label")
    ap.add_argument("--strategi", choices=["berstrata", "acak_sederhana"], default="berstrata")
    ap.add_argument("--timpa", action="store_true", help="izinkan --buat menimpa pekerjaan yang ada")
    ap.add_argument("--file1", type=Path, default=FILE_PENILAI[1])
    ap.add_argument("--file2", type=Path, default=FILE_PENILAI[2])
    args = ap.parse_args()
    args.file1 = args.file1 if args.file1.is_absolute() else CWD_AWAL / args.file1
    args.file2 = args.file2 if args.file2.is_absolute() else CWD_AWAL / args.file2
    make_dirs()
    set_seed(SEED)
    if args.buat:
        langkah_buat(args.strategi, args.timpa)
    else:
        langkah_hitung(args.file1, args.file2)


if __name__ == "__main__":
    main()
