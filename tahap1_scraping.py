"""
tahap1_scraping.py  -  Tahap 1: scraping ulasan Gojek, Grab, Maxim (sekali jalan).

Letakkan file ini di SALAH SATU tempat berikut:
  - di sebelah folder penelitian-ojol/ (satu folder dengan setup_proyek.py), atau
  - di dalam folder penelitian-ojol/ (sejajar dengan config.py)

Cara pakai:
    python tahap1_scraping.py --uji          # uji coba kecil (400 ulasan/aplikasi), TANPA sampling
    python tahap1_scraping.py                # scraping sebenarnya + ringkasan + sampling + gabungan

Opsi:
    --app Gojek --app Maxim     hanya aplikasi tertentu (boleh diulang)
    --maks 80000                batas pengaman ulasan yang diambil per aplikasi (default 50000)
    --strategi terbaru          ambil N ulasan terbaru (default: acak_periode)
    --jeda 2.0                  jeda antarpermintaan dalam detik (default: SLEEP_SECONDS di config.py)
    --lewati-sampling           hanya scraping + ringkasan, tidak membuat ulasan_gabungan.csv

Aman dihentikan (Ctrl+C) kapan saja: jalankan lagi dengan perintah yang sama untuk melanjutkan.

Output:
    data/raw/<app>_raw.csv, <app>_selesai.txt     (data mentah, jangan diubah lagi)
    outputs/tables/ringkasan_scraping.csv
    data/interim/ulasan_gabungan.csv              (dibaca Tahap 2)
"""
import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Temukan folder proyek, pindah ke dalamnya, lalu impor config/utils
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROYEK = HERE if (HERE / "config.py").exists() else HERE / "penelitian-ojol"
if not (PROYEK / "config.py").exists():
    sys.exit("config.py tidak ditemukan. Jalankan dulu: python setup_proyek.py, "
             "lalu letakkan skrip ini di sebelah folder penelitian-ojol/ atau di dalamnya.")
os.chdir(PROYEK)
sys.path.insert(0, str(PROYEK))

try:
    import pandas as pd
    from tqdm import tqdm
    from google_play_scraper import reviews, Sort
    from google_play_scraper.exceptions import NotFoundError
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")

from config import *            # noqa: F401,F403  (SEED, APPS, RAW_DIR, dst.)
from utils import set_seed, make_dirs

KOLOM_SIMPAN = ["reviewId", "content", "score", "at", "thumbsUpCount"]


# --------------------------------------------------------------------------
# Scraping
# --------------------------------------------------------------------------
def path_app(raw_dir: Path, nama: str):
    """Tiga file per aplikasi: data, token paginasi (sementara), penanda selesai."""
    k = nama.lower()
    return (raw_dir / f"{k}_raw.csv",
            raw_dir / f"{k}_token.pkl",
            raw_dir / f"{k}_selesai.txt")


def ambil_batch(pkg, token, jeda):
    """Satu permintaan dengan retry + exponential backoff.
    Mengembalikan (daftar_ulasan, token_baru); (None, token) bila gagal total."""
    for percobaan in range(1, MAX_RETRY + 1):
        try:
            if token is None:
                return reviews(pkg, lang=LANG, country=COUNTRY,
                               sort=Sort.NEWEST, count=BATCH_SIZE)
            return reviews(pkg, continuation_token=token)   # pengaturan lain dibawa oleh token
        except NotFoundError:
            raise                                           # package ID salah: tidak perlu diulang
        except Exception as e:
            tunggu = max(jeda, 1.0) * (2 ** percobaan)
            print(f"  [!] percobaan {percobaan}/{MAX_RETRY} gagal "
                  f"({type(e).__name__}); menunggu {tunggu:.0f} dtk")
            time.sleep(tunggu)
    return None, token


def scrape_app(nama, pkg, raw_dir, maks, tgl_mulai, tgl_akhir, jeda):
    """Scrape satu aplikasi. Mengembalikan True bila selesai (sudah/baru), False bila belum."""
    f_csv, f_token, f_selesai = path_app(raw_dir, nama)
    if f_selesai.exists():
        print(f"[{nama}] sudah selesai sebelumnya, dilewati.")
        return True

    # --- Lanjutkan dari progres sebelumnya bila ada ---
    token, id_ada = None, set()
    if f_csv.exists():
        id_ada = set(pd.read_csv(f_csv, usecols=["reviewId"])["reviewId"])
    if f_token.exists():
        try:
            with open(f_token, "rb") as f:
                token = pickle.load(f)
        except Exception:
            print(f"[{nama}] file token tidak terbaca, mengulang dari ulasan terbaru "
                  f"(duplikat otomatis disaring).")
            token = None
    if id_ada:
        print(f"[{nama}] melanjutkan: {len(id_ada):,} ulasan sudah tersimpan")

    jumlah = len(id_ada)
    tulis_header = not f_csv.exists()
    alasan = None
    pbar = tqdm(total=maks, initial=min(jumlah, maks), desc=nama, unit="ulasan")

    try:
        while jumlah < maks:
            hasil, token_baru = ambil_batch(pkg, token, jeda)

            if hasil is None:   # gagal total: progres aman, tinggal jalankan ulang
                pbar.close()
                print(f"[{nama}] gagal setelah {MAX_RETRY} percobaan. "
                      f"Progres tersimpan ({jumlah:,} ulasan). Jalankan ulang skrip nanti.")
                return False
            if not hasil:
                alasan = "tidak ada ulasan lagi"
                break

            df = pd.DataFrame(hasil)[KOLOM_SIMPAN]
            tgl_terlama = df["at"].min()

            # Saring: hanya dalam periode, dan belum pernah tersimpan
            df = df[(df["at"] >= tgl_mulai) & (df["at"] < tgl_akhir)]
            df = df[~df["reviewId"].isin(id_ada)]

            if len(df) > 0:
                df.insert(0, "app", nama)
                df.to_csv(f_csv, mode="a", header=tulis_header, index=False, encoding="utf-8")
                tulis_header = False
                id_ada.update(df["reviewId"])
                jumlah += len(df)
                pbar.update(len(df))

            token = token_baru
            with open(f_token, "wb") as f:      # simpan token SETELAH data tersimpan
                pickle.dump(token, f)

            if tgl_terlama < tgl_mulai:
                alasan = f"mencapai tanggal mulai ({START_DATE})"
                break
            if token.token is None:
                alasan = "token paginasi habis"
                break
            time.sleep(jeda)
    except KeyboardInterrupt:
        pbar.close()
        print(f"\n[{nama}] dihentikan manual. Progres tersimpan ({jumlah:,} ulasan). "
              f"Jalankan ulang untuk melanjutkan.")
        raise
    pbar.close()

    if alasan is None:
        alasan = ("batas --maks tercapai "
                  "(periode mungkin TIDAK terliput penuh, cek tanggal terlama)")

    f_selesai.write_text(f"selesai: {datetime.now():%Y-%m-%d %H:%M:%S} | alasan: {alasan} | "
                         f"jumlah: {jumlah}", encoding="utf-8")
    if f_token.exists():
        f_token.unlink()
    print(f"[{nama}] selesai: {jumlah:,} ulasan | {alasan}")
    return True


# --------------------------------------------------------------------------
# Ringkasan
# --------------------------------------------------------------------------
def ringkasan(raw_dir, simpan: bool):
    ringkas, semua = [], []
    for nama in APPS:
        f_csv, _, f_selesai = path_app(raw_dir, nama)
        if not f_csv.exists():
            print(f"[{nama}] belum ada data.")
            continue
        d = pd.read_csv(f_csv, parse_dates=["at"])
        semua.append(d)
        ringkas.append({
            "app": nama,
            "jumlah": len(d),
            "tanggal_terlama": d["at"].min().date(),
            "tanggal_terbaru": d["at"].max().date(),
            "reviewId_duplikat": int(d["reviewId"].duplicated().sum()),
            "teks_kosong": int(d["content"].isna().sum()),
            "status": f_selesai.read_text(encoding="utf-8") if f_selesai.exists() else "BELUM SELESAI",
        })
    if not ringkas:
        print("Belum ada data untuk diringkas.")
        return None

    tabel = pd.DataFrame(ringkas)
    print("\n=== RINGKASAN DATA MENTAH ===")
    print(tabel.drop(columns="status").to_string(index=False))
    print("\nStatus:")
    for _, r in tabel.iterrows():
        print(f"- {r['app']}: {r['status']}")
    if simpan:
        tabel.to_csv(TAB_DIR / "ringkasan_scraping.csv", index=False, encoding="utf-8")

    mentah = pd.concat(semua, ignore_index=True)
    print("\nDistribusi rating (proporsi per aplikasi):")
    print(pd.crosstab(mentah["app"], mentah["score"], normalize="index").round(3))
    print("\nContoh data:")
    print(mentah.sample(min(5, len(mentah)), random_state=SEED)
          [["app", "content", "score", "at", "thumbsUpCount"]]
          .to_string(index=False, max_colwidth=60))
    return tabel


# --------------------------------------------------------------------------
# Sampling + penggabungan
# --------------------------------------------------------------------------
def buat_gabungan(raw_dir, strategi):
    semua = []
    for nama in APPS:
        f_csv, _, _ = path_app(raw_dir, nama)
        semua.append(pd.read_csv(f_csv, parse_dates=["at"]))
    semua = pd.concat(semua, ignore_index=True)

    hasil = []
    for nama, d in semua.groupby("app"):
        d = d.drop_duplicates("reviewId")
        if len(d) <= TARGET_PER_APP:
            s = d
            print(f"{nama}: hanya {len(d):,} ulasan (<= target {TARGET_PER_APP:,}), seluruhnya dipakai")
        elif strategi == "terbaru":
            s = d.sort_values("at", ascending=False).head(TARGET_PER_APP)
        else:   # sampel acak sederhana dari seluruh periode
            s = d.sample(n=TARGET_PER_APP, random_state=SEED)
        hasil.append(s)

    gabungan = (pd.concat(hasil)
                  .rename(columns={"content": "teks", "score": "rating",
                                   "at": "tanggal", "thumbsUpCount": "thumbs_up"})
                  .sort_values(["app", "tanggal"])
                  .reset_index(drop=True))
    tujuan = INTERIM_DIR / "ulasan_gabungan.csv"
    gabungan.to_csv(tujuan, index=False, encoding="utf-8")

    print("\nJumlah ulasan akhir per aplikasi:")
    print(gabungan["app"].value_counts().to_string())
    print("\nJumlah ulasan per aplikasi per tahun:")
    print(pd.crosstab(gabungan["app"], gabungan["tanggal"].dt.year))
    print("\nTersimpan di:", tujuan)


# --------------------------------------------------------------------------
# Program utama
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Tahap 1: scraping ulasan ojol (sekali jalan)")
    ap.add_argument("--uji", action="store_true", help="mode uji coba (400 ulasan/aplikasi, tanpa sampling)")
    ap.add_argument("--app", action="append", choices=list(APPS.keys()),
                    help="hanya aplikasi tertentu (boleh diulang)")
    ap.add_argument("--maks", type=int, default=50_000, help="batas pengaman ulasan per aplikasi")
    ap.add_argument("--strategi", choices=["acak_periode", "terbaru"], default="acak_periode")
    ap.add_argument("--jeda", type=float, default=SLEEP_SECONDS, help="jeda antarpermintaan (detik)")
    ap.add_argument("--lewati-sampling", action="store_true", help="tanpa membuat ulasan_gabungan.csv")
    args = ap.parse_args()

    make_dirs()
    set_seed(SEED)

    maks = 400 if args.uji else args.maks
    raw_dir = RAW_DIR / "_uji" if args.uji else RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    daftar = args.app or list(APPS.keys())

    tgl_mulai = pd.Timestamp(START_DATE)
    tgl_akhir = pd.Timestamp(END_DATE) + pd.Timedelta(days=1)   # END_DATE ikut dihitung

    print(f"Proyek : {PROYEK}")
    print(f"Mode   : {'UJI COBA' if args.uji else 'SEBENARNYA'} | folder mentah: {raw_dir}")
    print(f"Periode: {START_DATE} s.d. {END_DATE} | batas per aplikasi: {maks:,} | jeda {args.jeda}s\n")

    try:
        for nama in daftar:
            try:
                scrape_app(nama, APPS[nama], raw_dir, maks, tgl_mulai, tgl_akhir, args.jeda)
            except NotFoundError:
                print(f"[{nama}] package ID '{APPS[nama]}' tidak ditemukan. Perbaiki APPS di config.py.")
    except KeyboardInterrupt:
        sys.exit("\nDihentikan. Jalankan ulang perintah yang sama untuk melanjutkan.")

    ringkasan(raw_dir, simpan=not args.uji)

    # --- Sampling hanya bila mode sebenarnya dan SEMUA aplikasi sudah selesai ---
    if args.uji:
        print("\nMode uji: sampling dilewati. Jika hasil di atas wajar, jalankan tanpa --uji.")
        return
    if args.lewati_sampling:
        print("\n--lewati-sampling: ulasan_gabungan.csv tidak dibuat.")
        return
    belum = [n for n in APPS if not path_app(raw_dir, n)[2].exists()]
    if belum:
        print(f"\nSampling ditunda: aplikasi belum selesai -> {', '.join(belum)}. "
              f"Jalankan ulang skrip ini (progres dilanjutkan) sampai semuanya selesai.")
        return
    print("\n=== SAMPLING & PENGGABUNGAN ===")
    buat_gabungan(raw_dir, args.strategi)


if __name__ == "__main__":
    main()
