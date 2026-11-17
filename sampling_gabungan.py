"""
sampling_gabungan.py  -  Membuat data gabungan dari data mentah, hanya dari ulasan
yang cukup panjang (minimal "satu kalimat"), lalu mengambil sampel acak per aplikasi.

Letakkan file ini di sebelah folder penelitian-ojol/ (satu folder dengan setup_proyek.py)
atau di dalam folder penelitian-ojol/.

Cara pakai:
    python sampling_gabungan.py --simulasi      # LIHAT dulu dampak filter, tanpa menyimpan apa pun
    python sampling_gabungan.py                  # buat data/interim/ulasan_gabungan.csv

Opsi:
    --min-kata 3            minimal jumlah kata (huruf) agar ulasan dianggap "satu kalimat" (default 3)
    --target 10000          jumlah ulasan per aplikasi (default: TARGET_PER_APP di config.py)
    --strategi terbaru      ambil yang terbaru, bukan sampel acak dari seluruh periode
    --izinkan-duplikat      jangan buang ulasan dengan teks yang sama persis
    --simulasi              hanya tampilkan tabel, tidak menulis file

Urutan penyaringan (per aplikasi):
    data mentah -> dalam periode (START_DATE..END_DATE) -> teks tidak kosong
    -> jumlah kata >= --min-kata -> buang duplikat teks -> sampel acak (seed tetap)

CATATAN: Play Store tidak menandai kalimat, dan ulasan informal sering tanpa tanda baca.
"Satu kalimat" di sini didekati dengan jumlah kata (hanya huruf; angka, emoji, URL, dan
tanda baca tidak dihitung). Ini proksi, bukan deteksi kalimat yang sebenarnya.

Output:
    data/interim/ulasan_gabungan.csv          (dibaca Tahap 2; versi lama dicadangkan otomatis)
    outputs/tables/ringkasan_sampling.csv     (jumlah ulasan di tiap langkah penyaringan)
    outputs/tables/parameter_sampling.txt     (parameter untuk bagian Metode)
"""
import argparse
import os
import re
import shutil
import sys
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
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")

from config import *            # noqa: F401,F403  (SEED, APPS, RAW_DIR, dst.)
from utils import set_seed, make_dirs

# Pola: URL dibuang dulu; kata = deretan huruf (unicode), angka dan underscore tidak dihitung
POLA_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
POLA_KATA = re.compile(r"[^\W\d_]+", re.UNICODE)

BATAS_UJI_KATA = [1, 2, 3, 4, 5, 6, 8, 10]    # untuk tabel sensitivitas


def daftar_kata(teks):
    """Daftar kata (huruf saja) dari sebuah ulasan; URL tidak dihitung."""
    if not isinstance(teks, str):
        return []
    return POLA_KATA.findall(POLA_URL.sub(" ", teks))


def hitung_kata(teks):
    return len(daftar_kata(teks))


def kunci_duplikat(teks):
    """Teks dinormalkan (huruf kecil, hanya kata) untuk mendeteksi duplikat yang sama persis."""
    return " ".join(daftar_kata(teks.lower()))


def label_tiga_kelas(rating):
    return rating.map(rating_ke_label)


def proporsi_label(df):
    """Proporsi negatif/netral/positif per aplikasi (dari rating)."""
    lab = label_tiga_kelas(df["score"])
    t = pd.crosstab(df["app"], lab, normalize="index")
    for k in ["negatif", "netral", "positif"]:
        if k not in t.columns:
            t[k] = 0.0
    return t[["negatif", "netral", "positif"]].round(3)


def muat_mentah():
    bagian = []
    for nama in APPS:
        f = RAW_DIR / f"{nama.lower()}_raw.csv"
        if not f.exists():
            sys.exit(f"File {f} tidak ada. Jalankan scraping (Tahap 1) lebih dulu.")
        bagian.append(pd.read_csv(f, parse_dates=["at"]))
    return pd.concat(bagian, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Sampling data gabungan (minimal satu kalimat)")
    ap.add_argument("--min-kata", type=int, default=3,
                    help="minimal jumlah kata agar dianggap satu kalimat (default 3)")
    ap.add_argument("--target", type=int, default=TARGET_PER_APP, help="jumlah ulasan per aplikasi")
    ap.add_argument("--strategi", choices=["acak_periode", "terbaru"], default="acak_periode")
    ap.add_argument("--izinkan-duplikat", action="store_true", help="jangan buang duplikat teks")
    ap.add_argument("--simulasi", action="store_true", help="hanya tampilkan tabel, tidak menyimpan")
    args = ap.parse_args()

    make_dirs()
    set_seed(SEED)
    pd.set_option("display.width", 200)

    tgl_mulai = pd.Timestamp(START_DATE)
    tgl_akhir = pd.Timestamp(END_DATE) + pd.Timedelta(days=1)   # END_DATE ikut dihitung

    print(f"Proyek : {PROYEK}")
    print(f"Periode: {START_DATE} s.d. {END_DATE} | min kata: {args.min_kata} | "
          f"target/app: {args.target:,} | strategi: {args.strategi} | "
          f"duplikat teks: {'DIIZINKAN' if args.izinkan_duplikat else 'dibuang'}\n")

    # ---------- Langkah 1-3: muat, periode, tidak kosong ----------
    mentah = muat_mentah().drop_duplicates("reviewId")
    n_mentah = mentah.groupby("app").size()

    dlm_periode = mentah[(mentah["at"] >= tgl_mulai) & (mentah["at"] < tgl_akhir)]
    n_periode = dlm_periode.groupby("app").size()

    tidak_kosong = dlm_periode[dlm_periode["content"].fillna("").str.strip() != ""].copy()
    n_tidak_kosong = tidak_kosong.groupby("app").size()

    # ---------- Langkah 4: hitung kata ----------
    tidak_kosong["jumlah_kata"] = tidak_kosong["content"].map(hitung_kata)

    # Tabel sensitivitas: berapa ulasan tersisa bila syarat minimal kata dinaikkan
    sens = {}
    for n in BATAS_UJI_KATA:
        sens[f">= {n} kata"] = (tidak_kosong[tidak_kosong["jumlah_kata"] >= n]
                                .groupby("app").size())
    sens = pd.DataFrame(sens).reindex(list(APPS.keys())).fillna(0).astype(int)
    print("=== SENSITIVITAS: jumlah ulasan tersisa menurut syarat minimal kata ===")
    print(sens.to_string())
    print("\nPersentase terhadap ulasan tidak kosong:")
    print((sens.div(n_tidak_kosong.reindex(sens.index), axis=0) * 100).round(1).to_string())

    # ---------- Langkah 5: filter minimal kata ----------
    lolos = tidak_kosong[tidak_kosong["jumlah_kata"] >= args.min_kata].copy()
    n_lolos = lolos.groupby("app").size()

    print(f"\n=== PROPORSI LABEL (dari rating) SEBELUM vs SESUDAH filter >= {args.min_kata} kata ===")
    pb, ps = proporsi_label(tidak_kosong), proporsi_label(lolos)
    cmp_ = pb.join(ps, lsuffix="_sebelum", rsuffix="_sesudah")
    print(cmp_.to_string())
    print("(Bandingkan kolom sebelum dan sesudah: bila proporsi bergeser, filter panjang mengubah "
          "komposisi sentimen dan perlu dilaporkan di Metode.)")

    # ---------- Langkah 6: duplikat teks ----------
    if args.izinkan_duplikat:
        unik = lolos
    else:
        lolos = lolos.sort_values("at")
        lolos["_kunci"] = lolos["content"].map(kunci_duplikat)
        unik = lolos.drop_duplicates("_kunci", keep="first").drop(columns="_kunci")
    n_unik = unik.groupby("app").size()

    # ---------- Langkah 7: sampling ----------
    hasil = []
    for nama in APPS:
        d = unik[unik["app"] == nama]
        if len(d) <= args.target:
            s = d
            print(f"[PERHATIAN] {nama}: hanya {len(d):,} ulasan memenuhi syarat "
                  f"(<= target {args.target:,}), seluruhnya dipakai")
        elif args.strategi == "terbaru":
            s = d.sort_values("at", ascending=False).head(args.target)
        else:
            s = d.sample(n=args.target, random_state=SEED)
        hasil.append(s)
    gabungan = (pd.concat(hasil)
                  .rename(columns={"content": "teks", "score": "rating",
                                   "at": "tanggal", "thumbsUpCount": "thumbs_up"})
                  [["app", "reviewId", "teks", "rating", "tanggal", "thumbs_up", "jumlah_kata"]]
                  .sort_values(["app", "tanggal"])
                  .reset_index(drop=True))
    n_terpilih = gabungan.groupby("app").size()

    # ---------- Ringkasan ----------
    ringkas = pd.DataFrame({
        "mentah": n_mentah, "dalam_periode": n_periode, "teks_tidak_kosong": n_tidak_kosong,
        f"min_{args.min_kata}_kata": n_lolos, "setelah_buang_duplikat": n_unik,
        "terpilih": n_terpilih,
    }).reindex(list(APPS.keys())).fillna(0).astype(int)
    print("\n=== RINGKASAN PENYARINGAN ===")
    print(ringkas.to_string())

    print("\n=== PERIKSA SECARA VISUAL ===")
    print("Contoh ulasan yang DIBUANG karena kurang dari "
          f"{args.min_kata} kata (acak, 8 baris):")
    dibuang = tidak_kosong[tidak_kosong["jumlah_kata"] < args.min_kata]
    if len(dibuang):
        print(dibuang.sample(min(8, len(dibuang)), random_state=SEED)
              [["app", "content", "score"]].to_string(index=False, max_colwidth=50))
    print("\nUlasan TERPENDEK yang masih terpilih (5 baris):")
    print(gabungan.nsmallest(5, "jumlah_kata")[["app", "teks", "rating", "jumlah_kata"]]
          .to_string(index=False, max_colwidth=50))
    print("\nStatistik jumlah kata ulasan terpilih:")
    print(gabungan["jumlah_kata"].describe().round(1).to_string())
    print("\nJumlah ulasan terpilih per aplikasi per tahun:")
    print(pd.crosstab(gabungan["app"], gabungan["tanggal"].dt.year))
    print("\nProporsi label (dari rating) pada data terpilih:")
    lab = gabungan["rating"].map(rating_ke_label)
    print(pd.crosstab(gabungan["app"], lab, normalize="index").round(3).to_string())
    print("\nJumlah per label (semua aplikasi):")
    print(lab.value_counts().to_string())

    if args.simulasi:
        print("\n--simulasi: TIDAK ada file yang ditulis. Jika tabel sudah sesuai, jalankan tanpa --simulasi.")
        return

    # ---------- Simpan ----------
    tujuan = INTERIM_DIR / "ulasan_gabungan.csv"
    if tujuan.exists():
        cadangan = INTERIM_DIR / f"ulasan_gabungan_cadangan_{datetime.now():%Y%m%d_%H%M%S}.csv"
        shutil.copy2(tujuan, cadangan)
        print(f"\nVersi lama dicadangkan ke: {cadangan.name}")
    gabungan.to_csv(tujuan, index=False, encoding="utf-8")
    ringkas.to_csv(TAB_DIR / "ringkasan_sampling.csv", encoding="utf-8")
    (TAB_DIR / "parameter_sampling.txt").write_text(
        f"waktu_sampling: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"periode: {START_DATE} s.d. {END_DATE}\n"
        f"min_kata: {args.min_kata}\n"
        f"target_per_aplikasi: {args.target}\n"
        f"strategi: {args.strategi}\n"
        f"buang_duplikat_teks: {not args.izinkan_duplikat}\n"
        f"seed: {SEED}\n", encoding="utf-8")
    print(f"Tersimpan di: {tujuan}")
    print(f"Ringkasan   : {TAB_DIR / 'ringkasan_sampling.csv'}")


if __name__ == "__main__":
    main()
