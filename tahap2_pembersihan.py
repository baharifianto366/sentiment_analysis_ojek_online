"""
tahap2_pembersihan.py  -  Tahap 2: pembersihan dan pra-pemrosesan teks ulasan.

Letakkan di sebelah folder penelitian-ojol/ (satu folder dengan setup_proyek.py) atau di
dalam folder penelitian-ojol/. Jalankan:

    python tahap2_pembersihan.py

Bisa juga ditempel per blok (ditandai '# %%') ke Jupyter/Colab; jalankan berurutan dari Blok 0.

Input : data/interim/ulasan_gabungan.csv   (hasil sampling_gabungan.py, harus punya kolom jumlah_kata)
Output: data/processed/ulasan_bersih.csv
        outputs/tables/ringkasan_praproses.csv, contoh_praproses.csv,
                       slang_terganti.csv, frekuensi_kata_top500.csv

Tiga versi teks dihasilkan agar tiap kelompok model memakai pra-pemrosesan yang sesuai:
    teks_ringan : untuk IndoBERT (tanda baca dan angka dipertahankan; tanpa stopword/stemming)
    teks_bersih : untuk TF-IDF/SVM/NB dan Word2Vec/FastText (huruf a-z saja + normalisasi slang)
    teks_topik  : untuk LDA (teks_bersih tanpa stopword; stemming opsional)
"""

# %% BLOK 0: Impor dan lokasi proyek
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

try:                                     # cegah error cetak emoji/aksara non-Latin di konsol Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    HERE = Path(__file__).resolve().parent
except NameError:                       # bila ditempel di Jupyter/Colab
    HERE = Path.cwd()
PROYEK = HERE if (HERE / "config.py").exists() else HERE / "penelitian-ojol"
if not (PROYEK / "config.py").exists():
    sys.exit("config.py tidak ditemukan. Jalankan dulu setup_proyek.py, lalu letakkan skrip ini "
             "di sebelah folder penelitian-ojol/ atau di dalamnya.")
os.chdir(PROYEK)
sys.path.insert(0, str(PROYEK))

try:
    import pandas as pd
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")

from config import *            # noqa: F401,F403  (SEED, BASE_DIR, INTERIM_DIR, PROC_DIR, TAB_DIR, ...)
from utils import set_seed, make_dirs

make_dirs()
set_seed(SEED)

# %% BLOK 1: Konfigurasi tahap 2 (ubah di sini)
FILE_MASUK  = INTERIM_DIR / "ulasan_gabungan.csv"
FILE_KELUAR = PROC_DIR / "ulasan_bersih.csv"
# Kamus slang tambahan (opsional): CSV dengan header 'slang,baku'. Entri bawaan di bawah
# diprioritaskan bila bentrok, karena sudah disesuaikan dengan konteks ulasan ojol.
FILE_KAMUS_EKSTERNAL = BASE_DIR / "data" / "kamus" / "slang.csv"

MIN_KATA_BERSIH = 3                # ulasan dengan kata kurang dari ini setelah dibersihkan dibuang
NORMALISASI_SLANG_BERSIH = True    # normalisasi slang pada teks_bersih (TF-IDF, W2V, LDA)
NORMALISASI_SLANG_RINGAN = False   # normalisasi slang pada teks_ringan (IndoBERT): disarankan False
LAKUKAN_STEMMING = False           # stemming Sastrawi pada teks_topik (lambat; default mati)
PERTAHANKAN_NEGASI_DI_TOPIK = False  # True = kata 'tidak', 'belum', dst. tidak dibuang dari teks_topik
STOPWORD_TAMBAHAN = {"aplikasi", "apk", "app", "gojek", "grab", "maxim",
                     "gojeknya", "grabnya", "maximnya", "aplikasinya", "nya"}
NEGASI = {"tidak", "bukan", "belum", "tanpa", "jangan", "kurang", "tak"}
UKURAN_CONTOH = 25

# %% BLOK 2: Kamus slang dan stopword
KAMUS_BAWAAN = {
    # --- negasi ---
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak", "nggk": "tidak",
    "enggak": "tidak", "engga": "tidak", "ngak": "tidak", "tdk": "tidak", "tak": "tidak", "gx": "tidak",
    "kagak": "tidak", "gaada": "tidak ada", "gabisa": "tidak bisa", "gamau": "tidak mau",
    "gabole": "tidak boleh", "bkn": "bukan", "blm": "belum", "blum": "belum", "jgn": "jangan",
    "krg": "kurang",
    # --- pronomina ---
    "sy": "saya", "gw": "saya", "gue": "saya", "gua": "saya", "aq": "aku", "lu": "kamu", "lo": "kamu",
    # --- konjungsi, preposisi, partikel ---
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "untk": "untuk", "krn": "karena",
    "karna": "karena", "krna": "karena", "jd": "jadi", "jg": "juga", "tp": "tapi", "tpi": "tapi",
    "sm": "sama", "dr": "dari", "dlm": "dalam", "pd": "pada", "sbg": "sebagai", "spy": "supaya",
    "sblm": "sebelum", "stlh": "setelah", "klo": "kalau", "kalo": "kalau", "pdhl": "padahal",
    "pdhal": "padahal", "gmn": "bagaimana", "gimana": "bagaimana", "gmna": "bagaimana",
    "bgmn": "bagaimana", "knp": "kenapa", "kpn": "kapan", "emg": "memang", "emang": "memang",
    "gitu": "begitu", "gini": "begini", "kyk": "seperti", "kayak": "seperti", "mulu": "melulu",
    # --- kata kerja, keterangan, lainnya ---
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "sudh": "sudah", "dah": "sudah",
    "bs": "bisa", "bsa": "bisa", "hrs": "harus", "lg": "lagi", "lbh": "lebih", "trs": "terus",
    "trus": "terus", "aja": "saja", "aj": "saja", "ajah": "saja", "doang": "saja",
    "bgt": "banget", "bngt": "banget", "bgtt": "banget", "bener": "benar", "bnr": "benar",
    "dpt": "dapat", "dapet": "dapat", "org": "orang", "brp": "berapa", "skrg": "sekarang",
    "skrng": "sekarang", "kmrn": "kemarin", "cm": "cuma", "cuman": "cuma", "pake": "pakai",
    "msh": "masih", "masi": "masih", "slalu": "selalu", "sllu": "selalu", "tdi": "tadi",
    "bnyk": "banyak", "sampe": "sampai", "smpe": "sampai", "smpai": "sampai", "nyampe": "sampai",
    "tau": "tahu", "smua": "semua", "sgt": "sangat", "kasi": "kasih", "wkt": "waktu",
    "lgsg": "langsung", "lngsg": "langsung", "pesen": "pesan", "pesenan": "pesanan",
    "tlg": "tolong", "mhn": "mohon", "hrg": "harga", "rb": "ribu", "disc": "diskon",
    # --- penilaian ---
    "bgs": "bagus", "jlk": "jelek", "mantul": "mantap", "mantab": "mantap", "mantep": "mantap",
    "mantaps": "mantap", "ok": "oke", "okey": "oke", "okay": "oke", "okeh": "oke",
    "thx": "terima kasih", "thanks": "terima kasih", "tq": "terima kasih",
    "makasih": "terima kasih", "makasi": "terima kasih", "mksh": "terima kasih",
    "trims": "terima kasih", "trimakasih": "terima kasih", "terimakasih": "terima kasih",
    # --- istilah aplikasi ---
    "apl": "aplikasi", "drvr": "driver", "dvr": "driver", "eror": "error", "erorr": "error",
    "cancle": "cancel",
}
POLA_TAWA = re.compile(r"(wk)+w?|(ha){2,}h?|(he){2,}h?|(hi){2,}h?|(xi)+x?")


def muat_kamus():
    kamus = {}
    if FILE_KAMUS_EKSTERNAL.exists():
        ext = pd.read_csv(FILE_KAMUS_EKSTERNAL, dtype=str, keep_default_na=False)
        if {"slang", "baku"} <= set(ext.columns):
            kamus.update({str(a).strip().lower(): str(b).strip().lower()
                          for a, b in zip(ext["slang"], ext["baku"]) if str(a).strip()})
            print(f"Kamus eksternal dimuat: {len(kamus):,} entri ({FILE_KAMUS_EKSTERNAL.name})")
        else:
            print("[PERHATIAN] kamus eksternal harus punya kolom 'slang' dan 'baku'; diabaikan.")
    else:
        print(f"Kamus eksternal tidak ada ({FILE_KAMUS_EKSTERNAL}); memakai kamus bawaan saja "
              f"({len(KAMUS_BAWAAN)} entri). Disarankan menambah kamus eksternal (lihat catatan).")
    kamus.update(KAMUS_BAWAAN)      # entri bawaan menang bila bentrok
    return kamus


def muat_stopword():
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stop = set(StopWordRemoverFactory().get_stop_words())
        print(f"Stopword Sastrawi dimuat: {len(stop)} kata")
    except Exception as e:
        print(f"[PERHATIAN] Sastrawi tidak tersedia ({type(e).__name__}); memakai daftar stopword "
              f"mini. Pasang Sastrawi agar daftarnya lengkap.")
        stop = set("yang dan di ke dari untuk pada dengan ini itu atau juga saya aku kamu dia mereka "
                   "kami kita adalah ada akan sudah telah masih bisa dapat saja lagi karena jadi "
                   "sama oleh dalam agar supaya sebagai tapi tetapi hanya sangat sekali banget "
                   "tidak bukan belum tanpa jangan kurang tak".split())
    stop |= STOPWORD_TAMBAHAN
    if PERTAHANKAN_NEGASI_DI_TOPIK:
        stop -= NEGASI
    return stop


KAMUS = muat_kamus()
STOPWORD = muat_stopword()

# %% BLOK 3: Fungsi pembersihan
try:
    from emoji import replace_emoji

    def hapus_emoji(t):
        return replace_emoji(t, replace=" ")
except ImportError:                      # versi emoji lama: pakai rentang Unicode
    _POLA_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D]")

    def hapus_emoji(t):
        return _POLA_EMOJI.sub(" ", t)

POLA_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
POLA_EMAIL = re.compile(r"\S+@\S+\.\S+")
POLA_MENTION = re.compile(r"[@#]\w+")
POLA_ULANG_HURUF = re.compile(r"([^\W\d_])\1{2,}")   # 3+ huruf sama berturut-turut ('mantappp')
POLA_ULANG_TANDA = re.compile(r"([!?.,])\1+")        # tanda baca berulang ('!!!')
POLA_SPASI = re.compile(r"\s+")
POLA_HURUF = re.compile(r"[a-z]+")
POLA_NYA = re.compile(r"([a-z]+)\s+nya\b")          # 'driver nya' -> 'drivernya'
POLA_ANGKA_SATUAN = re.compile(r"\d+(?:[.,]\d+)?\s?(?:rb|ribu|jt|juta|k|x)\b")   # '15rb', '3x' dibuang utuh


def dasar(t):
    """Langkah umum: Unicode NFKC, huruf kecil, buang URL/email/mention/emoji."""
    t = unicodedata.normalize("NFKC", t).lower()
    t = POLA_URL.sub(" ", t)
    t = POLA_EMAIL.sub(" ", t)
    t = POLA_MENTION.sub(" ", t)
    return hapus_emoji(t)


def terapkan_kamus(t, kamus, penghitung=None):
    """Ganti kata slang -> baku, dan seragamkan tawa ('wkwk', 'hahaha') menjadi 'haha'."""
    def ganti(m):
        w = m.group(0)
        if POLA_TAWA.fullmatch(w):
            return "haha"
        baku = kamus.get(w)
        if baku is None:
            return w
        if penghitung is not None:
            penghitung[w] += 1
        return baku
    return POLA_HURUF.sub(ganti, t)


def bersihkan_klasik(t, kamus, pakai_kamus=True, penghitung=None):
    """Teks bersih penuh: hanya huruf a-z (angka, tanda baca, simbol, aksara lain dibuang)."""
    t = dasar(t)
    t = POLA_ANGKA_SATUAN.sub(" ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = POLA_ULANG_HURUF.sub(r"\1", t)
    t = POLA_NYA.sub(r"\1nya", t)
    if pakai_kamus:
        t = terapkan_kamus(t, kamus, penghitung)
    return " ".join(w for w in t.split() if len(w) > 1)     # token 1 huruf dibuang


def bersihkan_ringan(t, kamus, pakai_kamus=False):
    """Teks ringan untuk IndoBERT: tanda baca dan angka dipertahankan."""
    t = dasar(t)
    t = POLA_ULANG_HURUF.sub(r"\1", t)
    t = POLA_ULANG_TANDA.sub(r"\1", t)
    t = POLA_NYA.sub(r"\1nya", t)
    if pakai_kamus:
        t = terapkan_kamus(t, kamus)
    return POLA_SPASI.sub(" ", t).strip()


def buat_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    cache = {}

    def stem(w):
        if w not in cache:
            cache[w] = stemmer.stem(w)
        return cache[w]
    return stem


def untuk_topik(t, stop, stem=None):
    """Teks topik: buang stopword dan token < 3 huruf; stemming opsional."""
    tok = [w for w in t.split() if len(w) >= 3 and w not in stop]
    if stem is not None:
        tok = [stem(w) for w in tok]
    return " ".join(tok)


# %% BLOK 4: Muat data dan periksa bahwa file masukan adalah versi terbaru
if not FILE_MASUK.exists():
    sys.exit(f"{FILE_MASUK} tidak ada. Jalankan sampling_gabungan.py lebih dulu.")
df = pd.read_csv(FILE_MASUK, parse_dates=["tanggal"], keep_default_na=False, dtype={"teks": str})
perlu = {"app", "reviewId", "teks", "rating", "tanggal", "thumbs_up", "jumlah_kata"}
if not perlu <= set(df.columns):
    sys.exit("ulasan_gabungan.csv tidak punya kolom 'jumlah_kata': ini masih versi LAMA (tanpa filter "
             "satu kalimat). Jalankan sampling_gabungan.py sampai muncul 'Tersimpan di:', lalu ulangi.")
df = df.rename(columns={"teks": "teks_asli", "jumlah_kata": "jumlah_kata_asli"})
print(f"\nData masuk: {len(df):,} baris")
print(df["app"].value_counts().to_string())

# %% BLOK 5: Terapkan pembersihan (tiga versi teks)
penghitung_slang = Counter()
df["teks_ringan"] = df["teks_asli"].map(
    lambda t: bersihkan_ringan(t, KAMUS, pakai_kamus=NORMALISASI_SLANG_RINGAN))
df["teks_bersih"] = df["teks_asli"].map(
    lambda t: bersihkan_klasik(t, KAMUS, pakai_kamus=NORMALISASI_SLANG_BERSIH,
                               penghitung=penghitung_slang))

# Untuk laporan: seberapa besar normalisasi slang mengecilkan kosakata?
vocab_sesudah = set(w for t in df["teks_bersih"] for w in t.split())
tanpa_kamus = df["teks_asli"].map(lambda t: bersihkan_klasik(t, KAMUS, pakai_kamus=False))
vocab_sebelum = set(w for t in tanpa_kamus for w in t.split())
print(f"\nKosakata teks_bersih: {len(vocab_sebelum):,} (tanpa kamus) -> {len(vocab_sesudah):,} "
      f"(dengan kamus), turun {100 * (1 - len(vocab_sesudah) / max(len(vocab_sebelum), 1)):.1f}%")

stem = buat_stemmer() if LAKUKAN_STEMMING else None
df["teks_topik"] = df["teks_bersih"].map(lambda t: untuk_topik(t, STOPWORD, stem))
df["jumlah_kata_bersih"] = df["teks_bersih"].str.split().str.len()
# Penanda ulasan yang menyebut rating sendiri ('kasih bintang 1'): label bisa 'bocor' lewat teks.
# Kolom ini TIDAK dipakai untuk membuang data; hanya untuk analisis sensitivitas di Tahap 12.
POLA_BINTANG = re.compile(r"bintang|\bstars?\b|\u2b50")
df["menyebut_bintang"] = df["teks_asli"].map(lambda t: bool(POLA_BINTANG.search(t.lower())))   # regex Python, bebas dari mesin pyarrow

# %% BLOK 6: Saring ulasan yang terlalu pendek, lalu buang duplikat
n_awal = df.groupby("app").size()

df = df[df["jumlah_kata_bersih"] >= MIN_KATA_BERSIH].copy()
n_min_kata = df.groupby("app").size()

# Duplikat berdasarkan teks_bersih (setelah normalisasi, lebih ketat daripada sebelum sampling).
# Dicek dulu: berapa kelompok duplikat yang punya label berbeda (derau label).
df["_label"] = df["rating"].map(rating_ke_label)
konflik = int((df.groupby("teks_bersih")["_label"].nunique() > 1).sum())
df = (df.sort_values("tanggal")
        .drop_duplicates("teks_bersih", keep="first")
        .drop(columns="_label"))
n_unik = df.groupby("app").size()

df = df.sort_values(["app", "tanggal"]).reset_index(drop=True)
ringkas = pd.DataFrame({"awal": n_awal, f"min_{MIN_KATA_BERSIH}_kata_bersih": n_min_kata,
                        "setelah_buang_duplikat": n_unik}).reindex(list(APPS.keys())).fillna(0).astype(int)
print("\n=== RINGKASAN PRA-PEMROSESAN ===")
print(ringkas.to_string())
print(f"Kelompok teks duplikat dengan label (dari rating) BERBEDA: {konflik} "
      f"(yang dipertahankan hanya yang paling awal)")

# %% BLOK 7: Simpan dan pemeriksaan
kolom = ["app", "reviewId", "tanggal", "rating", "thumbs_up", "jumlah_kata_asli",
         "jumlah_kata_bersih", "menyebut_bintang", "teks_asli", "teks_ringan", "teks_bersih",
         "teks_topik"]
df[kolom].to_csv(FILE_KELUAR, index=False, encoding="utf-8")
ringkas.to_csv(TAB_DIR / "ringkasan_praproses.csv", encoding="utf-8")

contoh = df.sample(min(UKURAN_CONTOH, len(df)), random_state=SEED)[
    ["app", "rating", "teks_asli", "teks_ringan", "teks_bersih", "teks_topik"]]
contoh.to_csv(TAB_DIR / "contoh_praproses.csv", index=False, encoding="utf-8")

pd.Series(penghitung_slang).sort_values(ascending=False).rename("jumlah").rename_axis("slang") \
    .to_frame().assign(baku=lambda x: [KAMUS[s] for s in x.index]) \
    .to_csv(TAB_DIR / "slang_terganti.csv", encoding="utf-8")

frek = Counter(w for t in df["teks_bersih"] for w in t.split())
pd.DataFrame(frek.most_common(500), columns=["kata", "frekuensi"]) \
    .to_csv(TAB_DIR / "frekuensi_kata_top500.csv", index=False, encoding="utf-8")

pd.set_option("display.width", 200)
print("\n=== PERIKSA SECARA VISUAL (acak) ===")
for _, r in contoh.head(6).iterrows():
    print(f"[{r['app']}|{r['rating']}]")
    print(f"  asli   : {r['teks_asli'][:110]}")
    print(f"  ringan : {r['teks_ringan'][:110]}")
    print(f"  bersih : {r['teks_bersih'][:110]}")
    print(f"  topik  : {r['teks_topik'][:110]}")
print("\n10 pengganti slang paling sering:")
print(pd.Series(penghitung_slang).sort_values(ascending=False).head(10).to_string())
print("\nPanjang teks_bersih (jumlah kata):")
print(df["jumlah_kata_bersih"].describe().round(1).to_string())
print("\nBaris dengan teks_topik kosong:", int((df["teks_topik"].str.strip() == "").sum()))
label = df["rating"].map(rating_ke_label)
jumlah_label = pd.crosstab(df["app"], label).reindex(columns=["negatif", "netral", "positif"], fill_value=0)
jumlah_label.loc["SEMUA"] = jumlah_label.sum()
print("\nJumlah ulasan per aplikasi per label (dari rating):")
print(jumlah_label.to_string())
print("\nProporsi label per aplikasi:")
print(pd.crosstab(df["app"], label, normalize="index").round(3).to_string())
jumlah_label.to_csv(TAB_DIR / "distribusi_label_praproses.csv", encoding="utf-8")

bocor = (df.groupby(label)["menyebut_bintang"].agg(jumlah="sum", persen=lambda x: round(100 * x.mean(), 1)))
print("\nUlasan yang menyebut 'bintang' menurut label (potensi kebocoran label lewat teks):")
print(bocor.to_string())
bocor.to_csv(TAB_DIR / "kebocoran_label_bintang.csv", encoding="utf-8")

(TAB_DIR / "parameter_praproses.txt").write_text(
    f"min_kata_bersih: {MIN_KATA_BERSIH}\n"
    f"normalisasi_slang_bersih: {NORMALISASI_SLANG_BERSIH}\n"
    f"normalisasi_slang_ringan: {NORMALISASI_SLANG_RINGAN}\n"
    f"stemming: {LAKUKAN_STEMMING}\n"
    f"pertahankan_negasi_di_topik: {PERTAHANKAN_NEGASI_DI_TOPIK}\n"
    f"entri_kamus_slang: {len(KAMUS)}\n"
    f"jumlah_stopword: {len(STOPWORD)}\n"
    f"kosakata_tanpa_kamus: {len(vocab_sebelum)}\n"
    f"kosakata_dengan_kamus: {len(vocab_sesudah)}\n"
    f"jumlah_akhir: {len(df)}\n", encoding="utf-8")
print(f"\nTersimpan di: {FILE_KELUAR}  ({len(df):,} baris)")
