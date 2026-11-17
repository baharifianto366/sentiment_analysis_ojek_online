"""
tahap5_lda.py  -  Tahap 5: pemodelan topik LDA, jumlah topik k dipilih lewat koherensi C_v.

Letakkan di sebelah folder penelitian-ojol/ atau di dalamnya.

Cara pakai:
    python tahap5_lda.py --uji       # uji cepat (4.000 ulasan, k = 5 dan 8, beberapa menit): JALANKAN INI DULU
    python tahap5_lda.py             # proses sebenarnya (semua ulasan, beberapa nilai k)

Opsi:
    --k 12            paksa k akhir tertentu (model dilatih bila belum ada)
    --ulang           latih ulang semua model (abaikan model yang sudah tersimpan)
    --proses 1        jumlah proses untuk menghitung koherensi (1 paling aman di Windows)
    --no-below 5      ganti batas minimal kemunculan kata (uji sensitivitas)
    --min-token 2     ganti batas minimal token per ulasan (uji sensitivitas)
    --tag lebih_inklusif   akhiran nama file agar varian tidak menimpa hasil utama
    --stabilitas      uji kestabilan C_v: beberapa seed untuk k tertentu (tanpa keluaran topik)

Input : data/processed/ulasan_berlabel.csv
Output: outputs/tables/lda_*.csv, outputs/figures/lda_*.png, outputs/models/lda_k*.model
        data/processed sengaja tidak diubah (hasil topik per ulasan ada di outputs/tables/lda_topik_dokumen.csv)

Model tiap k disimpan; menjalankan ulang dengan --k lain tidak melatih ulang model yang sudah ada.
Cache dibatalkan otomatis bila pengaturan (no_below, stopword, dst.) berubah.
Semua keluaran yang bergantung pada k memakai akhiran _k{K}, sehingga beberapa k bisa dibandingkan.
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")
try:
    from gensim import corpora
    from gensim.models import LdaModel, CoherenceModel
except ImportError as e:
    sys.exit(f"gensim belum terpasang atau bentrok ({e}). Coba: pip install -U gensim "
             f"(bila pesan menyebut 'triu' dari scipy, itu bentrok versi gensim-scipy: update gensim).")

from config import *            # noqa: F401,F403  (SEED, PROC_DIR, TAB_DIR, FIG_DIR, MODEL_DIR, ...)
from utils import set_seed, make_dirs

logging.getLogger("gensim").setLevel(logging.WARNING)

# --------------------------------------------------------------------------
# Konfigurasi tahap 5 (ubah di sini)
# --------------------------------------------------------------------------
FILE_MASUK = PROC_DIR / "ulasan_berlabel.csv"
K_LIST = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25]    # nilai k yang dibandingkan
K_STABIL = [5, 8, 10, 12]      # k yang diuji pada --stabilitas
N_SEED_STABIL = 5
K_LIST_UJI = [5, 8]
NO_BELOW = 10                  # kata harus muncul di >= 10 ulasan
NO_ABOVE = 0.5                 # kata yang muncul di > 50% ulasan dibuang
MIN_TOKEN_DOK = 3              # ulasan dengan token kurang dari ini (setelah penyaringan) tidak dilatih
PASSES = 10
ITERASI = 100
CHUNKSIZE = 2000
TOP_KATA = 15                  # kata per topik pada tabel
TOP_KATA_KOHERENSI = 10        # kata per topik untuk menghitung C_v
PERTAHANKAN_NEGASI = False     # True = 'tidak', 'belum', dst. tetap dipakai oleh LDA
UKURAN_UJI = 4000
TOLERANSI_K_HEMAT = 0.01       # k 'hemat' = k terkecil dengan C_v >= maksimum - toleransi
NEGASI = {"tidak", "bukan", "belum", "tanpa", "jangan", "kurang", "tak"}

# Kata fungsi/informal yang tidak ada di daftar stopword Sastrawi tetapi sangat umum pada ulasan
STOPWORD_INFORMAL = set("""
kalau banget sangat sekali padahal jadi sama mau kasih terima memang terus makanya pas ya yah sih deh dong
kok kan tuh nih woi haha oke kemana mana begitu begini buat bikin lagi udah sudah saja cuma tapi kayak
seperti gimana bagaimana kenapa apa aja bisa dapat ada pakai nggak gak ga gk lebih paling semua masih
harus sempat sempet dulu nanti langsung banyak sedikit kali dari jadi aku kamu saya kita kami dia mereka
malah sekarang makin sering selalu
""".split())
# Nama merek dan variannya (ejaan ganda) dibuang agar topik tidak sekadar memisahkan aplikasi
KATA_MEREK = set("""
gojek gojeg gojeknya grab grabnya grep greb grap grabnya maxim maksim maxsim maximnya aplikasi aplikasinya
apk app apps
""".split())



# --------------------------------------------------------------------------
# Stopword dan tokenisasi
# --------------------------------------------------------------------------
def muat_stopword():
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stop = set(StopWordRemoverFactory().get_stop_words())
        print(f"Stopword Sastrawi dimuat: {len(stop)} kata")
    except Exception as e:
        print(f"[PERHATIAN] Sastrawi tidak tersedia ({type(e).__name__}); memakai daftar mini. "
              f"Pasang Sastrawi agar daftar stopword lengkap.")
        stop = set("yang dan di ke dari untuk pada dengan ini itu atau juga adalah akan telah oleh dalam "
                   "agar supaya sebagai karena hanya".split())
    stop |= STOPWORD_INFORMAL | KATA_MEREK | NEGASI
    if PERTAHANKAN_NEGASI:
        stop -= NEGASI
    return stop


STOP = muat_stopword()


def hapus_nya(w):
    """'drivernya' -> 'driver'; hanya bila sisa kata >= 4 huruf (menjaga 'punya', 'tanya', 'hanya')."""
    return w[:-3] if (w.endswith("nya") and len(w) >= 7) else w


def buat_token(teks_bersih):
    t = teks_bersih.replace("terima kasih", " ")
    hasil = []
    for w in t.split():
        w = hapus_nya(w)
        if len(w) >= 3 and w not in STOP:
            hasil.append(w)
    return hasil


# --------------------------------------------------------------------------
# Fungsi bantu model
# --------------------------------------------------------------------------
def diversitas_topik(model, k, topn=10):
    """Proporsi kata unik pada top-n kata seluruh topik (1 = tidak ada kata yang berulang antartopik)."""
    kata = [w for t in range(k) for w, _ in model.show_topic(t, topn=topn)]
    return len(set(kata)) / max(len(kata), 1)


def latih_dan_ukur(k, corpus, dictionary, tokens_lda, proses, seed=SEED):
    t0 = time.perf_counter()
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=seed, passes=PASSES,
                     iterations=ITERASI, chunksize=CHUNKSIZE, alpha="auto", eta="auto", eval_every=None)
    waktu_latih = time.perf_counter() - t0
    t1 = time.perf_counter()
    cv = CoherenceModel(model=model, texts=tokens_lda, dictionary=dictionary, coherence="c_v",
                        topn=TOP_KATA_KOHERENSI, processes=proses).get_coherence()
    waktu_koh = time.perf_counter() - t1
    return model, float(cv), diversitas_topik(model, k), waktu_latih, waktu_koh


def main():
    ap = argparse.ArgumentParser(description="Tahap 5: LDA dengan pemilihan k lewat C_v")
    ap.add_argument("--uji", action="store_true", help="uji cepat pada subset kecil")
    ap.add_argument("--k", type=int, default=None, help="paksa k akhir")
    ap.add_argument("--ulang", action="store_true", help="latih ulang semua model")
    ap.add_argument("--proses", type=int, default=1, help="proses paralel untuk koherensi (default 1)")
    ap.add_argument("--no-below", type=int, default=NO_BELOW)
    ap.add_argument("--min-token", type=int, default=MIN_TOKEN_DOK)
    ap.add_argument("--tag", default="", help="akhiran nama file untuk varian")
    ap.add_argument("--stabilitas", action="store_true", help="uji kestabilan C_v antar-seed")
    args = ap.parse_args()
    no_below, min_token = args.no_below, args.min_token

    make_dirs()
    set_seed(SEED)
    sfx = ("_uji" if args.uji else "") + (f"_{args.tag}" if args.tag else "")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25})
    pd.set_option("display.width", 200)

    # ---------- 1. Muat dan tokenisasi ----------
    if not FILE_MASUK.exists():
        sys.exit(f"{FILE_MASUK} tidak ada. Selesaikan Tahap 3 (--buat) lebih dulu.")
    d = pd.read_csv(FILE_MASUK, keep_default_na=False,
                    dtype={"reviewId": str, "teks_asli": str, "teks_bersih": str})
    perlu = {"reviewId", "app", "label_final", "teks_asli", "teks_bersih"}
    if not perlu <= set(d.columns):
        sys.exit(f"Kolom tidak lengkap di {FILE_MASUK.name}; kurang: {sorted(perlu - set(d.columns))}")
    if args.uji:
        d = d.sample(n=min(UKURAN_UJI, len(d)), random_state=SEED).reset_index(drop=True)
        print(f"[MODE UJI] memakai {len(d)} ulasan")
    print(f"Ulasan masuk: {len(d):,}")

    tokens = d["teks_bersih"].map(buat_token).tolist()
    dictionary = corpora.Dictionary(tokens)
    n_kosakata_awal = len(dictionary)
    dictionary.filter_extremes(no_below=no_below, no_above=NO_ABOVE, keep_n=None)
    print(f"Kosakata: {n_kosakata_awal:,} -> {len(dictionary):,} "
          f"(no_below={no_below}, no_above={NO_ABOVE})")

    tok_f = [[w for w in doc if w in dictionary.token2id] for doc in tokens]
    simpan_idx = [i for i, doc in enumerate(tok_f) if len(doc) >= min_token]
    d["_dipakai"] = False
    d.loc[simpan_idx, "_dipakai"] = True
    ter = []
    for kolom in ("label_final", "app"):
        g = d.groupby(kolom)["_dipakai"].agg(total="size", dipakai="sum")
        g["persen_terbuang"] = (100 * (1 - g["dipakai"] / g["total"])).round(1)
        ter.append(g.reset_index().rename(columns={kolom: "kelompok"}).assign(jenis=kolom))
    ter = pd.concat(ter)
    ter.to_csv(TAB_DIR / f"lda_ulasan_terbuang{sfx}.csv", index=False, encoding="utf-8")
    d_lda = d.iloc[simpan_idx].drop(columns="_dipakai").reset_index(drop=True)
    tokens_lda = [tok_f[i] for i in simpan_idx]
    corpus = [dictionary.doc2bow(doc) for doc in tokens_lda]
    print(f"Ulasan dipakai LDA: {len(d_lda):,} dari {len(d):,} "
          f"({len(d) - len(d_lda)} dibuang karena < {min_token} token setelah penyaringan)")
    print("Persentase ulasan yang TERBUANG menurut kelompok (ulasan pendek cenderung tersisih):")
    for jenis, g in ter.groupby("jenis", sort=False):
        print(f"  menurut {jenis}: " + ", ".join(f"{r.kelompok} {r.persen_terbuang}%" for r in g.itertuples()))
    print(f"Token per ulasan: median {np.median([len(x) for x in tokens_lda]):.0f}, "
          f"rata-rata {np.mean([len(x) for x in tokens_lda]):.1f}")

    # ---------- 1b. Uji kestabilan (opsional) ----------
    if args.stabilitas:
        print(f"\n=== UJI KESTABILAN C_v: {N_SEED_STABIL} seed per k ===")
        baris = []
        for k in K_STABIL:
            nilai = []
            for i in range(N_SEED_STABIL):
                _, cv, _, _, _ = latih_dan_ukur(k, corpus, dictionary, tokens_lda, args.proses, seed=SEED + i)
                nilai.append(cv)
            baris.append({"k": k, "c_v_rata2": np.mean(nilai), "c_v_sd": np.std(nilai, ddof=1),
                          "c_v_min": np.min(nilai), "c_v_maks": np.max(nilai)})
            print(f"k={k:>2}: C_v rata-rata {np.mean(nilai):.4f} (sd {np.std(nilai, ddof=1):.4f}; "
                  f"rentang {np.min(nilai):.4f} - {np.max(nilai):.4f})")
        st = pd.DataFrame(baris)
        st.round(4).to_csv(TAB_DIR / f"lda_stabilitas_k{sfx}.csv", index=False, encoding="utf-8")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(st["k"], st["c_v_rata2"], yerr=st["c_v_sd"], marker="o", capsize=4, color="#1f77b4")
        ax.set_xlabel("Jumlah topik (k)")
        ax.set_ylabel("Koherensi C_v (rata-rata ± sd)")
        ax.set_title(f"Kestabilan C_v antar-seed (n = {N_SEED_STABIL})")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"lda_stabilitas_k{sfx}.png", dpi=200)
        plt.close(fig)
        print("Selesai (mode kestabilan tidak membuat keluaran topik).")
        return

    # ---------- 2. Latih model untuk beberapa k, ukur koherensi C_v ----------
    tanda = {"no_below": no_below, "no_above": NO_ABOVE, "min_token": min_token, "passes": PASSES,
             "iterasi": ITERASI, "chunksize": CHUNKSIZE, "n_ulasan": len(d), "n_dipakai": len(d_lda),
             "kosakata": len(dictionary), "negasi": PERTAHANKAN_NEGASI,
             "stopword": hashlib.md5("|".join(sorted(STOP)).encode("utf-8")).hexdigest()}
    file_tanda = TAB_DIR / f"lda_cache_param{sfx}.json"
    gunakan_cache = not args.ulang
    if gunakan_cache and file_tanda.exists():
        if json.loads(file_tanda.read_text(encoding="utf-8")) != tanda:
            print("[INFO] Pengaturan berbeda dari proses sebelumnya (stopword/no_below/dll.): "
                  "model lama diabaikan dan semua k dilatih ulang.")
            gunakan_cache = False
    elif gunakan_cache:
        gunakan_cache = False        # belum ada tanda tangan: tidak memakai cache lama
    file_tanda.write_text(json.dumps(tanda, indent=1), encoding="utf-8")
    daftar_k = sorted(set((K_LIST_UJI if args.uji else K_LIST) + ([args.k] if args.k else [])))
    file_koh = TAB_DIR / f"lda_koherensi_per_k{sfx}.csv"
    lama = pd.read_csv(file_koh) if (file_koh.exists() and gunakan_cache) else pd.DataFrame(
        columns=["k", "c_v", "diversitas_topik", "detik_latih", "detik_koherensi"])
    baris, model_k = [], {}
    print(f"\n=== LATIH LDA untuk k = {daftar_k} (PASSES={PASSES}, ITERASI={ITERASI}) ===")
    for k in daftar_k:
        path = MODEL_DIR / f"lda_k{k}{sfx}.model"
        ada = (lama["k"] == k).any() and path.exists()
        if ada:
            model_k[k] = LdaModel.load(str(path))
            r = lama[lama["k"] == k].iloc[0].to_dict()
            print(f"k={k:>2}: dimuat dari model tersimpan | C_v = {r['c_v']:.4f}")
            baris.append(r)
        else:
            model, cv, div, wl, wk = latih_dan_ukur(k, corpus, dictionary, tokens_lda, args.proses)
            model.save(str(path))
            model_k[k] = model
            print(f"k={k:>2}: C_v = {cv:.4f} | diversitas = {div:.3f} | latih {wl:.0f} dtk, koherensi {wk:.0f} dtk")
            baris.append({"k": k, "c_v": cv, "diversitas_topik": div, "detik_latih": round(wl, 1),
                          "detik_koherensi": round(wk, 1)})
        pd.DataFrame(baris).to_csv(file_koh, index=False, encoding="utf-8")   # simpan bertahap
    hasil = pd.DataFrame(baris).sort_values("k").reset_index(drop=True)

    k_maks = int(hasil.loc[hasil["c_v"].idxmax(), "k"])
    k_hemat = int(hasil[hasil["c_v"] >= hasil["c_v"].max() - TOLERANSI_K_HEMAT]["k"].min())
    k_final = args.k or k_maks
    print("\n=== PEMILIHAN k ===")
    print(hasil.round(4).to_string(index=False))
    print(f"k dengan C_v tertinggi : {k_maks}")
    print(f"k 'hemat' (C_v dalam {TOLERANSI_K_HEMAT} dari maksimum, k terkecil): {k_hemat}")
    print(f"k yang dipakai untuk keluaran: {k_final}" + ("  (dipaksa lewat --k)" if args.k else ""))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hasil["k"], hasil["c_v"], marker="o", color="#1f77b4", label="Koherensi C_v")
    ax.axvline(k_final, color="gray", ls="--", lw=1, label=f"k terpilih = {k_final}")
    ax.set_xlabel("Jumlah topik (k)")
    ax.set_ylabel("Koherensi C_v", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(hasil["k"], hasil["diversitas_topik"], marker="s", color="#ff7f0e", label="Diversitas topik")
    ax2.set_ylabel("Diversitas topik", color="#ff7f0e")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    ax.set_title("Pemilihan jumlah topik LDA")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"lda_koherensi_per_k{sfx}.png", dpi=200)
    plt.close(fig)

    # ---------- 3. Keluaran untuk k terpilih ----------
    model = model_k[k_final]
    K = k_final
    theta = np.zeros((len(corpus), K))
    for i, bow in enumerate(corpus):
        for t, p in model.get_document_topics(bow, minimum_probability=0.0):
            theta[i, t] = p
    dominan = theta.argmax(axis=1)
    prob_dom = theta.max(axis=1)
    n_tok = np.array([len(x) for x in tokens_lda])

    # (a) kata per topik
    baris_kata = []
    for t in range(K):
        for r, (w, p) in enumerate(model.show_topic(t, topn=TOP_KATA), 1):
            baris_kata.append({"topik": t, "peringkat": r, "kata": w, "bobot": round(float(p), 5)})
    pd.DataFrame(baris_kata).to_csv(TAB_DIR / f"lda_topik_k{K}{sfx}.csv", index=False, encoding="utf-8")

    # (b) topik per ulasan
    dok = d_lda[["reviewId", "app", "label_final"]].copy()
    dok["topik_dominan"] = dominan
    dok["prob_dominan"] = prob_dom.round(4)
    for t in range(K):
        dok[f"p_topik_{t}"] = theta[:, t].round(4)
    dok.to_csv(TAB_DIR / f"lda_topik_dokumen_k{K}{sfx}.csv", index=False, encoding="utf-8")

    # (c) prevalensi dan perbandingan antaraplikasi (berdasar topik dominan)
    prev = pd.Series(dominan).value_counts(normalize=True).reindex(range(K), fill_value=0)
    per_app = pd.crosstab(dok["topik_dominan"], dok["app"], normalize="columns").reindex(range(K), fill_value=0)
    per_app.round(4).to_csv(TAB_DIR / f"lda_topik_per_app_k{K}{sfx}.csv", encoding="utf-8")

    # (d) templat penamaan manual: kata teratas + contoh ulasan paling representatif
    penamaan = []
    for t in range(K):
        kata = ", ".join(w for w, _ in model.show_topic(t, topn=10))
        calon = np.flatnonzero((dominan == t) & (n_tok >= 5))
        calon = calon[np.argsort(-theta[calon, t])][:3]
        contoh = [d_lda.loc[i, "teks_asli"].replace("\n", " ")[:180] for i in calon]
        contoh += [""] * (3 - len(contoh))
        penamaan.append({"topik": t, "persen_ulasan_dominan": round(100 * prev[t], 1), "kata_teratas": kata,
                         "contoh_1": contoh[0], "contoh_2": contoh[1], "contoh_3": contoh[2],
                         "nama_manual": "", "catatan": ""})
    penamaan = pd.DataFrame(penamaan)
    penamaan.to_csv(TAB_DIR / f"lda_penamaan_topik_k{K}{sfx}.csv", index=False, encoding="utf-8")

    # (e) grafik prevalensi topik
    fig, ax = plt.subplots(figsize=(7, 0.35 * K + 1.5))
    ax.barh([f"T{t}" for t in range(K)][::-1], (100 * prev.values)[::-1], color="#2a9d8f")
    ax.set_xlabel("% ulasan (topik dominan)")
    ax.set_title(f"Prevalensi topik LDA (k = {K})")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"lda_prevalensi_topik_k{K}{sfx}.png", dpi=200)
    plt.close(fig)

    (TAB_DIR / f"lda_parameter_k{K}{sfx}.txt").write_text(
        f"k_final: {K}\nk_C_v_tertinggi: {k_maks}\nk_hemat: {k_hemat}\nk_diuji: {daftar_k}\n"
        f"passes: {PASSES}\niterasi: {ITERASI}\nchunksize: {CHUNKSIZE}\nalpha: auto\neta: auto\nseed: {SEED}\n"
        f"no_below: {no_below}\nno_above: {NO_ABOVE}\nmin_token_dokumen: {min_token}\n"
        f"pertahankan_negasi: {PERTAHANKAN_NEGASI}\njumlah_stopword: {len(STOP)}\n"
        f"kosakata_akhir: {len(dictionary)}\nulasan_dipakai: {len(d_lda)}\nulasan_total: {len(d)}\n"
        f"c_v_k_final: {float(hasil.loc[hasil['k'] == K, 'c_v'].iloc[0]):.4f}\n", encoding="utf-8")

    print(f"\n=== TOPIK (k = {K}) ===")
    for t in range(K):
        print(f"T{t:<2} ({100 * prev[t]:4.1f}%): " + ", ".join(w for w, _ in model.show_topic(t, topn=10)))
    print("\nProporsi topik dominan per aplikasi (kolom = aplikasi, 5 topik terbesar per aplikasi):")
    for a in per_app.columns:
        besar = per_app[a].sort_values(ascending=False).head(5)
        print(f"  {a:6s}: " + ", ".join(f"T{t}={100 * v:.0f}%" for t, v in besar.items()))
    print(f"\nTemplat penamaan: {TAB_DIR / f'lda_penamaan_topik_k{K}{sfx}.csv'}  (isi kolom nama_manual)")
    print(f"Tabel dan gambar lain ada di {TAB_DIR} dan {FIG_DIR}")
    if args.uji:
        print("\n[MODE UJI] Bila semuanya berjalan tanpa error, jalankan tanpa --uji untuk hasil sebenarnya.")


if __name__ == "__main__":
    main()
