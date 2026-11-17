"""
tahap6_bertopic.py  -  Tahap 6: BERTopic (embedding kalimat + UMAP + HDBSCAN), perbandingan dengan LDA,
                       dan templat penamaan topik.

Letakkan di sebelah folder penelitian-ojol/ atau di dalamnya.

Cara pakai:
    python tahap6_bertopic.py --uji        # uji cepat (3.000 ulasan): JALANKAN INI DULU
    python tahap6_bertopic.py              # proses sebenarnya (semua ulasan)

Opsi:
    --min-cluster 80      ukuran klaster minimal HDBSCAN (makin besar = makin sedikit topik)
    --nr-topics 12        gabungkan topik hingga sejumlah ini (mis. agar sebanding dengan LDA k = 12)
    --tag nama            akhiran nama file agar varian tidak saling menimpa
    --stabilitas 5        ulangi UMAP+HDBSCAN dengan 5 seed, laporkan jumlah topik, outlier, dan C_v
    --lda-k 12 --lda-tag inklusif    model LDA yang dijadikan pembanding (hasil Tahap 5)
    --proses 1            proses paralel untuk koherensi (1 paling aman di Windows)

Input : data/processed/ulasan_berlabel.csv (kolom teks_ringan untuk embedding, teks_bersih untuk kata topik)
        outputs/models/lda_k{K}_{tag}.model dan outputs/tables/lda_topik_dokumen_k{K}_{tag}.csv  (opsional,
        untuk perbandingan dengan LDA)
Output: outputs/tables/bertopic_*.csv, perbandingan_lda_bertopic*.csv, outputs/figures/bertopic_*.png

Embedding disimpan di data/interim/ (cache) agar proses berikutnya tidak menghitung ulang.
Pertama kali dijalankan, model embedding diunduh dari internet (ratusan MB).
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
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import normalized_mutual_info_score
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")
try:
    from gensim import corpora
    from gensim.models import CoherenceModel, LdaModel
except ImportError as e:
    sys.exit(f"gensim belum terpasang atau bentrok ({e}). Coba: pip install -U gensim")
try:
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from bertopic import BERTopic
except ImportError as e:
    sys.exit(f"Library BERTopic belum lengkap ({e}). Pasang: pip install bertopic sentence-transformers umap-learn")
try:
    from hdbscan import HDBSCAN as _HDBSCAN
    ADA_HDBSCAN_PKG = True
except ImportError:
    ADA_HDBSCAN_PKG = False
    try:
        from sklearn.cluster import HDBSCAN as _HDBSCAN        # scikit-learn >= 1.3
    except ImportError:
        sys.exit("hdbscan tidak terpasang dan scikit-learn Anda < 1.3. Pasang hdbscan "
                 "(butuh Microsoft C++ Build Tools di Windows) atau perbarui scikit-learn.")

from config import *            # noqa: F401,F403  (SEED, PROC_DIR, INTERIM_DIR, TAB_DIR, FIG_DIR, MODEL_DIR, ...)
from utils import set_seed, make_dirs

logging.getLogger("gensim").setLevel(logging.WARNING)

# --------------------------------------------------------------------------
# Konfigurasi tahap 6 (ubah di sini)
# --------------------------------------------------------------------------
FILE_MASUK = PROC_DIR / "ulasan_berlabel.csv"
MODEL_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"    # model multilingual (termasuk bahasa Indonesia)
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
MIN_CLUSTER_SIZE = 80
MIN_SAMPLES = 10
TOP_N_KATA = 10
CV_MIN_DF = 5                    # kosakata topik: kata muncul di >= 5 ulasan ...
CV_MAX_DF = 0.5                  # ... dan di <= 50% ulasan (sama dengan LDA varian inklusif)
UKURAN_UJI = 3000
BATCH_EMBEDDING = 64
PERTAHANKAN_NEGASI = False
NEGASI = {"tidak", "bukan", "belum", "tanpa", "jangan", "kurang", "tak"}

# --- Tokenisasi untuk kata topik dan koherensi: HARUS sama dengan tahap5_lda.py ---
STOPWORD_INFORMAL = set("""
kalau banget sangat sekali padahal jadi sama mau kasih terima memang terus makanya pas ya yah sih deh dong
kok kan tuh nih woi haha oke kemana mana begitu begini buat bikin lagi udah sudah saja cuma tapi kayak
seperti gimana bagaimana kenapa apa aja bisa dapat ada pakai nggak gak ga gk lebih paling semua masih
harus sempat sempet dulu nanti langsung banyak sedikit kali dari jadi aku kamu saya kita kami dia mereka
malah sekarang makin sering selalu
the and is to of for you it in this that are with my but
""".split())
KATA_MEREK = set("""
gojek gojeg gojeknya grab grabnya grep greb grap maxim maksim maxsim maximnya aplikasi aplikasinya
apk app apps
""".split())


def muat_stopword():
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stop = set(StopWordRemoverFactory().get_stop_words())
        print(f"Stopword Sastrawi dimuat: {len(stop)} kata")
    except Exception as e:
        print(f"[PERHATIAN] Sastrawi tidak tersedia ({type(e).__name__}); memakai daftar mini.")
        stop = set("yang dan di ke dari untuk pada dengan ini itu atau juga adalah akan telah oleh dalam "
                   "agar supaya sebagai karena hanya".split())
    stop |= STOPWORD_INFORMAL | KATA_MEREK | NEGASI
    if PERTAHANKAN_NEGASI:
        stop -= NEGASI
    return stop


STOP = muat_stopword()


def hapus_nya(w):
    return w[:-3] if (w.endswith("nya") and len(w) >= 7) else w


def buat_token(teks_bersih):
    hasil = []
    for w in teks_bersih.replace("terima kasih", " ").split():
        w = hapus_nya(w)
        if len(w) >= 3 and w not in STOP:
            hasil.append(w)
    return hasil


# --------------------------------------------------------------------------
# Fungsi bantu
# --------------------------------------------------------------------------
def hitung_embedding(teks, ids, sfx):
    slug = MODEL_EMBEDDING.replace("/", "_")
    f_npy = INTERIM_DIR / f"embedding_{slug}{sfx}.npy"
    f_sig = INTERIM_DIR / f"embedding_{slug}{sfx}.json"
    sig = hashlib.md5(("\n".join(ids) + "||" + "\n".join(teks)).encode("utf-8")).hexdigest()
    if f_npy.exists() and f_sig.exists() and json.loads(f_sig.read_text())["sig"] == sig:
        emb = np.load(f_npy)
        if emb.shape[0] == len(teks):
            print(f"Embedding dimuat dari cache: {f_npy.name} {emb.shape}")
            return emb
    print(f"Menghitung embedding dengan '{MODEL_EMBEDDING}' untuk {len(teks):,} ulasan "
          f"(pertama kali mengunduh model; di CPU bisa beberapa menit)...")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_EMBEDDING)
    emb = model.encode(teks, batch_size=BATCH_EMBEDDING, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(f_npy, emb)
    f_sig.write_text(json.dumps({"sig": sig, "model": MODEL_EMBEDDING, "n": len(teks)}), encoding="utf-8")
    print(f"Embedding selesai dalam {time.perf_counter() - t0:.0f} dtk, disimpan: {f_npy.name} {emb.shape}")
    return emb


def buat_hdbscan(min_cluster):
    if ADA_HDBSCAN_PKG:
        return _HDBSCAN(min_cluster_size=min_cluster, min_samples=MIN_SAMPLES, metric="euclidean",
                        cluster_selection_method="eom", prediction_data=True)
    return _HDBSCAN(min_cluster_size=min_cluster, min_samples=MIN_SAMPLES, metric="euclidean",
                    cluster_selection_method="eom")


def fit_bertopic(emb, docs, seed, min_cluster, nr_topics):
    """UMAP -> HDBSCAN -> c-TF-IDF. Kosakata topik dibuat dari ulasan individual (bukan per topik)."""
    vocab = list(CountVectorizer(token_pattern=r"\S+", min_df=CV_MIN_DF, max_df=CV_MAX_DF).fit(docs)
                 .get_feature_names_out())
    vect = CountVectorizer(vocabulary=vocab, token_pattern=r"\S+")
    umap_model = UMAP(n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS, min_dist=UMAP_MIN_DIST,
                      metric="cosine", random_state=seed)
    tm = BERTopic(embedding_model=None, umap_model=umap_model, hdbscan_model=buat_hdbscan(min_cluster),
                  vectorizer_model=vect, nr_topics=nr_topics, top_n_words=TOP_N_KATA,
                  calculate_probabilities=False, verbose=False)
    tm.fit_transform(docs, embeddings=emb)
    return tm, np.array(tm.topics_)


def kata_topik(tm):
    """{topik: [kata,...]} tanpa topik -1 (outlier)."""
    return {t: [w for w, _ in kata[:TOP_N_KATA]] for t, kata in tm.get_topics().items() if t != -1}


def diversitas(daftar_kata):
    semua = [w for k in daftar_kata for w in k]
    return len(set(semua)) / max(len(semua), 1)


def hitung_cv(daftar_kata, tokens, kamus, proses):
    t = [[w for w in k if w in kamus.token2id] for k in daftar_kata]
    t = [x for x in t if len(x) >= 2]
    if len(t) < 2:
        return float("nan")
    topn = min(TOP_N_KATA, min(len(x) for x in t))
    return float(CoherenceModel(topics=t, texts=tokens, dictionary=kamus, coherence="c_v", topn=topn,
                                processes=proses).get_coherence())


def main():
    ap = argparse.ArgumentParser(description="Tahap 6: BERTopic")
    ap.add_argument("--uji", action="store_true")
    ap.add_argument("--min-cluster", type=int, default=MIN_CLUSTER_SIZE)
    ap.add_argument("--nr-topics", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stabilitas", type=int, default=0, metavar="N")
    ap.add_argument("--lda-k", type=int, default=12)
    ap.add_argument("--lda-tag", default="inklusif")
    ap.add_argument("--proses", type=int, default=1)
    args = ap.parse_args()

    make_dirs()
    set_seed(SEED)
    sfx = ("_uji" if args.uji else "") + (f"_{args.tag}" if args.tag else "")
    min_cluster = 15 if (args.uji and args.min_cluster == MIN_CLUSTER_SIZE) else args.min_cluster
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25})
    pd.set_option("display.width", 200)

    # ---------- 1. Data ----------
    if not FILE_MASUK.exists():
        sys.exit(f"{FILE_MASUK} tidak ada. Selesaikan Tahap 3 (--buat) lebih dulu.")
    d = pd.read_csv(FILE_MASUK, keep_default_na=False,
                    dtype={"reviewId": str, "teks_asli": str, "teks_ringan": str, "teks_bersih": str})
    perlu = {"reviewId", "app", "label_final", "teks_asli", "teks_ringan", "teks_bersih"}
    if not perlu <= set(d.columns):
        sys.exit(f"Kolom tidak lengkap di {FILE_MASUK.name}; kurang: {sorted(perlu - set(d.columns))}")
    if args.uji:
        d = d.sample(n=min(UKURAN_UJI, len(d)), random_state=SEED).reset_index(drop=True)
        print(f"[MODE UJI] memakai {len(d)} ulasan")
    print(f"Ulasan: {len(d):,}")

    tokens = d["teks_bersih"].map(buat_token).tolist()
    docs = [" ".join(t) for t in tokens]                 # teks untuk c-TF-IDF (kata topik)
    n_tok = np.array([len(t) for t in tokens])
    kamus = corpora.Dictionary(tokens)                    # kamus acuan untuk C_v (sama untuk LDA dan BERTopic)

    # ---------- 2. Embedding ----------
    emb = hitung_embedding(d["teks_ringan"].tolist(), d["reviewId"].tolist(), "_uji" if args.uji else "")

    # ---------- 3. Uji kestabilan (opsional) ----------
    if args.stabilitas > 0:
        print(f"\n=== UJI KESTABILAN: {args.stabilitas} seed (UMAP + HDBSCAN), min_cluster={min_cluster}, "
              f"nr_topics={args.nr_topics} ===")
        baris = []
        for i in range(args.stabilitas):
            tm, tp = fit_bertopic(emb, docs, SEED + i, min_cluster, args.nr_topics)
            kt = kata_topik(tm)
            cv = hitung_cv(list(kt.values()), tokens, kamus, args.proses)
            baris.append({"seed": SEED + i, "n_topik": len(kt), "persen_outlier": round(100 * float((tp == -1).mean()), 1),
                          "c_v": round(cv, 4), "diversitas": round(diversitas(list(kt.values())), 3)})
            print(baris[-1])
        st = pd.DataFrame(baris)
        st.to_csv(TAB_DIR / f"bertopic_stabilitas{sfx}.csv", index=False, encoding="utf-8")
        print(f"\nRata-rata: topik {st['n_topik'].mean():.1f} (sd {st['n_topik'].std(ddof=1):.1f}), "
              f"outlier {st['persen_outlier'].mean():.1f}%, C_v {st['c_v'].mean():.4f} (sd {st['c_v'].std(ddof=1):.4f})")
        return

    # ---------- 4. Latih BERTopic ----------
    print(f"\n=== BERTopic: min_cluster_size={min_cluster}, nr_topics={args.nr_topics} ===")
    t0 = time.perf_counter()
    tm, topics = fit_bertopic(emb, docs, SEED, min_cluster, args.nr_topics)
    print(f"Selesai dalam {time.perf_counter() - t0:.0f} dtk")
    kt = kata_topik(tm)
    ids_topik = sorted(kt)
    n_topik = len(ids_topik)
    persen_out = 100 * float((topics == -1).mean())
    print(f"Jumlah topik (tanpa outlier): {n_topik} | ulasan outlier (-1): {persen_out:.1f}%")
    if n_topik < 2:
        sys.exit("Kurang dari 2 topik terbentuk. Kecilkan --min-cluster (mis. 40) lalu ulangi.")

    # Alokasikan ulang outlier ke topik terdekat (untuk analisis per topik di Tahap 12)
    try:
        topics_red = np.array(tm.reduce_outliers(docs, topics.tolist(), strategy="embeddings", embeddings=emb))
    except Exception as e:
        print(f"[PERHATIAN] reduce_outliers gagal ({type(e).__name__}: {e}); outlier tidak dialokasikan ulang.")
        topics_red = topics.copy()

    # ---------- 5. Koherensi dan perbandingan dengan LDA ----------
    cv_bt = hitung_cv([kt[t] for t in ids_topik], tokens, kamus, args.proses)
    ukuran = pd.Series(topics[topics != -1]).value_counts()
    perb = [{"metode": f"BERTopic (min_cluster={min_cluster}" + (f", nr_topics={args.nr_topics}" if args.nr_topics else "") + ")",
             "n_topik": n_topik, "c_v": round(cv_bt, 4), "diversitas": round(diversitas([kt[t] for t in ids_topik]), 3),
             "persen_ulasan_tak_terpetakan": round(persen_out, 1), "ukuran_topik_median": int(ukuran.median())}]

    path_lda = MODEL_DIR / f"lda_k{args.lda_k}_{args.lda_tag}.model"
    f_lda_dok = TAB_DIR / f"lda_topik_dokumen_k{args.lda_k}_{args.lda_tag}.csv"
    lda_dok = None
    if path_lda.exists():
        lda = LdaModel.load(str(path_lda))
        kata_lda = [[w for w, _ in lda.show_topic(t, topn=TOP_N_KATA)] for t in range(args.lda_k)]
        cv_lda = hitung_cv(kata_lda, tokens, kamus, args.proses)
        perb.insert(0, {"metode": f"LDA (k={args.lda_k}, {args.lda_tag})", "n_topik": args.lda_k,
                        "c_v": round(cv_lda, 4), "diversitas": round(diversitas(kata_lda), 3),
                        "persen_ulasan_tak_terpetakan": np.nan, "ukuran_topik_median": np.nan})
    else:
        print(f"[PERHATIAN] Model LDA {path_lda.name} tidak ditemukan; perbandingan C_v dengan LDA dilewati. "
              f"Buat dengan: python tahap5_lda.py --no-below 5 --min-token 2 --tag {args.lda_tag} --k {args.lda_k}")
    if f_lda_dok.exists():
        lda_dok = pd.read_csv(f_lda_dok, usecols=["reviewId", "topik_dominan"], dtype={"reviewId": str})
    perb = pd.DataFrame(perb)
    perb.to_csv(TAB_DIR / f"perbandingan_lda_bertopic{sfx}.csv", index=False, encoding="utf-8")
    print("\n=== PERBANDINGAN (korpus dan kamus acuan C_v yang sama) ===")
    print(perb.to_string(index=False))
    print("(C_v bergantung seed; bandingkan dengan rentang antar-seed, bukan angka tunggal. "
          "Interpretabilitas dinilai manual lewat templat penamaan.)")

    # ---------- 6. Keluaran ----------
    dok = d[["reviewId", "app", "label_final"]].copy()
    dok["topik"] = topics
    dok["topik_tanpa_outlier"] = topics_red
    dok.to_csv(TAB_DIR / f"bertopic_topik_dokumen{sfx}.csv", index=False, encoding="utf-8")

    info = pd.DataFrame({"topik": ids_topik, "ukuran": [int((topics == t).sum()) for t in ids_topik]})
    info["persen"] = (100 * info["ukuran"] / len(d)).round(2)
    info["kata_teratas"] = [", ".join(kt[t]) for t in ids_topik]
    info.sort_values("ukuran", ascending=False).to_csv(TAB_DIR / f"bertopic_topik{sfx}.csv", index=False, encoding="utf-8")

    # padanan dengan LDA (berdasarkan ulasan yang sama)
    padanan, nmi = {}, np.nan
    if lda_dok is not None:
        gab = dok.merge(lda_dok, on="reviewId", how="inner")
        gab = gab[gab["topik"] != -1]
        if len(gab):
            silang = pd.crosstab(gab["topik"], gab["topik_dominan"])
            silang.to_csv(TAB_DIR / f"silang_bertopic_lda{sfx}.csv", encoding="utf-8")
            for t in silang.index:
                j = int(silang.loc[t].idxmax())
                padanan[t] = f"LDA T{j} ({100 * silang.loc[t, j] / silang.loc[t].sum():.0f}%)"
            nmi = normalized_mutual_info_score(gab["topik"], gab["topik_dominan"])
            print(f"\nKemiripan pengelompokan BERTopic vs LDA (NMI, ulasan non-outlier): {nmi:.3f} "
                  f"(0 = tak berkaitan, 1 = identik; kedua metode memakai granularitas berbeda)")

    # templat penamaan: kata teratas + ulasan paling dekat dengan pusat klaster
    penamaan = []
    for t in ids_topik:
        idx = np.flatnonzero(topics == t)
        pusat = emb[idx].mean(axis=0)
        pusat = pusat / (np.linalg.norm(pusat) + 1e-12)
        calon = idx[n_tok[idx] >= 5] if (n_tok[idx] >= 5).sum() >= 3 else idx
        terdekat = calon[np.argsort(-(emb[calon] @ pusat))][:3]
        contoh = [d.loc[i, "teks_asli"].replace("\n", " ")[:180] for i in terdekat] + [""] * 3
        penamaan.append({"topik": t, "ukuran": len(idx), "persen": round(100 * len(idx) / len(d), 2),
                         "kata_teratas": ", ".join(kt[t]), "padanan_LDA": padanan.get(t, ""),
                         "contoh_1": contoh[0], "contoh_2": contoh[1], "contoh_3": contoh[2],
                         "nama_manual": "", "catatan": ""})
    pd.DataFrame(penamaan).sort_values("ukuran", ascending=False) \
        .to_csv(TAB_DIR / f"bertopic_penamaan{sfx}.csv", index=False, encoding="utf-8")

    per_app = pd.crosstab(dok["topik_tanpa_outlier"], dok["app"], normalize="columns")
    per_app.round(4).to_csv(TAB_DIR / f"bertopic_topik_per_app{sfx}.csv", encoding="utf-8")
    out_app = (dok.assign(out=dok["topik"] == -1).groupby("app")["out"].mean() * 100).round(1)
    out_lab = (dok.assign(out=dok["topik"] == -1).groupby("label_final")["out"].mean() * 100).round(1)
    out_app.to_csv(TAB_DIR / f"bertopic_outlier_per_app{sfx}.csv", header=["persen_outlier"], encoding="utf-8")

    # gambar ukuran topik
    top = info.sort_values("ukuran", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(top) + 1.5))
    ax.barh([f"T{t}" for t in top["topik"]][::-1], top["persen"].values[::-1], color="#2a9d8f")
    ax.set_xlabel("% ulasan")
    ax.set_title(f"Ukuran topik BERTopic (outlier = {persen_out:.0f}% ulasan, tidak digambar)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"bertopic_ukuran_topik{sfx}.png", dpi=200)
    plt.close(fig)

    try:
        tm.save(str(MODEL_DIR / f"bertopic{sfx}"), serialization="safetensors", save_ctfidf=True,
                save_embedding_model=MODEL_EMBEDDING)
    except Exception as e:
        print(f"[INFO] Model BERTopic tidak disimpan ({type(e).__name__}); hasil CSV tetap lengkap.")

    (TAB_DIR / f"bertopic_parameter{sfx}.txt").write_text(
        f"model_embedding: {MODEL_EMBEDDING}\numap: n_neighbors={UMAP_N_NEIGHBORS}, n_components={UMAP_N_COMPONENTS}, "
        f"min_dist={UMAP_MIN_DIST}, metric=cosine, random_state={SEED}\n"
        f"hdbscan: min_cluster_size={min_cluster}, min_samples={MIN_SAMPLES}, metode=eom, "
        f"paket={'hdbscan' if ADA_HDBSCAN_PKG else 'scikit-learn'}\nnr_topics: {args.nr_topics}\n"
        f"kosakata_topik: min_df={CV_MIN_DF}, max_df={CV_MAX_DF}\nn_ulasan: {len(d)}\nn_topik: {n_topik}\n"
        f"persen_outlier: {persen_out:.1f}\nc_v: {cv_bt:.4f}\nnmi_vs_lda: {nmi}\n", encoding="utf-8")

    print(f"\n=== 15 TOPIK TERBESAR ===")
    for _, r in info.sort_values("ukuran", ascending=False).head(15).iterrows():
        print(f"T{int(r['topik']):<3}({r['persen']:5.1f}%): {r['kata_teratas']}")
    print("\nOutlier menurut aplikasi (%):", out_app.to_dict())
    print("Outlier menurut label (%)   :", out_lab.to_dict())
    print(f"\nTemplat penamaan: {TAB_DIR / f'bertopic_penamaan{sfx}.csv'}  (isi kolom nama_manual)")
    if args.uji:
        print("[MODE UJI] Bila berjalan tanpa error, jalankan tanpa --uji.")


if __name__ == "__main__":
    main()
