"""
tahap9_embedding.py  -  Tahap 9: klasifikasi kelompok 2, Word2Vec / FastText (dirata-ratakan) + SVM.

Letakkan di sebelah folder penelitian-ojol/ atau di dalamnya, lalu jalankan:

    python tahap9_embedding.py --uji       # uji cepat: JALANKAN INI DULU
    python tahap9_embedding.py             # proses sebenarnya

Opsi:
    --dim 100        dimensi vektor kata
    --epochs 30      epoch pelatihan embedding (korpus kecil: epoch banyak membantu)
    --kernel linear  linear (cepat, default) atau rbf (jauh lebih lambat; grid dikecilkan)
    --jobs 2  --cv 5

Alur (data uji TIDAK dipakai untuk memilih apa pun):
  1. memuat pembagian Tahap 7 dan memverifikasi md5
  2. melatih Word2Vec dan FastText HANYA pada teks data latih (tanpa label; val/uji tidak ikut)
  3. vektor ulasan = rata-rata vektor kata (Word2Vec: kata yang tidak dikenal diabaikan;
     FastText: kata baru tetap mendapat vektor lewat subkata)
  4. SVM dengan standarisasi fitur; tuning C dengan validasi silang pada DATA LATIH; tiga strategi
     ketidakseimbangan (tanpa / bobot kelas / oversampling di dalam lipatan)
  5. strategi terbaik per embedding dipilih dari macro-F1 DATA VALIDASI; evaluasi akhir pada data uji

Output: outputs/models/emb_*, outputs/predictions/emb_*_{val,test}.csv, outputs/tables/emb_*.csv,
        outputs/figures/emb_*_matriks_uji.png
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
import zlib
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
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")
try:
    from gensim.models import FastText, Word2Vec
    import gensim
except ImportError as e:
    sys.exit(f"gensim belum terpasang atau bentrok ({e}). Coba: pip install -U gensim")

from config import *            # noqa: F401,F403  (SEED, SPLIT_DIR, MODEL_DIR, TAB_DIR, FIG_DIR, ID2LABEL, ...)
from utils import set_seed, make_dirs

warnings.filterwarnings("ignore", category=UserWarning)
PRED_DIR = BASE_DIR / "outputs" / "predictions"
KELAS = [0, 1, 2]
NAMA_KELAS = [ID2LABEL[k] for k in KELAS]
N_BOOTSTRAP = 1000
STRATEGI = ["tanpa", "bobot_kelas", "oversampling"]
KATA_CONTOH = ["bagus", "lama", "mahal", "driver", "error", "ramah", "tarif", "batal"]
NAMA_EMB = {"w2v": "Word2Vec (skip-gram)", "ft": "FastText (skip-gram)"}


def hash_tetap(s):
    """Fungsi hash yang tidak bergantung PYTHONHASHSEED agar inisialisasi bobot bisa direproduksi."""
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


# --------------------------------------------------------------------------
# Oversampling acak, di dalam tiap lipatan (sama dengan Tahap 8)
# --------------------------------------------------------------------------
class ClassifierDenganOversampling(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, seed=42):
        self.estimator = estimator
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        kelas, jumlah = np.unique(y, return_counts=True)
        target = jumlah.max()
        rng = np.random.default_rng(self.seed)
        idx = []
        for k, n in zip(kelas, jumlah):
            ik = np.flatnonzero(y == k)
            tambah = rng.choice(ik, size=target - n, replace=True) if n < target else np.array([], dtype=int)
            idx.append(np.concatenate([ik, tambah]))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        self.estimator_ = clone(self.estimator).fit(X[idx], y[idx])
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def muat_split():
    f_split, f_info = SPLIT_DIR / "split.csv", SPLIT_DIR / "split_info.json"
    if not (f_split.exists() and f_info.exists()):
        sys.exit("Pembagian data belum ada. Jalankan dulu: python tahap7_split.py")
    info = json.loads(f_info.read_text(encoding="utf-8"))
    md5 = hashlib.md5(f_split.read_bytes()).hexdigest()
    if md5 != info["md5_split_csv"]:
        sys.exit("split.csv berubah sejak dibuat (md5 tidak cocok dengan split_info.json). "
                 "Semua model harus memakai pembagian yang sama; pulihkan dari cadangan.")
    split = pd.read_csv(f_split, dtype={"reviewId": str})
    data = {}
    for nama in ("train", "val", "test"):
        d = pd.read_csv(SPLIT_DIR / f"{nama}.csv", keep_default_na=False,
                        dtype={"reviewId": str, "teks_bersih": str})
        if set(d["reviewId"]) != set(split.loc[split["split"] == nama, "reviewId"]):
            sys.exit(f"{nama}.csv tidak sesuai dengan split.csv.")
        data[nama] = d.reset_index(drop=True)
    print(f"Pembagian diverifikasi (md5 {md5[:10]}...): "
          f"latih {len(data['train']):,}, validasi {len(data['val']):,}, uji {len(data['test']):,}")
    return data, md5


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
def latih_embedding(jenis, kalimat, dim, epochs):
    umum = dict(sentences=kalimat, vector_size=dim, window=5, min_count=2, sg=1, negative=5, epochs=epochs,
                workers=1, seed=SEED, hashfxn=hash_tetap)      # workers=1 agar hasil bisa direproduksi
    if jenis == "w2v":
        return Word2Vec(**umum)
    return FastText(min_n=3, max_n=6, bucket=200000, **umum)    # bucket kecil: korpus kecil, hemat memori (bawaan 2 juta)


def vektor_ulasan(daftar_token, model, jenis, dim):
    """Rata-rata vektor kata. Word2Vec: kata di luar kosakata diabaikan. FastText: semua kata dihitung."""
    wv = model.wv
    hasil = np.zeros((len(daftar_token), dim), dtype=np.float32)
    kosong = 0
    for i, tok in enumerate(daftar_token):
        v = [wv[w] for w in tok] if jenis == "ft" else [wv[w] for w in tok if w in wv.key_to_index]
        if v:
            hasil[i] = np.mean(v, axis=0)
        else:
            kosong += 1
    return hasil, kosong


def tetangga(model, jenis, n=5):
    wv = model.wv
    baris = []
    for w in KATA_CONTOH:
        if w in wv.key_to_index:
            baris.append({"embedding": jenis, "kata": w, "tetangga": ", ".join(k for k, _ in wv.most_similar(w, topn=n))})
    if jenis == "ft":
        try:   # kata salah ketik/langka yang tidak ada di kosakata: FastText tetap bisa membuat vektor
            baris.append({"embedding": jenis, "kata": "bagusss (di luar kosakata)",
                          "tetangga": ", ".join(k for k, _ in wv.most_similar("bagusss", topn=n))})
        except Exception:
            pass
    return baris


# --------------------------------------------------------------------------
# Model dan evaluasi
# --------------------------------------------------------------------------
def buat_pipeline(strategi, kernel):
    if kernel == "linear":
        dasar = LinearSVC(C=1.0, class_weight="balanced" if strategi == "bobot_kelas" else None, dual=False,
                          max_iter=5000, random_state=SEED)
    else:
        dasar = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced" if strategi == "bobot_kelas" else None,
                    random_state=SEED)
    clf = ClassifierDenganOversampling(dasar, SEED) if strategi == "oversampling" else dasar
    return Pipeline([("skala", StandardScaler()), ("clf", clf)])


def buat_grid(strategi, kernel, uji):
    awal = "clf__estimator__" if strategi == "oversampling" else "clf__"
    if kernel == "linear":
        return {awal + "C": [0.01, 0.1] if uji else [0.001, 0.01, 0.1, 1.0]}
    return {awal + "C": [1.0] if uji else [1.0, 10.0]}


def metrik(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=KELAS, zero_division=0)
    hasil = {"akurasi": accuracy_score(y, p), "akurasi_seimbang": balanced_accuracy_score(y, p),
             "presisi_macro": pr.mean(), "recall_macro": rc.mean(), "f1_macro": f1.mean(),
             "f1_weighted": f1_score(y, p, average="weighted", zero_division=0)}
    for i, n in enumerate(NAMA_KELAS):
        hasil[f"presisi_{n}"], hasil[f"recall_{n}"], hasil[f"f1_{n}"] = pr[i], rc[i], f1[i]
    return hasil


def f1_macro_fn(y, p):
    return f1_score(y, p, labels=KELAS, average="macro", zero_division=0)


def f1_netral_fn(y, p):
    return f1_score(y, p, labels=[1], average="macro", zero_division=0)


def bootstrap_ci(y, p, fungsi, n=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    nilai = [fungsi(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return tuple(np.nanpercentile(nilai, [2.5, 97.5]))


def gambar_matriks(cm, judul, path):
    persen = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(persen, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3), NAMA_KELAS)
    ax.set_yticks(range(3), NAMA_KELAS)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title(judul, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({persen[i, j]:.0f}%)", ha="center", va="center", fontsize=9,
                    color="white" if persen[i, j] > 50 else "black")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Tahap 9: Word2Vec/FastText + SVM")
    ap.add_argument("--uji", action="store_true")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    make_dirs()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    sfx = "_uji" if args.uji else ""
    epochs = 5 if args.uji else args.epochs
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    pd.set_option("display.width", 220)

    data, md5 = muat_split()
    tr, va, te = data["train"], data["val"], data["test"]
    if args.uji:
        tr = tr.sample(n=min(3000, len(tr)), random_state=SEED).reset_index(drop=True)
        print(f"[MODE UJI] data latih dikecilkan menjadi {len(tr)}")
    tok = {n: d["teks_bersih"].map(str.split).tolist() for n, d in (("train", tr), ("val", va), ("test", te))}
    ytr, yva, yte = (d["label_id"].astype(int).to_numpy() for d in (tr, va, te))
    print("Distribusi latih:", {n: int(v) for n, v in zip(NAMA_KELAS, np.bincount(ytr, minlength=3))})
    cv = StratifiedKFold(n_splits=3 if args.uji else args.cv, shuffle=True, random_state=SEED)
    if args.kernel == "rbf":
        print("[INFO] kernel rbf dipilih: pelatihan jauh lebih lambat daripada linear.")

    cari, model_cari, emb_model, tetangga_baris = [], {}, {}, []
    X = {}
    for jenis in ("w2v", "ft"):
        t0 = time.perf_counter()
        # ---------- 1. Latih embedding HANYA pada teks data latih ----------
        m = latih_embedding(jenis, tok["train"], args.dim, epochs)
        emb_model[jenis] = m
        vocab = m.wv.key_to_index
        tok_val = [w for t in tok["val"] for w in t]
        oov = np.mean([w not in vocab for w in tok_val])
        print(f"\n=== {NAMA_EMB[jenis]}: kosakata {len(vocab):,} | dimensi {args.dim} | epoch {epochs} | "
              f"{time.perf_counter() - t0:.0f} dtk ===")
        print(f"Token validasi di luar kosakata latih: {100 * oov:.1f}% "
              f"({'diabaikan' if jenis == 'w2v' else 'tetap dipetakan lewat subkata'})")
        tetangga_baris += tetangga(m, jenis)
        for r in tetangga_baris[-3:]:
            print(f"  {r['kata']:>28s} -> {r['tetangga']}")
        # ---------- 2. Vektor ulasan ----------
        X[jenis] = {}
        for nama in ("train", "val", "test"):
            X[jenis][nama], kosong = vektor_ulasan(tok[nama], m, jenis, args.dim)
            if kosong:
                print(f"  {nama}: {kosong} ulasan tanpa kata yang dikenal -> vektor nol")
        # ---------- 3. Tuning (CV pada data latih) dan seleksi strategi (validasi) ----------
        for strategi in STRATEGI:
            t1 = time.perf_counter()
            gs = GridSearchCV(buat_pipeline(strategi, args.kernel), buat_grid(strategi, args.kernel, args.uji),
                              scoring="f1_macro", cv=cv, n_jobs=args.jobs, refit=True)
            gs.fit(X[jenis]["train"], ytr)
            pv = gs.predict(X[jenis]["val"])
            mv = metrik(yva, pv)
            cari.append({"embedding": jenis, "strategi": strategi, "param_terbaik": json.dumps(gs.best_params_),
                         "cv_f1_macro": gs.best_score_, "cv_sd": gs.cv_results_["std_test_score"][gs.best_index_],
                         "val_f1_macro": mv["f1_macro"], "val_akurasi": mv["akurasi"],
                         "val_f1_netral": mv["f1_netral"], "detik": round(time.perf_counter() - t1, 1)})
            model_cari[(jenis, strategi)] = (gs.best_estimator_, pv)
            print(f"{jenis:3s} | {strategi:12s} | CV {cari[-1]['cv_f1_macro']:.4f} (sd {cari[-1]['cv_sd']:.4f}) | "
                  f"validasi macro-F1 {mv['f1_macro']:.4f}, F1 netral {mv['f1_netral']:.4f} | "
                  f"{gs.best_params_} | {cari[-1]['detik']:.0f} dtk")
    cari = pd.DataFrame(cari)
    cari.round(4).to_csv(TAB_DIR / f"emb_perbandingan_strategi{sfx}.csv", index=False, encoding="utf-8")
    pd.DataFrame(tetangga_baris).to_csv(TAB_DIR / f"emb_tetangga_kata{sfx}.csv", index=False, encoding="utf-8")

    terpilih = {}
    for jenis in ("w2v", "ft"):
        sub = cari[cari["embedding"] == jenis]
        terpilih[jenis] = sub.loc[sub["val_f1_macro"].idxmax(), "strategi"]
    print("\nStrategi terpilih menurut macro-F1 DATA VALIDASI: " + ", ".join(f"{a} -> {s}" for a, s in terpilih.items()))

    # ---------- 4. Evaluasi akhir ----------
    akhir, per_app = [], []
    for jenis in ("w2v", "ft"):
        strategi = terpilih[jenis]
        pipe, pv = model_cari[(jenis, strategi)]
        pt = pipe.predict(X[jenis]["test"])
        emb_model[jenis].save(str(MODEL_DIR / f"emb_{jenis}{sfx}.model"))
        joblib.dump(pipe, MODEL_DIR / f"emb_{jenis}_svm{sfx}.joblib")
        for nama, y, p, d in (("val", yva, pv, va), ("test", yte, pt, te)):
            pd.DataFrame({"reviewId": d["reviewId"], "y_true": y, "y_pred": p}) \
                .to_csv(PRED_DIR / f"emb_{jenis}_svm_{nama}{sfx}.csv", index=False, encoding="utf-8")
            mm = metrik(y, p)
            lo, hi = bootstrap_ci(y, p, f1_macro_fn)
            nlo, nhi = bootstrap_ci(y, p, f1_netral_fn)
            akhir.append({"model": f"{NAMA_EMB[jenis]} + SVM ({args.kernel})", "embedding": jenis,
                          "strategi": strategi, "split": nama, **mm, "f1_macro_ci_bawah": lo,
                          "f1_macro_ci_atas": hi, "f1_netral_ci_bawah": nlo, "f1_netral_ci_atas": nhi})
            cm = confusion_matrix(y, p, labels=KELAS)
            pd.DataFrame(cm, index=NAMA_KELAS, columns=NAMA_KELAS).to_csv(
                TAB_DIR / f"emb_{jenis}_matriks_{nama}{sfx}.csv", encoding="utf-8")
            if nama == "test":
                gambar_matriks(cm, f"{NAMA_EMB[jenis]} + SVM\n(data uji, strategi: {strategi})",
                               FIG_DIR / f"emb_{jenis}_matriks_uji{sfx}.png")
        for a, g in te.assign(y=yte, p=pt).groupby("app"):
            per_app.append({"model": NAMA_EMB[jenis], "app": a, "n": len(g), "n_netral": int((g["y"] == 1).sum()),
                            "f1_macro": f1_macro_fn(g["y"], g["p"]), "f1_netral": f1_netral_fn(g["y"], g["p"]),
                            "akurasi": accuracy_score(g["y"], g["p"])})
    akhir = pd.DataFrame(akhir)
    akhir.round(4).to_csv(TAB_DIR / f"emb_hasil_akhir{sfx}.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_app).round(4).to_csv(TAB_DIR / f"emb_per_aplikasi_uji{sfx}.csv", index=False, encoding="utf-8")

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== HASIL AKHIR ===")
    print(akhir[["model", "strategi", "split", "akurasi", "presisi_macro", "recall_macro", "f1_macro",
                 "f1_negatif", "f1_netral", "f1_positif"]].to_string(index=False))
    print("\nSelang kepercayaan 95% (bootstrap) pada DATA UJI:")
    for _, r in akhir[akhir["split"] == "test"].iterrows():
        print(f"  {r['embedding']:3s}: macro-F1 {r['f1_macro']:.4f} ({r['f1_macro_ci_bawah']:.4f} - {r['f1_macro_ci_atas']:.4f}) | "
              f"F1 netral {r['f1_netral']:.4f} ({r['f1_netral_ci_bawah']:.4f} - {r['f1_netral_ci_atas']:.4f})")
    val = akhir[akhir["split"] == "val"].set_index("embedding")["f1_macro"]
    wakil = val.idxmax()
    print(f"\nPerwakilan kelompok 2 (macro-F1 validasi tertinggi): {NAMA_EMB[wakil]} ({val[wakil]:.4f}); "
          f"yang lain: {val.drop(wakil).iloc[0]:.4f}. Kedua varian tetap dilaporkan.")
    print("\nMacro-F1 per aplikasi (data uji; netral per aplikasi sangat sedikit, tafsirkan hati-hati):")
    print(pd.DataFrame(per_app).to_string(index=False))

    (TAB_DIR / f"emb_parameter{sfx}.json").write_text(json.dumps({
        "md5_split": md5, "seed": SEED, "gensim": gensim.__version__, "sklearn": sklearn.__version__,
        "dim": args.dim, "epochs": epochs, "window": 5, "min_count": 2, "sg": 1, "negative": 5,
        "fasttext_ngram": [3, 6], "kernel": args.kernel, "cv_lipatan": cv.n_splits,
        "strategi_terpilih": terpilih, "perwakilan": wakil, "uji_coba": args.uji,
        "embedding_dilatih_pada": "data latih saja"}, indent=2), encoding="utf-8")
    print(f"\nTabel di {TAB_DIR}, prediksi di {PRED_DIR}, model di {MODEL_DIR}")
    if args.uji:
        print("[MODE UJI] Angka di atas tidak bermakna; bila berjalan tanpa error, jalankan tanpa --uji.")


if __name__ == "__main__":
    main()
