"""
tahap8_tfidf.py  -  Tahap 8: klasifikasi kelompok 1, TF-IDF (unigram+bigram) dengan SVM dan Naive Bayes.

Letakkan di sebelah folder penelitian-ojol/ atau di dalamnya, lalu jalankan:

    python tahap8_tfidf.py --uji      # uji cepat (3.000 data latih, grid kecil): JALANKAN INI DULU
    python tahap8_tfidf.py            # proses sebenarnya

Opsi:
    --jobs 2        jumlah proses paralel untuk validasi silang (naikkan bila komputer kuat)
    --cv 5          jumlah lipatan validasi silang

Alur (data uji TIDAK dipakai untuk memilih apa pun):
  1. memuat pembagian dari Tahap 7 dan memverifikasi tanda tangan md5
  2. untuk tiap algoritma (SVM, Naive Bayes) dan tiap strategi ketidakseimbangan
     (tanpa / bobot kelas / oversampling), tuning hiperparameter dengan validasi silang pada DATA LATIH
     (TF-IDF dan oversampling dilakukan di dalam tiap lipatan, hanya pada bagian latih lipatan)
  3. memilih strategi terbaik per algoritma berdasarkan macro-F1 pada DATA VALIDASI
  4. mengevaluasi model terpilih pada data uji: akurasi, presisi, recall, F1, macro-F1, matriks konfusi

Input : data/splits/{train,val,test}.csv, split.csv, split_info.json (Tahap 7); kolom teks_bersih, label_id
Output: outputs/models/tfidf_svm.joblib, tfidf_nb.joblib
        outputs/predictions/tfidf_{svm,nb}_{val,test}.csv   (reviewId, y_true, y_pred) untuk Tahap 11-12
        outputs/tables/tfidf_*.csv, outputs/figures/tfidf_*.png
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")

from config import *            # noqa: F401,F403  (SEED, SPLIT_DIR, MODEL_DIR, TAB_DIR, FIG_DIR, ID2LABEL, ...)
from utils import set_seed, make_dirs

warnings.filterwarnings("ignore", category=UserWarning)
PRED_DIR = BASE_DIR / "outputs" / "predictions"
KELAS = [0, 1, 2]
NAMA_KELAS = [ID2LABEL[k] for k in KELAS]
N_BOOTSTRAP = 1000


# --------------------------------------------------------------------------
# Oversampling acak, dilakukan DI DALAM tiap lipatan (hanya pada data latih lipatan)
# --------------------------------------------------------------------------
class ClassifierDenganOversampling(BaseEstimator, ClassifierMixin):
    """Menggandakan contoh kelas minoritas (acak, dengan pengembalian) hingga sama dengan kelas terbesar,
    hanya saat fit. Saat prediksi, tidak ada yang berubah. Setara RandomOverSampler tanpa dependensi tambahan."""

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

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# --------------------------------------------------------------------------
# Data dan verifikasi pembagian
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
    return data, info, md5


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
STRATEGI = ["tanpa", "bobot_kelas", "oversampling"]


def buat_pipeline(algo, strategi):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"\S+", lowercase=False, sublinear_tf=True,
                            max_df=0.95, dtype=np.float32)
    if algo == "svm":
        dasar = LinearSVC(C=1.0, class_weight="balanced" if strategi == "bobot_kelas" else None, dual=True,
                          max_iter=5000, random_state=SEED)
    else:   # Naive Bayes: 'bobot_kelas' = prior kelas seragam (fit_prior=False)
        dasar = MultinomialNB(fit_prior=(strategi != "bobot_kelas"))
    clf = ClassifierDenganOversampling(dasar, SEED) if strategi == "oversampling" else dasar
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def buat_grid(algo, strategi, uji):
    awal = "clf__estimator__" if strategi == "oversampling" else "clf__"
    g = {"tfidf__min_df": [2] if uji else [2, 5]}
    if algo == "svm":
        g[awal + "C"] = [0.1, 1.0] if uji else [0.03, 0.1, 0.3, 1.0, 3.0]
    else:
        g[awal + "alpha"] = [0.1, 1.0] if uji else [0.03, 0.1, 0.3, 1.0]
    return g


def metrik(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=KELAS, zero_division=0)
    hasil = {"akurasi": accuracy_score(y, p), "akurasi_seimbang": balanced_accuracy_score(y, p),
             "presisi_macro": pr.mean(), "recall_macro": rc.mean(), "f1_macro": f1.mean(),
             "f1_weighted": f1_score(y, p, average="weighted", zero_division=0)}
    for i, n in enumerate(NAMA_KELAS):
        hasil[f"presisi_{n}"], hasil[f"recall_{n}"], hasil[f"f1_{n}"] = pr[i], rc[i], f1[i]
    return hasil


def bootstrap_ci(y, p, fungsi, n=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    nilai = [fungsi(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return tuple(np.nanpercentile(nilai, [2.5, 97.5]))


def f1_macro_fn(y, p):
    return f1_score(y, p, labels=KELAS, average="macro", zero_division=0)


def f1_netral_fn(y, p):
    return f1_score(y, p, labels=[1], average="macro", zero_division=0)


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


def fitur_teratas(pipe, algo, n=15):
    tf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    clf = getattr(clf, "estimator_", clf)
    nama = np.array(tf.get_feature_names_out())
    if algo == "svm":
        skor = clf.coef_                                       # (3, fitur): satu lawan sisanya
    else:
        lp = clf.feature_log_prob_
        skor = np.stack([lp[k] - np.delete(lp, k, axis=0).mean(axis=0) for k in range(lp.shape[0])])
    baris = []
    for i, k in enumerate(clf.classes_):
        for r, j in enumerate(np.argsort(-skor[i])[:n], 1):
            baris.append({"model": algo, "kelas": ID2LABEL[int(k)], "peringkat": r, "fitur": nama[j],
                          "skor": round(float(skor[i, j]), 4)})
    return pd.DataFrame(baris)


def main():
    ap = argparse.ArgumentParser(description="Tahap 8: TF-IDF + SVM / Naive Bayes")
    ap.add_argument("--uji", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    make_dirs()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    sfx = "_uji" if args.uji else ""
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    pd.set_option("display.width", 220)

    data, info, md5 = muat_split()
    tr, va, te = data["train"], data["val"], data["test"]
    if args.uji:
        tr = tr.sample(n=min(3000, len(tr)), random_state=SEED).reset_index(drop=True)
        print(f"[MODE UJI] data latih dikecilkan menjadi {len(tr)}")
    Xtr, ytr = tr["teks_bersih"], tr["label_id"].astype(int).to_numpy()
    Xva, yva = va["teks_bersih"], va["label_id"].astype(int).to_numpy()
    Xte, yte = te["teks_bersih"], te["label_id"].astype(int).to_numpy()
    print("Distribusi latih:", {n: int(v) for n, v in zip(NAMA_KELAS, np.bincount(ytr, minlength=3))})
    cv = StratifiedKFold(n_splits=3 if args.uji else args.cv, shuffle=True, random_state=SEED)

    # ---------- 1. Tuning (CV pada data latih) dan pemilihan strategi (data validasi) ----------
    hasil_cari, model_cari = [], {}
    print(f"\n=== TUNING (validasi silang {cv.n_splits} lipatan pada data latih; skor = macro-F1) ===")
    for algo in ("svm", "nb"):
        for strategi in STRATEGI:
            t0 = time.perf_counter()
            gs = GridSearchCV(buat_pipeline(algo, strategi), buat_grid(algo, strategi, args.uji),
                              scoring="f1_macro", cv=cv, n_jobs=args.jobs, refit=True)
            gs.fit(Xtr, ytr)
            pv = gs.predict(Xva)
            m = metrik(yva, pv)
            baris = {"algoritma": algo, "strategi": strategi, "param_terbaik": json.dumps(gs.best_params_),
                     "cv_f1_macro": gs.best_score_,
                     "cv_sd": gs.cv_results_["std_test_score"][gs.best_index_],
                     "val_f1_macro": m["f1_macro"], "val_akurasi": m["akurasi"], "val_f1_netral": m["f1_netral"],
                     "detik": round(time.perf_counter() - t0, 1)}
            hasil_cari.append(baris)
            model_cari[(algo, strategi)] = (gs.best_estimator_, pv)
            print(f"{algo:3s} | {strategi:12s} | CV {baris['cv_f1_macro']:.4f} (sd {baris['cv_sd']:.4f}) | "
                  f"validasi macro-F1 {m['f1_macro']:.4f}, F1 netral {m['f1_netral']:.4f} | "
                  f"{gs.best_params_} | {baris['detik']:.0f} dtk")
    cari = pd.DataFrame(hasil_cari)
    cari.round(4).to_csv(TAB_DIR / f"tfidf_perbandingan_strategi{sfx}.csv", index=False, encoding="utf-8")

    terpilih = {}
    for algo in ("svm", "nb"):
        sub = cari[cari["algoritma"] == algo]
        terpilih[algo] = sub.loc[sub["val_f1_macro"].idxmax(), "strategi"]
    print("\nStrategi terpilih menurut macro-F1 DATA VALIDASI: " + ", ".join(f"{a} -> {s}" for a, s in terpilih.items()))

    # ---------- 2. Evaluasi akhir ----------
    nama_model = {"svm": "TF-IDF + SVM (linear)", "nb": "TF-IDF + Naive Bayes (multinomial)"}
    akhir, semua_fitur, per_app = [], [], []
    for algo in ("svm", "nb"):
        strategi = terpilih[algo]
        pipe, pv = model_cari[(algo, strategi)]
        pt = pipe.predict(Xte)
        joblib.dump(pipe, MODEL_DIR / f"tfidf_{algo}{sfx}.joblib")
        for nama, y, p, d in (("val", yva, pv, va), ("test", yte, pt, te)):
            pd.DataFrame({"reviewId": d["reviewId"], "y_true": y, "y_pred": p}) \
                .to_csv(PRED_DIR / f"tfidf_{algo}_{nama}{sfx}.csv", index=False, encoding="utf-8")
            m = metrik(y, p)
            lo, hi = bootstrap_ci(y, p, f1_macro_fn)
            nlo, nhi = bootstrap_ci(y, p, f1_netral_fn)
            akhir.append({"model": nama_model[algo], "algoritma": algo, "strategi": strategi, "split": nama,
                          **m, "f1_macro_ci_bawah": lo, "f1_macro_ci_atas": hi,
                          "f1_netral_ci_bawah": nlo, "f1_netral_ci_atas": nhi})
            cm = confusion_matrix(y, p, labels=KELAS)
            pd.DataFrame(cm, index=NAMA_KELAS, columns=NAMA_KELAS).to_csv(
                TAB_DIR / f"tfidf_{algo}_matriks_{nama}{sfx}.csv", encoding="utf-8")
            if nama == "test":
                gambar_matriks(cm, f"{nama_model[algo]}\n(data uji, strategi: {strategi})",
                               FIG_DIR / f"tfidf_{algo}_matriks_uji{sfx}.png")
        semua_fitur.append(fitur_teratas(pipe, algo))
        for a, g in te.assign(y=yte, p=pt).groupby("app"):
            per_app.append({"model": nama_model[algo], "app": a, "n": len(g), "n_netral": int((g["y"] == 1).sum()),
                            "f1_macro": f1_macro_fn(g["y"], g["p"]), "f1_netral": f1_netral_fn(g["y"], g["p"]),
                            "akurasi": accuracy_score(g["y"], g["p"])})

    akhir = pd.DataFrame(akhir)
    akhir.round(4).to_csv(TAB_DIR / f"tfidf_hasil_akhir{sfx}.csv", index=False, encoding="utf-8")
    pd.concat(semua_fitur).to_csv(TAB_DIR / f"tfidf_fitur_teratas{sfx}.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_app).round(4).to_csv(TAB_DIR / f"tfidf_per_aplikasi_uji{sfx}.csv", index=False, encoding="utf-8")

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== HASIL AKHIR ===")
    kolom = ["model", "strategi", "split", "akurasi", "presisi_macro", "recall_macro", "f1_macro",
             "f1_negatif", "f1_netral", "f1_positif"]
    print(akhir[kolom].to_string(index=False))
    print("\nSelang kepercayaan 95% (bootstrap) pada DATA UJI:")
    for _, r in akhir[akhir["split"] == "test"].iterrows():
        print(f"  {r['algoritma']:3s}: macro-F1 {r['f1_macro']:.4f} ({r['f1_macro_ci_bawah']:.4f} - {r['f1_macro_ci_atas']:.4f}) | "
              f"F1 netral {r['f1_netral']:.4f} ({r['f1_netral_ci_bawah']:.4f} - {r['f1_netral_ci_atas']:.4f})")
    print("\nMacro-F1 per aplikasi (data uji; netral per aplikasi sangat sedikit, tafsirkan hati-hati):")
    print(pd.DataFrame(per_app).to_string(index=False))
    f = pd.concat(semua_fitur)
    for algo in ("svm", "nb"):
        print(f"\nFitur teratas {algo} (5 per kelas):")
        for k in NAMA_KELAS:
            print(f"  {k:8s}: " + ", ".join(f[(f['model'] == algo) & (f['kelas'] == k)]['fitur'].head(5)))

    (TAB_DIR / f"tfidf_parameter{sfx}.json").write_text(json.dumps({
        "md5_split": md5, "seed": SEED, "sklearn": sklearn.__version__, "tfidf": "ngram (1,2), min_df dituning, max_df 0.95, sublinear_tf",
        "teks": "teks_bersih", "cv_lipatan": cv.n_splits, "skor_cv": "f1_macro", "strategi_terpilih": terpilih,
        "uji_coba": args.uji}, indent=2), encoding="utf-8")
    print(f"\nTabel di {TAB_DIR}, prediksi di {PRED_DIR}, model di {MODEL_DIR}")
    if args.uji:
        print("[MODE UJI] Angka di atas tidak bermakna; bila berjalan tanpa error, jalankan tanpa --uji.")


if __name__ == "__main__":
    main()
