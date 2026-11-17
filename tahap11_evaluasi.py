"""
tahap11_evaluasi.py  -  Tahap 11: evaluasi dan perbandingan semua model klasifikasi.

Membaca prediksi dari Tahap 8-10 (outputs/predictions/*.csv), lalu:
  - memverifikasi bahwa semua model dievaluasi pada ulasan dan label yang SAMA
  - menyusun tabel metrik (akurasi, presisi, recall, F1, macro-F1) dengan selang kepercayaan bootstrap
  - membandingkan model secara berpasangan (selisih macro-F1 + CI, uji McNemar, koreksi Holm)
  - memilih model terbaik berdasarkan macro-F1 DATA UJI (sesuai proposal) dan menyebut model yang tidak
    berbeda nyata dari yang terbaik
  - matriks konfusi, F1 per kelas, hasil per aplikasi, dan uji ketahanan (subset tervalidasi, tanpa kata 'bintang')
  - tabel kesalahan per ulasan sebagai bahan analisis kesalahan (Tahap 12)

Letakkan di sebelah folder penelitian-ojol/ atau di dalamnya, lalu jalankan:

    python tahap11_evaluasi.py

Prediksi yang dibutuhkan (model yang belum ada dilewati; jalankan ulang setelah IndoBERT selesai):
    outputs/predictions/tfidf_svm_{val,test}.csv      outputs/predictions/tfidf_nb_{val,test}.csv
    outputs/predictions/emb_w2v_svm_{val,test}.csv    outputs/predictions/emb_ft_svm_{val,test}.csv
    outputs/predictions/indobert_{val,test}.csv       (dari Colab, salin dari Google Drive)

Opsi:
    --bootstrap 2000     jumlah ulangan bootstrap
"""
import argparse
import json
import os
import sys
from itertools import combinations
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
    from scipy.stats import binomtest
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")

from config import *            # noqa: F401,F403  (SEED, SPLIT_DIR, TAB_DIR, FIG_DIR, ID2LABEL, ...)
from utils import set_seed, make_dirs

PRED_DIR = BASE_DIR / "outputs" / "predictions"
MODEL = [  # (kode file, nama, kelompok)
    ("tfidf_svm", "TF-IDF + SVM", 1),
    ("tfidf_nb", "TF-IDF + Naive Bayes", 1),
    ("emb_w2v_svm", "Word2Vec + SVM", 2),
    ("emb_ft_svm", "FastText + SVM", 2),
    ("indobert", "IndoBERT", 3),
]
KELAS = [0, 1, 2]
NAMA_KELAS = [ID2LABEL[k] for k in KELAS]
KOLOM_METRIK = ["f1_macro", "f1_negatif", "f1_netral", "f1_positif", "akurasi"]
WARNA = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2a9d8f"}


# --------------------------------------------------------------------------
# Metrik cepat dari matriks konfusi
# --------------------------------------------------------------------------
def cm3(y, p):
    return np.bincount(3 * y + p, minlength=9).reshape(3, 3)


def prf_dari_cm(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
    presisi = np.divide(tp, tp + fp, out=np.zeros(3), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(3), where=(tp + fn) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(3), where=(2 * tp + fp + fn) > 0)
    return presisi, recall, f1


def metrik(y, p):
    cm = cm3(y, p)
    pr, rc, f1 = prf_dari_cm(cm)
    n = cm.sum(axis=1)
    return {"akurasi": np.trace(cm) / cm.sum(), "akurasi_seimbang": rc.mean(), "presisi_macro": pr.mean(),
            "recall_macro": rc.mean(), "f1_macro": f1.mean(),
            "f1_weighted": float((f1 * n).sum() / n.sum()),
            **{f"presisi_{k}": pr[i] for i, k in enumerate(NAMA_KELAS)},
            **{f"recall_{k}": rc[i] for i, k in enumerate(NAMA_KELAS)},
            **{f"f1_{k}": f1[i] for i, k in enumerate(NAMA_KELAS)}}


def bootstrap_semua(y, P, B, seed=SEED):
    """Satu set sampel bootstrap dipakai untuk semua model, sehingga perbandingan berpasangan sahih."""
    rng = np.random.default_rng(seed)
    n = len(y)
    hasil = {m: np.zeros((B, 5)) for m in P}
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for m, p in P.items():
            cm = cm3(yb, p[idx])
            f1 = prf_dari_cm(cm)[2]
            hasil[m][b] = [f1.mean(), f1[0], f1[1], f1[2], np.trace(cm) / n]
    return hasil


def ci(arr):
    return tuple(np.nanpercentile(arr, [2.5, 97.5]))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    urut = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    maks = 0.0
    for rank, i in enumerate(urut):
        maks = max(maks, (m - rank) * p[i])
        adj[i] = min(1.0, maks)
    return adj


# --------------------------------------------------------------------------
# Muat dan verifikasi
# --------------------------------------------------------------------------
def muat_split():
    f_info = SPLIT_DIR / "split_info.json"
    if not f_info.exists():
        sys.exit("Pembagian data belum ada. Selesaikan Tahap 7.")
    info = json.loads(f_info.read_text(encoding="utf-8"))
    ref = {n: pd.read_csv(SPLIT_DIR / f"{n}.csv", keep_default_na=False,
                          dtype={"reviewId": str, "teks_asli": str, "teks_bersih": str}).reset_index(drop=True) for n in ("val", "test")}
    for n in ref:
        ref[n]["tervalidasi"] = ref[n]["tervalidasi"].astype(str).str.lower().eq("true")
        ref[n]["menyebut_bintang"] = ref[n]["menyebut_bintang"].astype(str).str.lower().eq("true") \
            if "menyebut_bintang" in ref[n] else False
    return info, ref


def muat_prediksi(kode, split, ref):
    f = PRED_DIR / f"{kode}_{split}.csv"
    if not f.exists():
        return None
    p = pd.read_csv(f, dtype={"reviewId": str})
    if set(p["reviewId"]) != set(ref["reviewId"]) or p["reviewId"].duplicated().any():
        sys.exit(f"{f.name}: kumpulan reviewId tidak sama dengan data {split} pada pembagian Tahap 7 "
                 f"(mungkin file hasil mode uji atau pembagian berbeda).")
    p = p.set_index("reviewId").loc[ref["reviewId"]]
    if not (p["y_true"].to_numpy() == ref["label_id"].astype(int).to_numpy()).all():
        sys.exit(f"{f.name}: label y_true tidak cocok dengan label_final pada pembagian Tahap 7.")
    return p["y_pred"].astype(int).to_numpy()


def main():
    ap = argparse.ArgumentParser(description="Tahap 11: evaluasi dan perbandingan model")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    make_dirs()
    set_seed(SEED)
    B = args.bootstrap
    pd.set_option("display.width", 220)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    info, ref = muat_split()
    nama = {k: n for k, n, _ in MODEL}
    kelompok = {k: g for k, _, g in MODEL}
    yv, yt = ref["val"]["label_id"].astype(int).to_numpy(), ref["test"]["label_id"].astype(int).to_numpy()
    Pv, Pt = {}, {}
    for k, n, _ in MODEL:
        pv, pt = muat_prediksi(k, "val", ref["val"]), muat_prediksi(k, "test", ref["test"])
        if pv is None or pt is None:
            print(f"[DILEWATI] {n}: prediksi belum ada ({k}_val.csv / {k}_test.csv)")
            continue
        Pv[k], Pt[k] = pv, pt
    if len(Pt) < 2:
        sys.exit("Butuh minimal dua model dengan prediksi untuk dibandingkan.")
    print(f"Model yang dievaluasi: {', '.join(nama[k] for k in Pt)} | uji n = {len(yt):,}, validasi n = {len(yv):,}")

    # jejak: apakah pembagian yang dipakai tiap model sama dengan pembagian sekarang?
    for k, f in (("tfidf_svm", "tfidf_parameter.json"), ("emb_ft_svm", "emb_parameter.json"),
                 ("indobert", "indobert_parameter.json")):
        fp = TAB_DIR / f
        if fp.exists():
            md5 = json.loads(fp.read_text(encoding="utf-8")).get("md5_split")
            print(f"  md5 pembagian pada {f}: {'cocok' if md5 == info['md5_split_csv'] else 'TIDAK COCOK'}")

    # ---------- 1. Tabel metrik + CI bootstrap ----------
    boot_t = bootstrap_semua(yt, Pt, B)
    baris = []
    for split, y, P in (("val", yv, Pv), ("test", yt, Pt)):
        for k in P:
            m = metrik(y, P[k])
            r = {"model": nama[k], "kelompok": kelompok[k], "split": split, **m}
            if split == "test":
                for j, kolom in enumerate(KOLOM_METRIK):
                    lo, hi = ci(boot_t[k][:, j])
                    r[f"{kolom}_ci_bawah"], r[f"{kolom}_ci_atas"] = lo, hi
            baris.append(r)
    tab = pd.DataFrame(baris)
    tab.round(4).to_csv(TAB_DIR / "evaluasi_perbandingan_model.csv", index=False, encoding="utf-8")

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    uji = tab[tab["split"] == "test"].sort_values("f1_macro", ascending=False)
    print("\n=== DATA UJI (diurutkan menurut macro-F1) ===")
    print(uji[["model", "akurasi", "presisi_macro", "recall_macro", "f1_macro", "f1_macro_ci_bawah",
               "f1_macro_ci_atas", "f1_negatif", "f1_netral", "f1_positif"]].to_string(index=False))
    val = tab[tab["split"] == "val"].sort_values("f1_macro", ascending=False)
    print("\n=== DATA VALIDASI (pembanding) ===")
    print(val[["model", "akurasi", "f1_macro", "f1_negatif", "f1_netral", "f1_positif"]].to_string(index=False))

    # ---------- 2. Perbandingan berpasangan ----------
    pas = []
    for a, b in combinations(Pt.keys(), 2):
        d = boot_t[a][:, 0] - boot_t[b][:, 0]
        lo, hi = ci(d)
        p_boot = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        benar_a, benar_b = Pt[a] == yt, Pt[b] == yt
        n10, n01 = int((benar_a & ~benar_b).sum()), int((~benar_a & benar_b).sum())
        p_mc = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) > 0 else 1.0
        pas.append({"model_A": nama[a], "model_B": nama[b],
                    "selisih_f1_macro_A_minus_B": f1m(yt, Pt[a]) - f1m(yt, Pt[b]), "ci_bawah": lo, "ci_atas": hi,
                    "p_bootstrap": p_boot, "A_benar_B_salah": n10, "A_salah_B_benar": n01, "p_mcnemar": p_mc})
    pas = pd.DataFrame(pas)
    pas["p_bootstrap_holm"] = holm(pas["p_bootstrap"])
    pas["p_mcnemar_holm"] = holm(pas["p_mcnemar"])
    pas["beda_nyata_macroF1"] = (pas["ci_bawah"] > 0) | (pas["ci_atas"] < 0)
    pas.round(4).to_csv(TAB_DIR / "evaluasi_uji_berpasangan.csv", index=False, encoding="utf-8")
    print("\n=== PERBANDINGAN BERPASANGAN (data uji) ===")
    print("selisih macro-F1 = A - B; 'beda nyata' bila CI 95% selisih tidak memuat 0")
    print(pas[["model_A", "model_B", "selisih_f1_macro_A_minus_B", "ci_bawah", "ci_atas", "beda_nyata_macroF1",
               "p_mcnemar", "p_mcnemar_holm"]].to_string(index=False))

    # ---------- 3. Model terbaik ----------
    terbaik = uji.iloc[0]
    kode_terbaik = [k for k in Pt if nama[k] == terbaik["model"]][0]
    tidak_beda = []
    for k in Pt:
        if k == kode_terbaik:
            continue
        d = boot_t[kode_terbaik][:, 0] - boot_t[k][:, 0]
        if ci(d)[0] <= 0:
            tidak_beda.append(nama[k])
    kode_val_terbaik = [k for k in Pv if nama[k] == val.iloc[0]["model"]][0]
    print(f"\n=== MODEL TERBAIK (macro-F1 data uji): {terbaik['model']} "
          f"({terbaik['f1_macro']:.4f}; CI {terbaik['f1_macro_ci_bawah']:.4f} - {terbaik['f1_macro_ci_atas']:.4f}) ===")
    if tidak_beda:
        print("Tidak berbeda nyata dari model terbaik (CI selisih memuat 0): " + ", ".join(tidak_beda))
        print("  -> Pertimbangkan biaya komputasi, kemudahan interpretasi, dan stabilitas saat memilih; "
              "tuliskan ini di pembahasan.")
    else:
        print("Model terbaik berbeda nyata dari semua model lain pada macro-F1 (CI selisih tidak memuat 0).")
    print(f"Terbaik menurut data validasi: {nama[kode_val_terbaik]} "
          f"{'(sama)' if kode_val_terbaik == kode_terbaik else '(BERBEDA dari pemenang data uji; laporkan keduanya)'}")

    # ---------- 4. Matriks konfusi semua model ----------
    n_m = len(Pt)
    fig, axs = plt.subplots(1, n_m, figsize=(3.6 * n_m, 3.8), squeeze=False)
    pasangan_salah = []
    for ax, k in zip(axs[0], Pt):
        cm = cm3(yt, Pt[k])
        pd.DataFrame(cm, index=NAMA_KELAS, columns=NAMA_KELAS).to_csv(TAB_DIR / f"evaluasi_matriks_{k}_uji.csv")
        persen = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(persen, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(3), NAMA_KELAS, fontsize=8)
        ax.set_yticks(range(3), NAMA_KELAS, fontsize=8)
        ax.set_title(nama[k], fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{persen[i, j]:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="white" if persen[i, j] > 50 else "black")
                if i != j:
                    pasangan_salah.append({"model": nama[k], "sebenarnya": NAMA_KELAS[i], "prediksi": NAMA_KELAS[j],
                                           "jumlah": int(cm[i, j]),
                                           "persen_dari_kelas": round(float(persen[i, j]), 1),
                                           "persen_dari_semua_kesalahan": round(100 * cm[i, j] / (cm.sum() - np.trace(cm)), 1)})
        ax.grid(False)
        ax.set_xlabel("Prediksi", fontsize=8)
    axs[0][0].set_ylabel("Sebenarnya", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "evaluasi_matriks_semua_model.png", dpi=200)
    plt.close(fig)
    ps = pd.DataFrame(pasangan_salah)
    ps.to_csv(TAB_DIR / "evaluasi_pasangan_kesalahan.csv", index=False, encoding="utf-8")
    print("\nKesalahan terbesar per model (3 pasangan teratas, data uji):")
    for k in Pt:
        t = ps[ps["model"] == nama[k]].sort_values("jumlah", ascending=False).head(3)
        print(f"  {nama[k]:22s}: " + "; ".join(f"{r.sebenarnya}->{r.prediksi} {r.jumlah} ({r.persen_dari_kelas:.0f}% kelas)"
                                              for r in t.itertuples()))

    # ---------- 5. Gambar macro-F1 dan F1 per kelas ----------
    urut = uji["model"].tolist()
    kode_urut = [[k for k in Pt if nama[k] == n][0] for n in urut]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    nilai = [uji.set_index("model").loc[n, "f1_macro"] for n in urut]
    lo = [nilai[i] - uji.set_index("model").loc[n, "f1_macro_ci_bawah"] for i, n in enumerate(urut)]
    hi = [uji.set_index("model").loc[n, "f1_macro_ci_atas"] - nilai[i] for i, n in enumerate(urut)]
    ax.barh(urut[::-1], nilai[::-1], xerr=[lo[::-1], hi[::-1]], color=[WARNA[kelompok[k]] for k in kode_urut[::-1]], capsize=3)
    ax.set_xlabel("Macro-F1 data uji (CI 95% bootstrap; sumbu tidak dimulai dari 0)")
    ax.set_xlim(max(0, min(nilai) - 0.08), min(1, max(nilai) + 0.06))
    ax.set_title("Perbandingan macro-F1 antarmodel (warna = kelompok)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "evaluasi_macro_f1.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    lebar = 0.8 / len(kode_urut)
    for i, k in enumerate(kode_urut):
        nilai_k = [uji.set_index("model").loc[nama[k], f"f1_{c}"] for c in NAMA_KELAS]
        err = [[boot_t[k][:, 1 + j].mean() - ci(boot_t[k][:, 1 + j])[0] for j in range(3)],
               [ci(boot_t[k][:, 1 + j])[1] - boot_t[k][:, 1 + j].mean() for j in range(3)]]
        ax.bar(np.arange(3) + (i - (len(kode_urut) - 1) / 2) * lebar, nilai_k, width=lebar, label=nama[k],
               color=WARNA[kelompok[k]], alpha=0.45 + 0.55 * (i + 1) / len(kode_urut), yerr=err, capsize=2)
    ax.set_xticks(range(3), NAMA_KELAS)
    ax.set_ylabel("F1 data uji")
    ax.set_title("F1 per kelas (CI 95% bootstrap)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "evaluasi_f1_per_kelas.png", dpi=200)
    plt.close(fig)

    # ---------- 6. Per aplikasi ----------
    per_app = []
    for a in sorted(ref["test"]["app"].unique()):
        m_ = (ref["test"]["app"] == a).to_numpy()
        ya = yt[m_]
        boot_a = bootstrap_semua(ya, {k: Pt[k][m_] for k in Pt}, max(500, B // 4))
        for k in Pt:
            mm = metrik(ya, Pt[k][m_])
            lo_, hi_ = ci(boot_a[k][:, 0])
            per_app.append({"app": a, "model": nama[k], "n": int(m_.sum()), "n_netral": int((ya == 1).sum()),
                            "f1_macro": mm["f1_macro"], "f1_macro_ci_bawah": lo_, "f1_macro_ci_atas": hi_,
                            "f1_netral": mm["f1_netral"], "akurasi": mm["akurasi"]})
    per_app = pd.DataFrame(per_app)
    per_app.round(4).to_csv(TAB_DIR / "evaluasi_per_aplikasi.csv", index=False, encoding="utf-8")
    print("\nMacro-F1 per aplikasi (data uji; CI lebar, netral per aplikasi sangat sedikit):")
    print(per_app.pivot(index="model", columns="app", values="f1_macro").round(4).to_string())

    # ---------- 7. Uji ketahanan ----------
    rob = []
    subset = {"tervalidasi manual": ref["test"]["tervalidasi"].to_numpy(),
              "tanpa kata 'bintang'": ~ref["test"]["menyebut_bintang"].to_numpy(),
              "ulasan >= 10 kata": ref["test"]["teks_bersih"].map(lambda t: len(t.split())).to_numpy() >= 10}
    for nm, mask in subset.items():
        if mask is None or mask.sum() < 20:
            continue
        for k in Pt:
            mm = metrik(yt[mask], Pt[k][mask])
            rob.append({"subset": nm, "n": int(mask.sum()), "n_netral": int((yt[mask] == 1).sum()), "model": nama[k],
                        "akurasi": mm["akurasi"], "f1_macro": mm["f1_macro"], "f1_netral": mm["f1_netral"]})
    rob = pd.DataFrame(rob)
    if len(rob):
        rob.round(4).to_csv(TAB_DIR / "evaluasi_ketahanan.csv", index=False, encoding="utf-8")
        print("\nUji ketahanan (macro-F1 pada subset; subset kecil = sangat tidak stabil):")
        print(rob.pivot_table(index="model", columns=["subset", "n"], values="f1_macro").round(3).to_string())

    # ---------- 8. Kesalahan per ulasan (bahan Tahap 12) ----------
    kes = ref["test"][["reviewId", "app", "rating", "label_final", "tervalidasi", "teks_asli"]].copy()
    kes["y_true"] = yt
    for k in Pt:
        kes[f"pred_{k}"] = Pt[k]
    benar = np.stack([Pt[k] == yt for k in Pt])
    kes["n_model_benar"] = benar.sum(axis=0)
    kes["kategori"] = np.where(kes["n_model_benar"] == len(Pt), "semua benar",
                               np.where(kes["n_model_benar"] == 0, "semua salah", "sebagian benar"))
    kes.to_csv(TAB_DIR / "evaluasi_kesalahan_per_ulasan.csv", index=False, encoding="utf-8")
    kat = kes["kategori"].value_counts()
    print("\nKesepakatan antarmodel pada data uji:", {k: f"{v} ({100 * v / len(kes):.1f}%)" for k, v in kat.items()})
    sa = kes[kes["kategori"] == "semua salah"]
    print(f"Ulasan yang SEMUA model salahkan: {len(sa)}; di antaranya berlabel netral: "
          f"{int((sa['y_true'] == 1).sum())} ({100 * (sa['y_true'] == 1).mean():.0f}%)" if len(sa) else "Tidak ada ulasan yang semua model salahkan.")

    (TAB_DIR / "evaluasi_ringkasan.txt").write_text(
        f"model_dievaluasi: {[nama[k] for k in Pt]}\nmodel_terbaik_uji: {terbaik['model']}\n"
        f"macro_f1_uji: {terbaik['f1_macro']:.4f} (CI {terbaik['f1_macro_ci_bawah']:.4f}-{terbaik['f1_macro_ci_atas']:.4f})\n"
        f"tidak_berbeda_nyata_dari_terbaik: {tidak_beda}\nterbaik_validasi: {nama[kode_val_terbaik]}\n"
        f"bootstrap: {B}\nmd5_split: {info['md5_split_csv']}\n", encoding="utf-8")
    print(f"\nTabel di {TAB_DIR}; gambar di {FIG_DIR} (evaluasi_*.csv / evaluasi_*.png)")
    if "indobert" not in Pt:
        print("[CATATAN] Prediksi IndoBERT belum ada. Jalankan notebook Tahap 10, salin indobert_val.csv dan "
              "indobert_test.csv ke outputs/predictions/, lalu jalankan ulang skrip ini.")


def f1m(y, p):
    return prf_dari_cm(cm3(y, p))[2].mean()


if __name__ == "__main__":
    main()
