# -*- coding: utf-8 -*-
"""
Tahap 12 (satu file): analisis lanjutan penelitian ulasan ojol.

  12A  distribusi sentimen per aplikasi (sampel analisis, populasi penuh, tren)
  12B  sentimen per topik/aspek (LDA k=12 inklusif + BERTopic) + kata kunci keamanan
  12C  analisis kesalahan klasifikasi + templat pengodean manual (Excel) + rekap

Cara pakai (PowerShell, dari folder Setup_proyek):
  python tahap12_analisis_lanjutan.py                      # jalankan 12A, 12B, 12C
  python tahap12_analisis_lanjutan.py --bagian a           # hanya 12A
  python tahap12_analisis_lanjutan.py --bagian b --sfx _n12
  python tahap12_analisis_lanjutan.py --bagian c
  python tahap12_analisis_lanjutan.py --rekap              # setelah kolom kode_1 di Excel diisi

Skrip ini hanya MEMBACA hasil tahap sebelumnya dan menulis file baru berawalan t12*.
Tidak melatih ulang model, tidak mengubah split.csv, tidak memilih apa pun dengan data uji.
"""
import sys, os, re, argparse, itertools
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = Path(__file__).resolve().parent
PROYEK = HERE if (HERE / "config.py").exists() else HERE / "penelitian-ojol"
os.chdir(PROYEK); sys.path.insert(0, str(PROYEK))
from config import *            # SEED, path, APPS, LABEL2ID, ID2LABEL, rating_ke_label, ...
from utils import set_seed, make_dirs

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, norm

# =====================================================================
# KONFIGURASI (ubah di sini)
# =====================================================================
ALPHA = 0.05
LABELS = ["negatif", "netral", "positif"]
MIN_N_BULAN = 30          # 12A: bulan dengan ulasan < ini tidak digambar
REF_POPULASI = {"Gojek": 92678, "Grab": 54387, "Maxim": 89675}   # acuan Tahap 1
KODE = ["sarkasme/ironi", "campur bahasa", "slang/salah ketik", "negasi",
        "ulasan sangat pendek/ambigu", "multi-aspek/campuran", "label rating keliru", "lainnya"]
KASAR = ["anjing", "bangsat", "tolol", "goblok", "kampret", "tai", "babi", "asu",
         "brengsek", "bajingan", "sialan", "fuck", "shit"]
KW_KEAMANAN = {
    "kecelakaan/tabrak": r"\b(kecelakaan|nabrak|menabrak|ditabrak|tabrakan)\b",
    "pelecehan": r"\b(pelecehan|dilecehkan|melecehkan|mesum)\b",
    "penipuan/modus": r"\b(penipuan|menipu|tipu|ditipu|modus|scam)\b",
    "kriminal": r"\b(begal|rampok|dirampok|perampokan|copet|kejahatan|pencuri|dicuri)\b",
    "barang hilang": r"\b(hilang|kehilangan)\b",
    "kata aman/bahaya": r"\b(keamanan|aman|bahaya|berbahaya|membahayakan)\b",
}

TAB, FIG = Path(TAB_DIR), Path(FIG_DIR)
PESAN_IZIN = ("Tidak bisa membaca/menulis {p}. Tutup file itu di Excel, tunggu OneDrive "
              "selesai sinkron, atau pindahkan proyek ke C:\\riset\\ lalu ulangi.")


# =====================================================================
# FUNGSI BANTU
# =====================================================================
def wilson(k, n, alpha=ALPHA):
    """Selang kepercayaan Wilson untuk proporsi k/n."""
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (c - h, c + h)


def baca_csv(path, **kw):
    kw.setdefault("encoding", "utf-8-sig")      # aman untuk file dengan/ tanpa BOM
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, **kw)
    except PermissionError:
        raise SystemExit(PESAN_IZIN.format(p=path))
    except FileNotFoundError:
        raise SystemExit(f"File tidak ditemukan: {path}")


def tulis_csv(df, path):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        raise SystemExit(PESAN_IZIN.format(p=path))
    print("  tersimpan:", path)


def simpan_gambar(fig, nama):
    try:
        fig.savefig(FIG / nama, dpi=200)
    except PermissionError:
        print("  PERINGATAN: gambar tidak bisa disimpan (file terbuka?):", nama)
    plt.close(fig)


def muat_berlabel():
    df = baca_csv(Path(PROC_DIR) / "ulasan_berlabel.csv")
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    for c in ["tervalidasi", "dikoreksi", "menyebut_bintang"]:
        if c in df.columns:
            df[c] = df[c].str.strip().str.lower().isin(["true", "1"])
    return df


def tabel_proporsi(df, kol_grup, kol_label="label_final", labels=LABELS):
    """Jumlah, persen, dan CI Wilson 95% tiap label di dalam tiap grup."""
    df = df.reset_index(drop=True)          # hindari galat "duplicate labels" bila indeks ganda
    ct = pd.crosstab(df[kol_grup], df[kol_label]).reindex(columns=labels, fill_value=0)
    baris = []
    for g, r in ct.iterrows():
        n = int(r.sum())
        for lab in labels:
            k = int(r[lab])
            lo, hi = wilson(k, n)
            baris.append({kol_grup: g, "label": lab, "n_grup": n, "n": k,
                          "persen": 100 * k / n if n else np.nan,
                          "ci_bawah": 100 * lo, "ci_atas": 100 * hi})
    return pd.DataFrame(baris)


# =====================================================================
# 12A  SENTIMEN PER APLIKASI
# =====================================================================
def selisih_antar_app(ct):
    n = ct.sum(axis=1)
    pasangan = list(itertools.combinations(ct.index, 2))
    m = len(pasangan) * len(LABELS)
    z = norm.ppf(1 - ALPHA / (2 * m))                  # CI Bonferroni
    baris, pv = [], []
    for a, b in pasangan:
        for lab in LABELS:
            pa, pb = ct.loc[a, lab] / n[a], ct.loc[b, lab] / n[b]
            se = np.sqrt(pa * (1 - pa) / n[a] + pb * (1 - pb) / n[b])
            d = pa - pb
            pval = 2 * (1 - norm.cdf(abs(d) / se)) if se > 0 else np.nan
            pv.append(pval)
            baris.append({"app_a": a, "app_b": b, "label": lab, "selisih_poin": 100 * d,
                          "ci_bawah_bonf": 100 * (d - z * se), "ci_atas_bonf": 100 * (d + z * se),
                          "p_mentah": pval})
    pv = np.nan_to_num(np.array(pv), nan=1.0)
    o = np.argsort(pv); adj = np.empty(m); run = 0
    for rank, i in enumerate(o):                       # koreksi Holm
        run = max(run, (m - rank) * pv[i]); adj[i] = min(1, run)
    out = pd.DataFrame(baris); out["p_holm"] = adj
    return out


def jalankan_12a(args):
    print("\n" + "=" * 70 + "\n12A: SENTIMEN PER APLIKASI\n" + "=" * 70)
    print("== 12A-1: sampel analisis (label_final) ==")
    df = muat_berlabel()
    print("  jumlah per aplikasi:", df["app"].value_counts().to_dict(), "(acuan: 9.917 / 9.869 / 9.817)")
    tp = tabel_proporsi(df, "app")
    tulis_csv(tp, TAB / "t12a_sentimen_app_sampel.csv")
    print(tp.pivot(index="app", columns="label", values="persen").round(1))

    ct = pd.crosstab(df["app"], df["label_final"]).reindex(columns=LABELS, fill_value=0)
    chi2, p, dof, _ = chi2_contingency(ct)
    V = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print(f"chi2={chi2:.1f}, dof={dof}, p={p:.3g}, Cramer's V={V:.3f}")
    tulis_csv(pd.DataFrame([{"chi2": chi2, "dof": dof, "p": p, "cramers_v": V}]),
              TAB / "t12a_uji_chi2_app.csv")
    tulis_csv(selisih_antar_app(ct), TAB / "t12a_selisih_antar_app.csv")

    # ---- populasi penuh (label dari rating saja)
    print("\n== 12A-2: populasi penuh (label dari rating saja) ==")
    semua, filt = [], []
    for app in APPS:
        f = Path(RAW_DIR) / f"{app.lower()}_raw.csv"
        if not f.exists():
            print(f"  {app}: file mentah tidak ada, dilewati"); continue
        d = baca_csv(f)
        d["rating"] = pd.to_numeric(d["score"], errors="coerce")
        d = d.dropna(subset=["rating"]).copy()
        d["label"] = d["rating"].astype(int).map(rating_ke_label)
        d["app"] = app
        semua.append(d[["app", "label"]])
        d["n_kata"] = d["content"].map(lambda t: len(re.findall(r"[A-Za-z]+", t)))
        d2 = d[d["n_kata"] >= 3].copy()
        d2["kunci"] = d2["content"].map(lambda t: t.strip().lower())
        d2 = d2.drop_duplicates("kunci")
        print(f"  {app}: semua={len(d):,}; >=3 kata & unik={len(d2):,} (acuan Tahap 1: {REF_POPULASI.get(app, 0):,})")
        filt.append(d2[["app", "label"]])
    print("  Catatan: bila selisih dengan acuan besar, aturan filter di sini berbeda dari "
          "sampling_gabungan.py; periksa sebelum angkanya dilaporkan.")
    for nama, lst in [("semua", semua), ("min3kata_unik", filt)]:
        if not lst:
            continue
        t = tabel_proporsi(pd.concat(lst, ignore_index=True), "app", kol_label="label")
        tulis_csv(t, TAB / f"t12a_sentimen_app_populasi_{nama}.csv")
        print(f"  populasi {nama} (% per label):")
        print(t.pivot(index="app", columns="label", values="persen").round(1))

    # ---- tren
    print("\n== 12A-3: tren ==")
    dd = df.dropna(subset=["tanggal"]).copy()
    dd["bulan"] = dd["tanggal"].dt.to_period("M")
    bulan_akhir = dd["bulan"].max()
    print(f"  bulan terakhir ({bulan_akhir}) dianggap belum lengkap dan tidak digambar.")
    baris = []
    for (app, b), g in dd[dd["bulan"] < bulan_akhir].groupby(["app", "bulan"]):
        n = len(g); k = int((g["label_final"] == "negatif").sum()); lo, hi = wilson(k, n)
        baris.append({"app": app, "bulan": str(b), "n": n, "persen_negatif": 100 * k / n,
                      "ci_bawah": 100 * lo, "ci_atas": 100 * hi})
    bln = pd.DataFrame(baris)
    tulis_csv(bln, TAB / "t12a_tren_bulanan_negatif.csv")

    dd["tahun"] = dd["tanggal"].dt.year
    baris = []
    for (app, y), g in dd.groupby(["app", "tahun"]):
        for lab in LABELS:
            k = int((g["label_final"] == lab).sum()); n = len(g); lo, hi = wilson(k, n)
            baris.append({"app": app, "tahun": int(y), "label": lab, "n_grup": n, "n": k,
                          "persen": 100 * k / n, "ci_bawah": 100 * lo, "ci_atas": 100 * hi})
    tulis_csv(pd.DataFrame(baris), TAB / "t12a_tren_tahunan.csv")
    print("  Catatan: tanggal ulasan = tanggal terakhir diperbarui; jumlah ulasan per bulan pada "
          "sampel tidak bisa dibandingkan antaraplikasi (sampel tetap 10.000 per aplikasi).")

    # ---- gambar
    fig, ax = plt.subplots(figsize=(7, 4))
    tp.pivot(index="app", columns="label", values="persen")[LABELS].plot(
        kind="barh", stacked=True, ax=ax, color=["#c0392b", "#f39c12", "#27ae60"])
    ax.set_xlabel("% ulasan (sampel analisis)"); ax.set_ylabel("")
    ax.set_title("Distribusi sentimen per aplikasi")
    fig.tight_layout(); simpan_gambar(fig, "t12a_sentimen_per_app.png")

    if len(bln):
        fig, ax = plt.subplots(figsize=(8, 4))
        bln["n"] = bln["n"].astype(int)
        for app, g in bln[bln["n"] >= MIN_N_BULAN].groupby("app"):
            ax.plot(pd.PeriodIndex(g["bulan"], freq="M").to_timestamp(), g["persen_negatif"],
                    marker="o", ms=3, label=app)
        ax.set_ylabel("% ulasan negatif")
        ax.set_title(f"Tren bulanan % negatif (bulan dengan n>={MIN_N_BULAN})")
        ax.legend(); fig.tight_layout(); simpan_gambar(fig, "t12a_tren_negatif.png")
    print("Selesai 12A.")


# =====================================================================
# 12B  SENTIMEN PER TOPIK / ASPEK
# =====================================================================
def jalankan_12b(args):
    import seaborn as sns
    SFX = args.sfx
    MODEL_BT = f"bertopic{SFX}"
    print("\n" + "=" * 70 + f"\n12B: SENTIMEN PER TOPIK (BERTopic sfx='{SFX}')\n" + "=" * 70)

    ber = muat_berlabel()[["reviewId", "app", "label_final", "teks_bersih"]]
    lda = baca_csv(TAB / "lda_topik_dokumen_k12_inklusif.csv")
    pcols = sorted([c for c in lda.columns if c.startswith("p_topik_")], key=lambda c: int(c.split("_")[-1]))
    lda = lda[["reviewId", "topik_dominan"] + pcols].copy()
    for c in pcols:
        lda[c] = pd.to_numeric(lda[c], errors="coerce")
    bt = None
    f_bt = TAB / f"bertopic_topik_dokumen{SFX}.csv"
    if f_bt.exists():
        bt = baca_csv(f_bt)[["reviewId", "topik", "topik_tanpa_outlier"]]
    else:
        print(f"  PERINGATAN: {f_bt.name} tidak ada; bagian BERTopic dilewati.")

    # ---- pemetaan nama topik -> kategori aspek (templat dibuat otomatis)
    F_PEM = TAB / "pemetaan_topik_aspek.csv"

    def nama_dari_file(f, model):
        if not f.exists():
            return []
        d = None
        for kw in ({"encoding": "utf-8-sig"}, {"sep": ";", "encoding": "utf-8-sig"},
                   {"encoding": "cp1252"}, {"sep": ";", "encoding": "cp1252"}):
            try:
                d = pd.read_csv(f, dtype=str, keep_default_na=False, **kw)
                if "topik" in d.columns and "nama_manual" in d.columns:
                    break
            except PermissionError:
                raise SystemExit(PESAN_IZIN.format(p=f))
            except Exception:
                pass
            d = None
        if d is None:
            print(f"  PERINGATAN: {f.name} tidak terbaca (baris rusak/pemisah/kolom nama_manual tidak ada). "
                  "Nama topik dikosongkan; isi nama_manual dan kategori_aspek langsung di pemetaan_topik_aspek.csv.")
            return []
        return [{"model": model, "topik": r["topik"], "nama_manual": r.get("nama_manual", "").strip(),
                 "kategori_aspek": ""} for _, r in d.iterrows()]

    if F_PEM.exists():
        pem = baca_csv(F_PEM)
    else:
        baris = (nama_dari_file(TAB / "lda_penamaan_topik_k12_inklusif.csv", "lda")
                 + nama_dari_file(TAB / f"bertopic_penamaan{SFX}.csv", MODEL_BT))
        pem = pd.DataFrame(baris, columns=["model", "topik", "nama_manual", "kategori_aspek"])
        ada = set(zip(pem["model"], pem["topik"]))
        tambah = [{"model": "lda", "topik": str(k), "nama_manual": "", "kategori_aspek": ""}
                  for k in range(len(pcols)) if ("lda", str(k)) not in ada]
        if bt is not None:
            tambah += [{"model": MODEL_BT, "topik": t, "nama_manual": "", "kategori_aspek": ""}
                       for t in sorted(set(bt["topik"]) - {"-1"}, key=int) if (MODEL_BT, t) not in ada]
        if tambah:
            pem = pd.concat([pem, pd.DataFrame(tambah)], ignore_index=True)
        tulis_csv(pem, F_PEM)
        print("  >> Templat pemetaan dibuat. Isi kolom kategori_aspek (mis. tarif/promo, pembayaran/akun, "
              "perilaku pengemudi, bug aplikasi/iklan, waktu tunggu, akurasi lokasi, pujian umum, lainnya) "
              "lalu jalankan ulang bagian ini. Sementara dipakai label T{id}.")
    PEM = {(r["model"], r["topik"]): (r["nama_manual"].strip(), r["kategori_aspek"].strip())
           for _, r in pem.iterrows()}

    def nama_topik(model, tid):
        if tid == "-1":
            return "outlier"
        return f"T{tid} {PEM.get((model, tid), ('', ''))[0]}".strip()

    def kategori(model, tid):
        if tid == "-1":
            return "(outlier)"
        return PEM.get((model, tid), ("", ""))[1] or "(belum dipetakan)"

    def analisis(d, kol, tag, judul):
        print(f"\n--- {judul} ({tag}) n={len(d):,} ---")
        tp = tabel_proporsi(d, kol)
        tulis_csv(tp, TAB / f"t12b_{tag}_sentimen_dalam_topik.csv")      # sudut pandang 1
        neg = d[d["label_final"] == "negatif"]                           # sudut pandang 2
        komp = neg[kol].value_counts().rename_axis(kol).reset_index(name="n_negatif")
        komp["persen_dari_negatif"] = 100 * komp["n_negatif"] / max(komp["n_negatif"].sum(), 1)
        tulis_csv(komp, TAB / f"t12b_{tag}_komposisi_dalam_negatif.csv")
        kompa = (neg.groupby("app")[kol].value_counts(normalize=True).mul(100)
                 .rename("persen_dari_negatif_app").reset_index())
        tulis_csv(kompa, TAB / f"t12b_{tag}_komposisi_negatif_per_app.csv")
        ta = (d.groupby("app")[kol].value_counts(normalize=True).mul(100)
              .rename("persen_dalam_app").reset_index())
        tulis_csv(ta, TAB / f"t12b_{tag}_topik_per_app.csv")
        g = d.groupby(["app", kol]).agg(
            n=("label_final", "size"),
            n_neg=("label_final", lambda s: int((s == "negatif").sum()))).reset_index()
        cis = [wilson(k, n) for k, n in zip(g["n_neg"], g["n"])]
        g["persen_negatif"] = 100 * g["n_neg"] / g["n"]
        g["ci_bawah"] = [100 * c[0] for c in cis]
        g["ci_atas"] = [100 * c[1] for c in cis]
        g = g[g["n"] >= args.min_sel]
        tulis_csv(g, TAB / f"t12b_{tag}_app_topik_negatif.csv")
        besar = tp.groupby(kol)["n_grup"].first()
        besar = besar[besar >= args.min_heatmap].index
        piv = tp[tp[kol].isin(besar)].pivot(index=kol, columns="label", values="persen")[LABELS]
        if len(piv):
            fig, ax = plt.subplots(figsize=(6, 0.38 * len(piv) + 2))
            sns.heatmap(piv, annot=True, fmt=".0f", cmap="RdYlGn_r", vmin=0, vmax=100,
                        cbar_kws={"label": "% di dalam topik"}, ax=ax)
            ax.set_title(f"{judul}: sentimen di dalam topik"); ax.set_ylabel("")
            fig.tight_layout(); simpan_gambar(fig, f"t12b_{tag}_heatmap_sentimen.png")
        if len(g):
            pv2 = g.pivot(index=kol, columns="app", values="persen_negatif")
            fig, ax = plt.subplots(figsize=(6, 0.38 * len(pv2) + 2))
            sns.heatmap(pv2, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=100,
                        cbar_kws={"label": "% negatif"}, ax=ax)
            ax.set_title(f"{judul}: % negatif per aplikasi (n>={args.min_sel})"); ax.set_ylabel("")
            fig.tight_layout(); simpan_gambar(fig, f"t12b_{tag}_heatmap_negatif_app.png")
        print("  Peringatan tafsir: topik LDA/BERTopic bercampur dengan polaritas (pujian vs keluhan), "
              "jadi sentimen di dalam topik akan terpolarisasi; ini bukan temuan mengejutkan.")
        return tp

    # ---- LDA
    print("== LDA k=12 inklusif ==")
    dl = ber.merge(lda, on="reviewId", how="inner")
    print(f"  cakupan: {len(dl):,}/{len(ber):,} ulasan ({100 * len(dl) / len(ber):.1f}%; acuan ±94%)")
    cov = (ber.assign(dipakai=ber["reviewId"].isin(dl["reviewId"]))
           .groupby("label_final")["dipakai"].mean().mul(100).round(1))
    print("  % dipakai LDA per label:\n", cov.to_string())
    dl["topik_nama"] = dl["topik_dominan"].map(lambda t: nama_topik("lda", t))
    analisis(dl, "topik_nama", "lda", "LDA k=12 (topik dominan)")

    rows = []
    for lab in LABELS:
        sub = dl[dl["label_final"] == lab]
        for k, c in enumerate(pcols):
            rows.append({"topik": str(k), "label": lab, "massa": sub[c].sum()})
    w = pd.DataFrame(rows)
    w["persen_dalam_topik"] = 100 * w["massa"] / w.groupby("topik")["massa"].transform("sum")
    w["topik_nama"] = w["topik"].map(lambda t: nama_topik("lda", t))
    tulis_csv(w, TAB / "t12b_lda_sentimen_berbobot_probabilitas.csv")
    print("  Sensitivitas berbobot probabilitas (% negatif per topik):")
    print(w[w["label"] == "negatif"][["topik_nama", "persen_dalam_topik"]].round(1).to_string(index=False))

    if (dl["topik_dominan"].map(lambda t: PEM.get(("lda", t), ("", ""))[1]) != "").any():
        dl["aspek"] = dl["topik_dominan"].map(lambda t: kategori("lda", t))
        analisis(dl, "aspek", "lda_aspek", "LDA: kategori aspek")

    # ---- BERTopic
    if bt is not None:
        print(f"\n== BERTopic{SFX} ==")
        db = ber.merge(bt, on="reviewId", how="inner")
        print(f"  n={len(db):,}; outlier: {100 * (db['topik'] == '-1').mean():.1f}% (acuan natural ±32,8%)")
        print("  % outlier per label:\n",
              db.groupby("label_final")["topik"].apply(lambda s: round(100 * (s == '-1').mean(), 1)).to_string())
        print("  % outlier per app:\n",
              db.groupby("app")["topik"].apply(lambda s: round(100 * (s == '-1').mean(), 1)).to_string())
        db["topik_nama_realok"] = db["topik_tanpa_outlier"].map(lambda t: nama_topik(MODEL_BT, t))
        analisis(db, "topik_nama_realok", f"bertopic{SFX}", "BERTopic (outlier dialokasikan ulang)")
        db["topik_nama_raw"] = db["topik"].map(lambda t: nama_topik(MODEL_BT, t))
        analisis(db, "topik_nama_raw", f"bertopic{SFX}_outlier_terpisah", "BERTopic (outlier dipisah)")
        if (db["topik_tanpa_outlier"].map(lambda t: PEM.get((MODEL_BT, t), ("", ""))[1]) != "").any():
            db["aspek"] = db["topik_tanpa_outlier"].map(lambda t: kategori(MODEL_BT, t))
            analisis(db, "aspek", f"bertopic{SFX}_aspek", "BERTopic: kategori aspek")

    # ---- aspek keamanan
    print("\n== Aspek keamanan (kata kunci pada teks_bersih) ==")
    pola = {k: re.compile(v) for k, v in KW_KEAMANAN.items()}
    for k, rx in pola.items():
        ber[k] = ber["teks_bersih"].map(lambda t, rx=rx: bool(rx.search(t)))
    ber["insiden_keamanan"] = ber[["kecelakaan/tabrak", "pelecehan", "penipuan/modus", "kriminal"]].any(axis=1)
    baris = []
    for flag in list(KW_KEAMANAN) + ["insiden_keamanan"]:
        for app, g in list(ber.groupby("app")) + [("Semua", ber)]:
            n = len(g); k = int(g[flag].sum()); lo, hi = wilson(k, n)
            sub = g[g[flag]]
            baris.append({"kata_kunci": flag, "app": app, "n_ulasan": n, "n_cocok": k,
                          "persen": 100 * k / n, "ci_bawah": 100 * lo, "ci_atas": 100 * hi,
                          "persen_negatif_di_antaranya":
                              100 * (sub["label_final"] == "negatif").mean() if k else np.nan})
    kw = pd.DataFrame(baris)
    tulis_csv(kw, TAB / "t12b_keamanan_kata_kunci.csv")
    print(kw[kw["app"] == "Semua"][["kata_kunci", "n_cocok", "persen", "persen_negatif_di_antaranya"]]
          .round(2).to_string(index=False))
    print("  Catatan: 'aman' sering muncul sebagai pujian (mis. 'aman nyaman'); jangan dihitung sebagai "
          "keluhan keamanan. Gunakan 'insiden_keamanan' sebagai indikator keluhan. Daftar kata kunci "
          "buatan manual dan tidak lengkap.")
    print("Selesai 12B.")


# =====================================================================
# 12C  ANALISIS KESALAHAN
# =====================================================================
F_KES = TAB / "evaluasi_kesalahan_per_ulasan.csv"
F_XLSX = TAB / "t12c_templat_pengodean_kesalahan.xlsx"
F_KUNCI = TAB / "t12c_kunci_sampel_kesalahan.csv"


def sensor(t):
    return re.sub(r"\b(" + "|".join(KASAR) + r")\b",
                  lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), str(t), flags=re.I)


def bersih_excel(t):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(t))


def muat_kesalahan():
    k = baca_csv(F_KES)
    pc = [c for c in k.columns if c.startswith("pred_")]
    if not pc:
        raise SystemExit("Kolom pred_<model> tidak ditemukan di evaluasi_kesalahan_per_ulasan.csv; "
                         "cek skema_file.txt dan nama kolom dari Tahap 11.")
    k["y_true"] = k["y_true"].astype(int)
    for c in pc:
        k[c] = k[c].astype(int)
    t = baca_csv(Path(SPLIT_DIR) / "test.csv")
    tambah = [c for c in ["reviewId", "teks_ringan", "thumbs_up"] if c in t.columns]
    k = k.merge(t[tambah], on="reviewId", how="left", suffixes=("", "_test"))
    k["rating"] = pd.to_numeric(k["rating"], errors="coerce")
    for c in ["tervalidasi", "dikoreksi", "menyebut_bintang"]:
        if c in k.columns:
            k[c] = k[c].astype(str).str.lower().isin(["true", "1"])
    if "dikoreksi" not in k.columns:                      # ambil dari test.csv bila tak ada
        td = baca_csv(Path(SPLIT_DIR) / "test.csv")
        if "dikoreksi" in td.columns:
            td["dikoreksi"] = td["dikoreksi"].str.lower().isin(["true", "1"])
            k = k.merge(td[["reviewId", "dikoreksi"]], on="reviewId", how="left")
    models = [c[5:] for c in pc]
    for m in models:
        k["salah_" + m] = k["pred_" + m] != k["y_true"]
    k["n_salah"] = k[["salah_" + m for m in models]].sum(axis=1)
    ib = [m for m in models if "indobert" in m]
    klas = [m for m in models if "indobert" not in m]

    def grup(r):
        if r["n_salah"] == 0: return "semua_benar"
        if r["n_salah"] == len(models): return "semua_salah"
        if ib and klas and (not r["salah_" + ib[0]]) and all(r["salah_" + m] for m in klas):
            return "hanya_indobert_benar"
        if ib and klas and r["salah_" + ib[0]] and not any(r["salah_" + m] for m in klas):
            return "hanya_indobert_salah"
        return "campuran"
    k["kelompok"] = k.apply(grup, axis=1)
    return k, pc, models, ib, klas


def rekap_pengodean():
    from sklearn.metrics import cohen_kappa_score
    if not F_XLSX.exists() or not F_KUNCI.exists():
        raise SystemExit("Templat/kunci belum ada. Jalankan dulu tanpa --rekap (bagian c).")
    try:
        x = pd.read_excel(F_XLSX, sheet_name="pengodean", dtype=str).fillna("")
    except PermissionError:
        raise SystemExit(PESAN_IZIN.format(p=F_XLSX))
    kunci = baca_csv(F_KUNCI)
    d = x.merge(kunci, on="reviewId", how="inner", suffixes=("", "_k"))
    d = d[d["kode_1"].str.strip() != ""].copy()
    salah = d[~d["kode_1"].isin(KODE)]
    if len(salah):
        print("PERINGATAN: kode tidak dikenal (diabaikan):", salah["kode_1"].unique().tolist())
        d = d[d["kode_1"].isin(KODE)]
    print(f"Ulasan terkode: {len(d)} dari {len(x)}")
    if len(d) == 0:
        raise SystemExit("Belum ada baris yang dikode.")
    d["pop"] = pd.to_numeric(d["ukuran_populasi_strata"])
    d["n_kode_strata"] = d.groupby("kelompok")["reviewId"].transform("size")
    d["bobot"] = d["pop"] / d["n_kode_strata"]            # bobot desain
    models = [c[5:] for c in kunci.columns if c.startswith("pred_")]

    def ringkas(sub, nama):
        t = sub.groupby("kode_1").agg(n=("reviewId", "size"), bobot=("bobot", "sum")) \
            .reindex(KODE, fill_value=0).reset_index()
        t["persen_tak_berbobot"] = 100 * t["n"] / max(t["n"].sum(), 1)
        t["persen_berbobot"] = 100 * t["bobot"] / max(t["bobot"].sum(), 1e-9)
        t.insert(0, "subset", nama)
        return t

    hasil = [ringkas(d, "semua_sampel")]
    for m in models:
        sub = d[d["pred_" + m].astype(int) != d["y_true"].astype(int)]
        if len(sub):
            hasil.append(ringkas(sub, f"model_salah:{m}"))
    for g, sub in d.groupby("kelompok"):
        hasil.append(ringkas(sub, f"kelompok:{g}"))
    out = pd.concat(hasil)
    tulis_csv(out, TAB / "t12c_rekap_pengodean.csv")
    print(out[out["subset"] == "semua_sampel"][["kode_1", "n", "persen_tak_berbobot", "persen_berbobot"]]
          .round(1).to_string(index=False))
    if "kode_penilai2" in d.columns:
        dua = d[d["kode_penilai2"].str.strip() != ""]
        if len(dua) >= 30:
            print(f"Kappa antar-pengode (n={len(dua)}): "
                  f"{cohen_kappa_score(dua['kode_1'], dua['kode_penilai2']):.3f}")
        else:
            print("Kolom kode_penilai2 terisi < 30 baris; kappa tidak dihitung.")
    contoh = []
    for kd in KODE:
        for _, r in d[d["kode_1"] == kd].head(3).iterrows():
            contoh.append({"kode": kd, "app": r["app"], "rating": r["rating"],
                           "label_final": r["label_final"], "teks": sensor(r["teks_asli"])})
    tulis_csv(pd.DataFrame(contoh), TAB / "t12c_contoh_per_kategori.csv")
    print("Persen berbobot = perkiraan pada seluruh kesalahan (>=1 model salah) di data uji; "
          "persen tak berbobot hanya menggambarkan sampel.")


def jalankan_12c(args):
    print("\n" + "=" * 70 + "\n12C: ANALISIS KESALAHAN\n" + "=" * 70)
    k, pc, models, ib, klas = muat_kesalahan()
    print(f"n data uji = {len(k)}; model: {models}")
    if not ib:
        print("PERINGATAN: kolom pred_indobert tidak ada; perbandingan IndoBERT dilewati.")

    kel = k["kelompok"].value_counts().rename_axis("kelompok").reset_index(name="n")
    kel["persen"] = 100 * kel["n"] / len(k)
    tulis_csv(kel, TAB / "t12c_kelompok_kesalahan.csv"); print(kel.round(1).to_string(index=False))

    nm = {0: "negatif", 1: "netral", 2: "positif"}
    baris = []
    for m in models:
        s = k[k["salah_" + m]]
        if not len(s):
            continue
        pola = (s["y_true"].map(nm) + " -> " + s["pred_" + m].map(nm)).value_counts()
        for p_, n in pola.items():
            baris.append({"model": m, "pola": p_, "n": int(n), "persen_dari_kesalahan": 100 * n / len(s)})
    tulis_csv(pd.DataFrame(baris), TAB / "t12c_pola_kesalahan.csv")

    # indikator kasar (BUKAN pendeteksi sarkasme / campur bahasa)
    teks = k["teks_ringan"].fillna("") if "teks_ringan" in k.columns else k["teks_asli"].fillna("").str.lower()
    k["jml_kata"] = teks.map(lambda t: len(t.split()))
    k["bin_panjang"] = pd.cut(k["jml_kata"], [-1, 3, 5, 10, 20, 10 ** 6],
                              labels=["<=3", "4-5", "6-10", "11-20", ">20"]).astype(str)
    ING = set("the is are and very so not good bad nice great thanks thank please fast slow love best worst but this my you no".split())
    NEG = re.compile(r"\b(tidak|gak|ga|nggak|ngga|enggak|bukan|belum|tak|jangan)\b")
    KON = re.compile(r"\b(tapi|tetapi|namun|cuma|hanya|sayang|sayangnya)\b")
    k["ada_kata_inggris"] = teks.map(lambda t: any(w in ING for w in t.split()))
    k["ada_negasi"] = teks.map(lambda t: bool(NEG.search(t)))
    k["ada_kontras"] = teks.map(lambda t: bool(KON.search(t)))
    print("  Catatan: 'ada_kata_inggris', 'ada_negasi', 'ada_kontras' hanyalah indikator kasar, "
          "bukan deteksi sarkasme atau campur bahasa.")

    dims = [d for d in ["app", "rating", "label_final", "bin_panjang", "menyebut_bintang", "tervalidasi",
                        "ada_kata_inggris", "ada_negasi", "ada_kontras"] if d in k.columns]
    F_LDA = TAB / "lda_topik_dokumen_k12_inklusif.csv"
    if F_LDA.exists():
        lda = baca_csv(F_LDA)[["reviewId", "topik_dominan"]].rename(columns={"topik_dominan": "topik_lda"})
        k = k.merge(lda, on="reviewId", how="left")
        k["topik_lda"] = k["topik_lda"].replace("", "(tidak dipetakan)").fillna("(tidak dipetakan)")
        dims.append("topik_lda")
    baris = []
    for dm in dims:
        for nilai, g in k.groupby(dm):
            for m in models:
                n = len(g); s = int(g["salah_" + m].sum()); lo, hi = wilson(s, n)
                baris.append({"dimensi": dm, "nilai": nilai, "model": m, "n": n, "n_salah": s,
                              "persen_salah": 100 * s / n, "ci_bawah": 100 * lo, "ci_atas": 100 * hi})
    sub = k[k["tervalidasi"]] if "tervalidasi" in k.columns else k.iloc[0:0]
    if "dikoreksi" in k.columns and len(sub):
        for nilai, g in sub.groupby("dikoreksi"):
            for m in models:
                n = len(g); s = int(g["salah_" + m].sum()); lo, hi = wilson(s, n)
                baris.append({"dimensi": "dikoreksi (hanya tervalidasi)", "nilai": nilai, "model": m,
                              "n": n, "n_salah": s, "persen_salah": 100 * s / n,
                              "ci_bawah": 100 * lo, "ci_atas": 100 * hi})
    dim = pd.DataFrame(baris)
    tulis_csv(dim, TAB / "t12c_galat_per_dimensi.csv")
    print(f"  Subset tervalidasi di data uji hanya n={len(sub)}: sangat kecil, pakai sebagai bukti penunjang saja.")

    fig, ax = plt.subplots(figsize=(7, 4))
    urut = ["<=3", "4-5", "6-10", "11-20", ">20"]
    for m in models:
        t = dim[(dim["dimensi"] == "bin_panjang") & (dim["model"] == m)].set_index("nilai").reindex(urut)
        ax.plot(urut, t["persen_salah"], marker="o", label=m)
    ax.set_xlabel("jumlah kata"); ax.set_ylabel("% salah")
    ax.set_title("Tingkat kesalahan menurut panjang ulasan")
    ax.legend(fontsize=7); fig.tight_layout(); simpan_gambar(fig, "t12c_galat_panjang.png")

    ss = k[k["kelompok"] == "semua_salah"].copy()
    if len(ss):
        ss["pred_seragam"] = ss[["pred_" + m for m in models]].nunique(axis=1) == 1
        print(f"\n  Semua model salah: n={len(ss)}; {100 * ss['pred_seragam'].mean():.1f}% di antaranya semua "
              "model memprediksi kelas yang SAMA (petunjuk, bukan bukti, adanya label berderau).")
        ringk = ss.groupby(["label_final", "rating"]).size().rename("n").reset_index()
        ringk["persen"] = 100 * ringk["n"] / len(ss)
        tulis_csv(ringk, TAB / "t12c_semua_salah_menurut_label_rating.csv")
        print(ringk.round(1).to_string(index=False))
        kolom = [c for c in ["reviewId", "app", "rating", "label_final", "tervalidasi", "teks_asli", "pred_seragam"]
                 if c in ss.columns]
        tulis_csv(ss[kolom], TAB / "t12c_semua_salah_daftar.csv")

    # templat pengodean manual
    if F_XLSX.exists() and not args.paksa:
        print(f"\nTemplat Excel sudah ada ({F_XLSX.name}); TIDAK ditimpa agar isianmu aman. "
              "Pakai --paksa bila memang ingin membuat ulang.")
    else:
        pop = k[k["n_salah"] > 0]
        kuota = ({"semua_salah": .5, "hanya_indobert_benar": .15, "hanya_indobert_salah": .15, "campuran": .2}
                 if ib and klas else {"semua_salah": .6, "campuran": .4})
        ambil = []
        for g, q in kuota.items():
            p = pop[pop["kelompok"] == g]
            n = min(len(p), int(round(args.n_sampel * q)))
            if n == 0:
                continue
            s = p.sample(n=n, random_state=SEED).copy()
            s["ukuran_populasi_strata"] = len(p); s["ukuran_sampel_strata"] = n
            ambil.append(s)
        if not ambil:
            print("  Tidak ada kesalahan untuk disampel."); return
        samp = pd.concat(ambil).sample(frac=1, random_state=SEED).reset_index(drop=True)
        tulis_csv(samp[["reviewId", "kelompok", "y_true"] + pc + ["ukuran_populasi_strata", "ukuran_sampel_strata"]],
                  F_KUNCI)
        from openpyxl import Workbook
        from openpyxl.worksheet.datavalidation import DataValidation
        from openpyxl.styles import Alignment, Font
        wb = Workbook(); ws = wb.active; ws.title = "pengodean"
        ws.append(["no", "reviewId", "app", "rating", "label_final", "teks_asli", "kode_1", "kode_penilai2", "catatan"])
        for i, r in samp.iterrows():
            ws.append([i + 1, r["reviewId"], r["app"], r["rating"], r["label_final"],
                       bersih_excel(r["teks_asli"]), "", "", ""])
        for c in ws[1]:
            c.font = Font(bold=True)
        for col, w in zip("ABCDEFGHI", [5, 20, 8, 7, 11, 70, 26, 26, 30]):
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws2 = wb.create_sheet("daftar"); ws2.append(["kode"])
        for c in KODE:
            ws2.append([c])
        dv = DataValidation(type="list", formula1=f"=daftar!$A$2:$A${len(KODE) + 1}", allow_blank=True)
        ws.add_data_validation(dv); dv.add(f"G2:H{len(samp) + 1}")
        ws.freeze_panes = "A2"
        try:
            wb.save(F_XLSX)
        except PermissionError:
            raise SystemExit(f"Tidak bisa menyimpan {F_XLSX}. Tutup Excel / pindahkan proyek ke C:\\riset\\.")
        print(f"\nTemplat pengodean dibuat: {F_XLSX} ({len(samp)} ulasan). Prediksi model SENGAJA "
              "disembunyikan (ada di kunci CSV) agar pengodean tidak bias.")
    print("Selesai 12C (otomatis). Isi kolom kode_1 di Excel, lalu jalankan dengan --rekap.")


# =====================================================================
# MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Tahap 12: analisis lanjutan (satu file)")
    ap.add_argument("--bagian", nargs="+", choices=["a", "b", "c"], default=["a", "b", "c"],
                    help="bagian yang dijalankan (default: a b c)")
    ap.add_argument("--rekap", action="store_true", help="rekap pengodean manual 12C (setelah Excel diisi)")
    ap.add_argument("--sfx", default="", help="12B: akhiran varian BERTopic: '' / _n12 / _n20")
    ap.add_argument("--min-sel", type=int, default=30, help="12B: n minimal sel app x topik")
    ap.add_argument("--min-heatmap", type=int, default=100, help="12B: topik dengan n < ini tidak masuk heatmap")
    ap.add_argument("--n-sampel", type=int, default=120, help="12C: ukuran sampel pengodean manual")
    ap.add_argument("--paksa", action="store_true", help="12C: timpa templat Excel (isian lama HILANG)")
    args = ap.parse_args()

    set_seed(SEED); make_dirs()
    if args.rekap:
        rekap_pengodean(); return

    fungsi = {"a": jalankan_12a, "b": jalankan_12b, "c": jalankan_12c}
    gagal = []
    for b in args.bagian:
        try:
            fungsi[b](args)
        except SystemExit as e:                          # file opsional belum ada, dsb.
            print(f"\n[12{b.upper()}] BERHENTI: {e}")
            gagal.append(b)
        except Exception as e:                           # galat tak terduga: lanjut ke bagian lain
            import traceback
            traceback.print_exc()
            print(f"\n[12{b.upper()}] GALAT: {type(e).__name__}: {e}")
            gagal.append(b)
    print("\n" + "=" * 70)
    print("Selesai. Bagian gagal/terlewati:", ", ".join(gagal) if gagal else "tidak ada")


if __name__ == "__main__":
    main()
