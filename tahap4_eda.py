"""
tahap4_eda.py  -  Tahap 4: eksplorasi data (EDA).

Letakkan di sebelah folder penelitian-ojol/ atau di dalamnya, lalu jalankan:

    python tahap4_eda.py

Input : data/processed/ulasan_berlabel.csv   (hasil tahap3_pelabelan.py --buat atau --hitung)
Output: outputs/figures/eda_*.png  dan  outputs/tables/eda_*.csv

Catatan: bila validasi manual (Tahap 3) belum selesai, label_final masih sama dengan label dari
rating. Setelah validasi selesai, JALANKAN ULANG skrip ini (hanya beberapa detik) agar angka
laporan memakai label yang sudah dikoreksi.

Opsional: pip install wordcloud  (tanpa itu, word cloud dilewati; tabel kata teratas tetap dibuat)
"""
import os
import sys
from collections import Counter
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
    from scipy import stats
except ImportError as e:
    sys.exit(f"Library belum terpasang ({e}). Jalankan: python setup_proyek.py --instal")
try:
    from wordcloud import WordCloud
    ADA_WORDCLOUD = True
except ImportError:
    ADA_WORDCLOUD = False

from config import *            # noqa: F401,F403  (SEED, APPS, PROC_DIR, FIG_DIR, TAB_DIR, ...)
from utils import set_seed, make_dirs

# --------------------------------------------------------------------------
# Konfigurasi tahap 4
# --------------------------------------------------------------------------
FILE_MASUK = PROC_DIR / "ulasan_berlabel.csv"
KOLOM_LABEL = "label_final"        # pakai label setelah validasi (sementara = label dari rating)
TOP_N_KATA = 15                    # jumlah kata teratas per kelompok
MAKS_KATA_PLOT = 60                # histogram panjang teks dipotong di sini agar terbaca
MIN_N_BULAN = 30                   # bulan dengan ulasan kurang dari ini tidak digambar pada tren
BUANG_BULAN_PARSIAL = True         # bulan terakhir yang belum lengkap tidak diikutkan pada tren
MIN_FREKUENSI_KHAS = 20            # kata dengan frekuensi total < ini diabaikan pada 'kata khas'
URUT_LABEL = ["negatif", "netral", "positif"]
URUT_APP = list(APPS.keys())
WARNA_LABEL = {"negatif": "#d1495b", "netral": "#8d99ae", "positif": "#2a9d8f"}
WARNA_APP = dict(zip(URUT_APP, ["#1f77b4", "#ff7f0e", "#9467bd"]))

make_dirs()
set_seed(SEED)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
pd.set_option("display.width", 200)


def simpan_fig(fig, nama):
    fig.tight_layout()
    fig.savefig(FIG_DIR / nama, dpi=200)
    plt.close(fig)
    print(f"  gambar: {FIG_DIR / nama}")


def simpan_tab(tabel, nama, index=True):
    tabel.to_csv(TAB_DIR / nama, index=index, encoding="utf-8")


# --------------------------------------------------------------------------
# Muat data
# --------------------------------------------------------------------------
if not FILE_MASUK.exists():
    sys.exit(f"{FILE_MASUK} tidak ada. Jalankan dulu: python tahap3_pelabelan.py --buat")
d = pd.read_csv(FILE_MASUK, parse_dates=["tanggal"], keep_default_na=False,
                dtype={"teks_asli": str, "teks_ringan": str, "teks_bersih": str,
                       "teks_topik": str, "reviewId": str})
perlu = {"app", "rating", "tanggal", "thumbs_up", "jumlah_kata_bersih", "teks_asli", "teks_topik",
         KOLOM_LABEL}
if not perlu <= set(d.columns):
    sys.exit(f"Kolom tidak lengkap di {FILE_MASUK.name}; kurang: {sorted(perlu - set(d.columns))}")
d["label"] = d[KOLOM_LABEL]
d["jumlah_karakter"] = d["teks_asli"].map(len)
n_valid = int(d["tervalidasi"].astype(str).str.lower().eq("true").sum()) if "tervalidasi" in d else 0
print(f"Data: {len(d):,} ulasan | label: {KOLOM_LABEL} | tervalidasi manual: {n_valid}"
      f"{'  (label masih SEMENTARA = dari rating)' if n_valid == 0 else ''}")
print(f"WordCloud tersedia: {ADA_WORDCLOUD}\n")

# --------------------------------------------------------------------------
# 1. Ringkasan dataset
# --------------------------------------------------------------------------
print("=== 1. RINGKASAN DATASET ===")
ring = d.groupby("app").agg(jumlah=("rating", "size"), tanggal_awal=("tanggal", "min"),
                            tanggal_akhir=("tanggal", "max"), rating_rata2=("rating", "mean"),
                            rating_median=("rating", "median"), kata_median=("jumlah_kata_bersih", "median"),
                            thumbs_up_rata2=("thumbs_up", "mean")).reindex(URUT_APP)
ring["tanggal_awal"] = ring["tanggal_awal"].dt.date
ring["tanggal_akhir"] = ring["tanggal_akhir"].dt.date
print(ring.round(2).to_string())
simpan_tab(ring.round(3), "eda_ringkasan_dataset.csv")

# --------------------------------------------------------------------------
# 2. Distribusi rating per aplikasi
# --------------------------------------------------------------------------
print("\n=== 2. DISTRIBUSI RATING ===")
tab_r = pd.crosstab(d["app"], d["rating"]).reindex(URUT_APP)
prop_r = tab_r.div(tab_r.sum(axis=1), axis=0)
print("Jumlah:")
print(tab_r.to_string())
print("\nProporsi:")
print(prop_r.round(3).to_string())
simpan_tab(tab_r, "eda_distribusi_rating_jumlah.csv")
simpan_tab(prop_r.round(4), "eda_distribusi_rating_proporsi.csv")

fig, ax = plt.subplots(figsize=(7, 4))
lebar = 0.8 / len(URUT_APP)
for i, a in enumerate(URUT_APP):
    ax.bar(np.arange(1, 6) + (i - (len(URUT_APP) - 1) / 2) * lebar, prop_r.loc[a].reindex(range(1, 6)).fillna(0),
           width=lebar, label=a, color=WARNA_APP[a])
ax.set_xlabel("Rating (bintang)")
ax.set_ylabel("Proporsi ulasan")
ax.set_title("Distribusi rating per aplikasi")
ax.legend(title="Aplikasi")
simpan_fig(fig, "eda_rating_per_app.png")

# --------------------------------------------------------------------------
# 3. Distribusi label sentimen per aplikasi
# --------------------------------------------------------------------------
print("\n=== 3. DISTRIBUSI LABEL SENTIMEN ===")
tab_l = pd.crosstab(d["app"], d["label"]).reindex(index=URUT_APP, columns=URUT_LABEL, fill_value=0)
prop_l = tab_l.div(tab_l.sum(axis=1), axis=0)
semua = tab_l.sum()
print("Jumlah:")
print(pd.concat([tab_l, semua.rename("SEMUA").to_frame().T]).to_string())
print("\nProporsi:")
print(pd.concat([prop_l, (semua / semua.sum()).rename("SEMUA").to_frame().T]).round(3).to_string())
simpan_tab(pd.concat([tab_l, semua.rename("SEMUA").to_frame().T]), "eda_distribusi_label_jumlah.csv")
simpan_tab(pd.concat([prop_l, (semua / semua.sum()).rename("SEMUA").to_frame().T]).round(4),
           "eda_distribusi_label_proporsi.csv")

chi2, p_chi, dof, harapan = stats.chi2_contingency(tab_l.values)
cramer_v = float(np.sqrt(chi2 / (tab_l.values.sum() * (min(tab_l.shape) - 1))))
print(f"\nUji chi-kuadrat aplikasi x label: chi2 = {chi2:.1f}, dof = {dof}, p = {p_chi:.3g}, "
      f"Cramer's V = {cramer_v:.3f}  (nilai harapan minimum = {harapan.min():.1f})")
print("(Dengan n sebesar ini hampir semua perbedaan 'signifikan'; ukuran efek V lebih informatif.)")
simpan_tab(pd.DataFrame({"chi2": [chi2], "dof": [dof], "p": [p_chi], "cramer_v": [cramer_v]}),
           "eda_uji_label_vs_app.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 3.8))
baris = URUT_APP[::-1] + ["SEMUA"]
kiri = np.zeros(len(baris))
prop_plot = pd.concat([prop_l, (semua / semua.sum()).rename("SEMUA").to_frame().T]).loc[baris]
cnt_plot = pd.concat([tab_l, semua.rename("SEMUA").to_frame().T]).loc[baris]
for lab in URUT_LABEL:
    ax.barh(baris, prop_plot[lab], left=kiri, color=WARNA_LABEL[lab], label=lab)
    for y, (v, c, k0) in enumerate(zip(prop_plot[lab], cnt_plot[lab], kiri)):
        if v >= 0.08:
            ax.text(k0 + v / 2, y, f"{v:.0%}\n({c:,})", ha="center", va="center", fontsize=8, color="white")
        elif v >= 0.03:        # segmen sempit: hanya persen
            ax.text(k0 + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white")
    kiri = kiri + prop_plot[lab].values
ax.set_xlim(0, 1)
ax.set_xlabel("Proporsi ulasan")
ax.set_title(f"Distribusi sentimen per aplikasi ({KOLOM_LABEL})")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
ax.grid(axis="y", visible=False)
simpan_fig(fig, "eda_label_per_app.png")

# --------------------------------------------------------------------------
# 4. Panjang teks
# --------------------------------------------------------------------------
print("\n=== 4. PANJANG TEKS (jumlah kata pada teks_bersih) ===")
persen = [0.5, 0.75, 0.9, 0.95, 0.99]


def ringkas_panjang(kelompok):
    g = d.groupby(kelompok)["jumlah_kata_bersih"]
    t = g.agg(["mean", "median"]).join(g.quantile(persen).unstack(), how="left")
    t.columns = ["rata2", "median"] + [f"p{int(q * 100)}" for q in persen]
    t["jumlah_karakter_median"] = d.groupby(kelompok)["jumlah_karakter"].median()
    return t


pj_app = ringkas_panjang("app").reindex(URUT_APP)
pj_lab = ringkas_panjang("label").reindex(URUT_LABEL)
pj_all = pd.DataFrame([{"rata2": d["jumlah_kata_bersih"].mean(), "median": d["jumlah_kata_bersih"].median(),
                        **{f"p{int(q * 100)}": d["jumlah_kata_bersih"].quantile(q) for q in persen},
                        "jumlah_karakter_median": d["jumlah_karakter"].median()}], index=["SEMUA"])
pj = pd.concat([pj_all, pj_app, pj_lab])
print(pj.round(1).to_string())
simpan_tab(pj.round(2), "eda_panjang_teks.csv")
rho, p_rho = stats.spearmanr(d["rating"], d["jumlah_kata_bersih"])
print(f"\nKorelasi Spearman rating vs panjang teks: rho = {rho:.3f} (p = {p_rho:.3g})")
print(f"Ulasan dengan > {MAKS_KATA_PLOT} kata: {(d['jumlah_kata_bersih'] > MAKS_KATA_PLOT).mean():.2%}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
data_box = [d.loc[d["label"] == lab, "jumlah_kata_bersih"] for lab in URUT_LABEL]
bp = a1.boxplot(data_box, showfliers=False, patch_artist=True)
for patch, lab in zip(bp["boxes"], URUT_LABEL):
    patch.set_facecolor(WARNA_LABEL[lab])
    patch.set_alpha(0.8)
a1.set_xticks(range(1, 4), URUT_LABEL)
a1.set_ylabel("Jumlah kata")
a1.set_title("Panjang ulasan menurut label (tanpa pencilan)")
for a in URUT_APP:
    a2.hist(d.loc[d["app"] == a, "jumlah_kata_bersih"].clip(upper=MAKS_KATA_PLOT), bins=range(3, MAKS_KATA_PLOT + 2),
            alpha=0.55, label=a, color=WARNA_APP[a])
a2.set_xlabel(f"Jumlah kata (dipotong di {MAKS_KATA_PLOT})")
a2.set_ylabel("Jumlah ulasan")
a2.set_title("Sebaran panjang ulasan per aplikasi")
a2.legend(title="Aplikasi")
simpan_fig(fig, "eda_panjang_teks.png")

# --------------------------------------------------------------------------
# 5. Tren waktu
# --------------------------------------------------------------------------
print("\n=== 5. TREN WAKTU ===")
d["bulan"] = d["tanggal"].dt.to_period("M")
tgl_akhir = d["tanggal"].max()
bulan_parsial = tgl_akhir.to_period("M") if (BUANG_BULAN_PARSIAL and tgl_akhir.day < tgl_akhir.days_in_month) else None
dt = d if bulan_parsial is None else d[d["bulan"] != bulan_parsial]
if bulan_parsial is not None:
    print(f"Bulan terakhir ({bulan_parsial}) belum lengkap (data sampai {tgl_akhir.date()}); tidak diikutkan pada tren.")
dt = dt.assign(neg=(dt["label"] == "negatif").astype(float), pos=(dt["label"] == "positif").astype(float))
tren = dt.groupby(["app", "bulan"]).agg(n=("rating", "size"), rating_rata2=("rating", "mean"),
                                        pct_negatif=("neg", "mean"), pct_positif=("pos", "mean")).reset_index()
simpan_tab(tren.assign(bulan=tren["bulan"].astype(str)).round(4), "eda_tren_bulanan.csv", index=False)

d["tahun"] = d["tanggal"].dt.year
dd = d.assign(neg=(d["label"] == "negatif").astype(float), pos=(d["label"] == "positif").astype(float))
tren_th = dd.groupby(["app", "tahun"]).agg(n=("rating", "size"), rating_rata2=("rating", "mean"),
                                            pct_negatif=("neg", "mean"), pct_positif=("pos", "mean")).round(3)
print("Ringkasan per tahun (2026 baru sebagian):")
print(tren_th.to_string())
simpan_tab(tren_th, "eda_tren_tahunan.csv")

fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for a in URUT_APP:
    g = tren[tren["app"] == a].copy()
    g.loc[g["n"] < MIN_N_BULAN, ["rating_rata2", "pct_negatif"]] = np.nan
    x = g["bulan"].dt.to_timestamp()
    ax[0].plot(x, g["n"], marker="o", ms=3, label=a, color=WARNA_APP[a])
    ax[1].plot(x, g["rating_rata2"], marker="o", ms=3, label=a, color=WARNA_APP[a])
    ax[2].plot(x, g["pct_negatif"] * 100, marker="o", ms=3, label=a, color=WARNA_APP[a])
ax[0].set_ylabel("Ulasan dalam sampel")
ax[0].set_title("Ulasan per bulan (HANYA bentuk tren yang bermakna; tiap aplikasi disampel 10.000)", fontsize=10)
ax[1].set_ylabel("Rata-rata rating")
ax[1].set_title("Rata-rata rating per bulan", fontsize=10)
ax[2].set_ylabel("% negatif")
ax[2].set_title(f"Persentase ulasan negatif per bulan (bulan dengan n < {MIN_N_BULAN} tidak digambar)", fontsize=10)
ax[0].legend(title="Aplikasi", ncol=3)
simpan_fig(fig, "eda_tren_bulanan.png")
print("PERINGATAN: jumlah ulasan per bulan pada sampel TIDAK bisa dibandingkan antaraplikasi "
      "(tiap aplikasi disampel 10.000); hanya bentuk perubahannya yang bermakna.")

# --------------------------------------------------------------------------
# 6. Thumbs up
# --------------------------------------------------------------------------
print("\n=== 6. THUMBS UP ===")
d["ada_thumbs"] = d["thumbs_up"] > 0
tu = pd.concat([d.groupby("app")["ada_thumbs"].mean().reindex(URUT_APP).rename("persen_ada_thumbs"),
                d.groupby("app")["thumbs_up"].max().reindex(URUT_APP).rename("thumbs_maks")], axis=1)
tu_l = d.groupby("label")["ada_thumbs"].mean().reindex(URUT_LABEL).rename("persen_ada_thumbs").to_frame()
print(tu.round(3).to_string())
print(tu_l.round(3).to_string())
simpan_tab(pd.concat([tu, tu_l.assign(thumbs_maks=np.nan)]).round(4), "eda_thumbs_up.csv")

# --------------------------------------------------------------------------
# 7. Kata teratas, kata khas, dan word cloud (dari teks_topik: tanpa stopword)
# --------------------------------------------------------------------------
print("\n=== 7. KATA TERATAS DAN KATA KHAS (teks_topik) ===")
tok = d["teks_topik"].map(lambda t: t.split())
cnt_label = {lab: Counter(w for ts in tok[d["label"] == lab] for w in ts) for lab in URUT_LABEL}
cnt_app = {a: Counter(w for ts in tok[d["app"] == a] for w in ts) for a in URUT_APP}

atas = []
for lab in URUT_LABEL:
    for w, c in cnt_label[lab].most_common(TOP_N_KATA):
        atas.append({"kelompok": f"label={lab}", "kata": w, "frekuensi": c})
for a in URUT_APP:
    for w, c in cnt_app[a].most_common(TOP_N_KATA):
        atas.append({"kelompok": f"app={a}", "kata": w, "frekuensi": c})
atas = pd.DataFrame(atas)
simpan_tab(atas, "eda_kata_teratas.csv", index=False)
for lab in URUT_LABEL:
    print(f"  {lab:8s}: " + ", ".join(f"{w}({c})" for w, c in cnt_label[lab].most_common(10)))


def kata_khas(cnt_a, cnt_lain, n_atas=TOP_N_KATA, alpha=0.5):
    """Skor log-odds (dengan penghalus alpha): positif = lebih khas untuk kelompok ini."""
    kosakata = set(cnt_a) | set(cnt_lain)
    Na, Nl, V = sum(cnt_a.values()), sum(cnt_lain.values()), len(kosakata)
    baris = []
    for w in kosakata:
        a, l = cnt_a.get(w, 0), cnt_lain.get(w, 0)
        if a + l < MIN_FREKUENSI_KHAS:
            continue
        s = np.log((a + alpha) / (Na + alpha * V)) - np.log((l + alpha) / (Nl + alpha * V))
        baris.append((w, a, l, s))
    return pd.DataFrame(baris, columns=["kata", "frek_kelompok", "frek_lainnya", "skor"]) \
        .sort_values("skor", ascending=False).head(n_atas)


khas = []
print("\nKata paling KHAS tiap label (log-odds terhadap label lainnya):")
for lab in URUT_LABEL:
    lain = Counter()
    for l2 in URUT_LABEL:
        if l2 != lab:
            lain.update(cnt_label[l2])
    kh = kata_khas(cnt_label[lab], lain)
    khas.append(kh.assign(kelompok=f"label={lab}"))
    print(f"  {lab:8s}: " + ", ".join(kh["kata"].head(10)))
for a in URUT_APP:
    lain = Counter()
    for a2 in URUT_APP:
        if a2 != a:
            lain.update(cnt_app[a2])
    khas.append(kata_khas(cnt_app[a], lain).assign(kelompok=f"app={a}"))
    print(f"  {a:8s}: " + ", ".join(khas[-1]["kata"].head(10)))
simpan_tab(pd.concat(khas)[["kelompok", "kata", "frek_kelompok", "frek_lainnya", "skor"]].round(3),
           "eda_kata_khas.csv", index=False)

fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, lab in zip(axs, URUT_LABEL):
    top = cnt_label[lab].most_common(TOP_N_KATA)[::-1]
    ax.barh([w for w, _ in top], [c for _, c in top], color=WARNA_LABEL[lab])
    ax.set_title(f"Kata teratas: {lab}")
    ax.set_xlabel("Frekuensi")
    ax.grid(axis="y", visible=False)
simpan_fig(fig, "eda_kata_teratas_label.png")

if ADA_WORDCLOUD:
    def buat_wc(ax, cnt, judul, warna):
        freq = dict(cnt.most_common(150))
        if not freq:
            ax.axis("off")
            return
        wc = WordCloud(width=900, height=500, background_color="white", max_words=100, collocations=False,
                       random_state=SEED, colormap=warna).generate_from_frequencies(freq)
        ax.imshow(wc.to_array(), interpolation="bilinear")
        ax.set_title(judul)
        ax.axis("off")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, lab, cm in zip(axs, URUT_LABEL, ["Reds", "Greys", "Greens"]):
        buat_wc(ax, cnt_label[lab], f"Word cloud: {lab}", cm)
    simpan_fig(fig, "eda_wordcloud_label.png")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, a in zip(axs, URUT_APP):
        buat_wc(ax, cnt_app[a], f"Word cloud: {a}", "viridis")
    simpan_fig(fig, "eda_wordcloud_app.png")
else:
    print("\n[LEWATI] word cloud: pustaka 'wordcloud' belum terpasang (pip install wordcloud). "
          "Tabel kata teratas dan kata khas tetap dibuat.")

print("\nSelesai. Semua tabel di:", TAB_DIR, "| semua gambar di:", FIG_DIR)
