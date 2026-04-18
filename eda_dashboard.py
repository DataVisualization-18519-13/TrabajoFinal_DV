"""
EDA para Dashboard Tableau: "¿Dónde abrir un negocio?"
Dataset: Yelp Open Dataset (reviews_enriched_v1, 8 partes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import glob
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── 1. Carga ─────────────────────────────────────────────────────────────────
print("Cargando datos...")
files = sorted(glob.glob("data/processed/reviews_enriched_v1_part_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.to_period("M")

print(f"Total reseñas:      {len(df):>10,}")
print(f"Negocios únicos:    {df['business_id'].nunique():>10,}")
print(f"Rango temporal:     {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Columnas:           {df.columns.tolist()}\n")

# ── Alerta: falta columna geográfica ─────────────────────────────────────────
GEO_COLS = {"city", "state", "latitude", "longitude", "postal_code"}
missing_geo = GEO_COLS - set(df.columns)
if missing_geo:
    print("⚠️  ALERTA: faltan columnas geográficas en los parquets procesados:")
    print(f"   {missing_geo}")
    print("   Para el dashboard 'dónde abrir un negocio' es CRÍTICO agregar")
    print("   city, state, lat, lon desde el archivo business.json original.\n")

# ── 2. Nivel negocio ─────────────────────────────────────────────────────────
biz = df.groupby("business_id").agg(
    business_name=("business_name", "first"),
    categories=("categories", "first"),
    business_stars=("business_stars", "first"),
    business_review_count=("business_review_count", "first"),
    avg_review_stars=("review_stars", "mean"),
    review_count_in_data=("review_id", "count"),
    date_first=("date", "min"),
    date_last=("date", "max"),
).reset_index()

biz["log_review_count"] = np.log1p(biz["business_review_count"])
mn_s, mx_s = biz["business_stars"].min(), biz["business_stars"].max()
mn_r, mx_r = biz["log_review_count"].min(), biz["log_review_count"].max()
biz["stars_norm"] = (biz["business_stars"] - mn_s) / (mx_s - mn_s)
biz["log_rc_norm"] = (biz["log_review_count"] - mn_r) / (mx_r - mn_r)
biz["divergence_score"] = biz["stars_norm"] - biz["log_rc_norm"]

print("── Estadísticas nivel negocio ──")
print(biz[["business_stars", "business_review_count", "avg_review_stars", "divergence_score"]].describe().round(2))
print()

# ── 3. Categorías ────────────────────────────────────────────────────────────
cat_rows = []
for _, row in biz.iterrows():
    for c in str(row["categories"]).split(","):
        cat_rows.append({"cat": c.strip(), "stars": row["business_stars"],
                         "rc": row["business_review_count"],
                         "divergence": row["divergence_score"]})
cat_df = pd.DataFrame(cat_rows)
cat_agg = (
    cat_df.groupby("cat")
    .agg(n_negocios=("stars", "count"), avg_stars=("stars", "mean"),
         avg_rc=("rc", "mean"), avg_divergence=("divergence", "mean"))
    .reset_index()
)
cat_min = cat_agg[cat_agg["n_negocios"] >= 50].copy()

print("Top 10 categorías por calificación promedio (min 50 negocios):")
print(cat_min.nlargest(10, "avg_stars")[["cat", "n_negocios", "avg_stars", "avg_rc"]].to_string(index=False))
print()
print("Top 10 categorías por volumen de reseñas:")
print(cat_min.nlargest(10, "avg_rc")[["cat", "n_negocios", "avg_stars", "avg_rc"]].to_string(index=False))
print()

# ── 4. Quadrants (oportunidad de negocio) ────────────────────────────────────
biz["quadrant"] = "Low Q / Low Pop"
biz.loc[(biz["stars_norm"] >= 0.5) & (biz["log_rc_norm"] >= 0.5), "quadrant"] = "High Q / High Pop"
biz.loc[(biz["stars_norm"] >= 0.5) & (biz["log_rc_norm"] <  0.5), "quadrant"] = "High Q / Low Pop (Oportunidad)"
biz.loc[(biz["stars_norm"] <  0.5) & (biz["log_rc_norm"] >= 0.5), "quadrant"] = "Low Q / High Pop"
print("Cuadrantes:")
print(biz["quadrant"].value_counts().to_string())
print()

# ── 5. Temporal ──────────────────────────────────────────────────────────────
yearly = df.groupby("year")["review_id"].count().reset_index(name="reviews")
print("Reseñas por año:")
print(yearly.to_string(index=False))
print()

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ═══════════════════════════════════════════════════════════════════════════════

# Fig 1: Distribución de estrellas de negocios
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
biz["business_stars"].value_counts().sort_index().plot.bar(ax=axes[0], color="#4C72B0", edgecolor="white")
axes[0].set_title("Distribución de estrellas\n(nivel negocio)")
axes[0].set_xlabel("Estrellas")
axes[0].set_ylabel("# Negocios")
axes[0].tick_params(axis="x", rotation=0)

df["review_stars"].value_counts().sort_index().plot.bar(ax=axes[1], color="#DD8452", edgecolor="white")
axes[1].set_title("Distribución de estrellas\n(nivel reseña)")
axes[1].set_xlabel("Estrellas")
axes[1].set_ylabel("# Reseñas")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_stars_distribution.png")
plt.close()
print("✓ 01_stars_distribution.png")

# Fig 2: Evolución temporal de reseñas
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(yearly["year"], yearly["reviews"], color="#4C72B0", edgecolor="white")
ax.axvspan(2019.5, 2021.5, color="red", alpha=0.12, label="Impacto COVID")
ax.set_title("Volumen de reseñas por año")
ax.set_xlabel("Año")
ax.set_ylabel("# Reseñas")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_reviews_over_time.png")
plt.close()
print("✓ 02_reviews_over_time.png")

# Fig 3: Top 15 categorías por avg_stars
top15_stars = cat_min.nlargest(15, "avg_stars")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top15_stars, x="avg_stars", y="cat", palette="Blues_r", ax=ax)
ax.set_title("Top 15 categorías por calificación promedio\n(min 50 negocios)")
ax.set_xlabel("Avg Stars")
ax.set_ylabel("")
ax.set_xlim(3.8, 4.6)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_top_categories_by_stars.png")
plt.close()
print("✓ 03_top_categories_by_stars.png")

# Fig 4: Top 15 categorías por volumen de reseñas
top15_rc = cat_min.nlargest(15, "avg_rc")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top15_rc, x="avg_rc", y="cat", palette="Oranges_r", ax=ax)
ax.set_title("Top 15 categorías por volumen de reseñas promedio")
ax.set_xlabel("Avg Review Count")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_top_categories_by_volume.png")
plt.close()
print("✓ 04_top_categories_by_volume.png")

# Fig 5: Scatterplot calidad vs popularidad (cuadrantes)
fig, ax = plt.subplots(figsize=(10, 7))
palette = {
    "High Q / High Pop":              "#2196F3",
    "High Q / Low Pop (Oportunidad)": "#4CAF50",
    "Low Q / Low Pop":                "#9E9E9E",
    "Low Q / High Pop":               "#F44336",
}
for quad, grp in biz.groupby("quadrant"):
    ax.scatter(grp["log_rc_norm"], grp["stars_norm"],
               label=quad, alpha=0.35, s=18, color=palette.get(quad, "gray"))
ax.axhline(0.5, color="black", lw=0.8, ls="--")
ax.axvline(0.5, color="black", lw=0.8, ls="--")
ax.set_xlabel("Popularidad normalizada (log review count)")
ax.set_ylabel("Calidad normalizada (estrellas)")
ax.set_title("Calidad vs Popularidad por negocio\n(cuadrantes de oportunidad)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_quality_vs_popularity.png")
plt.close()
print("✓ 05_quality_vs_popularity.png")

# Fig 6: Distribución del divergence_score
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(biz["divergence_score"], bins=50, color="#4C72B0", edgecolor="white")
ax.axvline(0, color="red", lw=1.2, ls="--", label="Divergencia = 0")
ax.set_title("Distribución del Divergence Score\n(calidad − popularidad normalizada)")
ax.set_xlabel("Divergence Score")
ax.set_ylabel("# Negocios")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_divergence_distribution.png")
plt.close()
print("✓ 06_divergence_distribution.png")

# Fig 7: review_count distribution (log scale)
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(np.log1p(biz["business_review_count"]), bins=50, color="#DD8452", edgecolor="white")
ax.set_title("Distribución del log(review_count) por negocio")
ax.set_xlabel("log(1 + review_count)")
ax.set_ylabel("# Negocios")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_log_review_count.png")
plt.close()
print("✓ 07_log_review_count.png")

# ── Exportar CSVs para Tableau ─────────────────────────────────────────────
print("\nExportando CSVs para Tableau...")

# CSV 1: nivel negocio (sin texto de reseñas)
biz_export = biz[[
    "business_id", "business_name", "categories",
    "business_stars", "business_review_count",
    "avg_review_stars", "review_count_in_data",
    "divergence_score", "stars_norm", "log_rc_norm", "quadrant",
    "date_first", "date_last"
]].copy()
biz_export.to_csv(OUTPUT_DIR / "businesses_enriched.csv", index=False)
print(f"✓ businesses_enriched.csv  ({len(biz_export):,} filas)")

# CSV 2: serie temporal mensual
monthly = df.groupby([df["date"].dt.to_period("M")]).agg(
    reviews=("review_id", "count"),
    avg_stars=("review_stars", "mean"),
).reset_index()
monthly["date"] = monthly["date"].dt.to_timestamp()
monthly.to_csv(OUTPUT_DIR / "monthly_reviews.csv", index=False)
print(f"✓ monthly_reviews.csv      ({len(monthly):,} filas)")

# CSV 3: categorías agregadas
cat_min.to_csv(OUTPUT_DIR / "categories_agg.csv", index=False)
print(f"✓ categories_agg.csv       ({len(cat_min):,} filas)")

# CSV 4: reseñas sin texto (para Tableau relacional)
reviews_lite = df[[
    "review_id", "business_id", "user_id", "review_stars",
    "review_useful", "review_funny", "review_cool", "date", "year"
]].copy()
reviews_lite.to_csv(OUTPUT_DIR / "reviews_lite.csv", index=False)
print(f"✓ reviews_lite.csv         ({len(reviews_lite):,} filas)")

print("\n¡EDA completo! Revisa la carpeta eda_output/")
print()
print("══════════════════════════════════════════════════════")
print("ACCIÓN REQUERIDA: Agregar datos geográficos")
print("══════════════════════════════════════════════════════")
print("Los parquets procesados NO tienen city/state/lat/lon.")
print("Para el dashboard 'dónde abrir un negocio' debes:")
print("1. Obtener yelp_academic_dataset_business.json")
print("2. Extraer: business_id, city, state, latitude, longitude, postal_code")
print("3. Hacer merge con businesses_enriched.csv por business_id")
print("4. Cargar el CSV enriquecido en Tableau junto con el mapa geográfico")
