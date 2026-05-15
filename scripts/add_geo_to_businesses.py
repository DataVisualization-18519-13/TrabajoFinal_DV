"""
Agrega columnas geográficas (city, state, latitude, longitude, postal_code)
a businesses_enriched.csv usando yelp_academic_dataset_business.json.
"""

import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("eda_output")
OUTPUT_DIR = Path("eda_output")

print("Leyendo business.json...")
geo_rows = []
with open(RAW_DIR / "yelp_academic_dataset_business.json", encoding="utf-8") as f:
    for line in f:
        b = json.loads(line)
        geo_rows.append({
            "business_id":  b.get("business_id"),
            "city":         b.get("city"),
            "state":        b.get("state"),
            "latitude":     b.get("latitude"),
            "longitude":    b.get("longitude"),
            "postal_code":  b.get("postal_code"),
        })

geo_df = pd.DataFrame(geo_rows)
print(f"  {len(geo_df):,} negocios leídos de business.json")

print("Leyendo businesses_enriched.csv...")
biz = pd.read_csv(PROCESSED_DIR / "businesses_enriched.csv")
print(f"  {len(biz):,} filas en businesses_enriched.csv")

print("Haciendo merge...")
biz_geo = biz.merge(geo_df, on="business_id", how="left")

missing_geo = biz_geo["city"].isna().sum()
print(f"  Negocios sin match geográfico: {missing_geo:,} ({missing_geo/len(biz_geo)*100:.1f}%)")

out_path = OUTPUT_DIR / "businesses_enriched_geo.csv"
biz_geo.to_csv(out_path, index=False)
print(f"\n✓ Guardado: {out_path}  ({len(biz_geo):,} filas, {len(biz_geo.columns)} columnas)")
print(f"  Columnas: {biz_geo.columns.tolist()}")
