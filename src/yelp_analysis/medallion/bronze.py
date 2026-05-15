from __future__ import annotations

import json
import glob
import pandas as pd
from pathlib import Path

from src.yelp_analysis.config import PipelineConfig


def load_raw_business(cfg: PipelineConfig) -> pd.DataFrame:
    path = Path(cfg.data.raw_dir) / "yelp_academic_dataset_business.json"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            rows.append({
                "business_id":  b.get("business_id"),
                "name":         b.get("name"),
                "city":         b.get("city"),
                "state":        b.get("state"),
                "latitude":     b.get("latitude"),
                "longitude":    b.get("longitude"),
                "postal_code":  b.get("postal_code"),
                "stars":        b.get("stars"),
                "review_count": b.get("review_count"),
                "is_open":      b.get("is_open"),
                "categories":   b.get("categories"),
            })
    return pd.DataFrame(rows)


def load_raw_reviews(cfg: PipelineConfig) -> pd.DataFrame:
    files = sorted(glob.glob(
        str(Path(cfg.data.processed_dir) / "reviews_enriched_v1_part_*.parquet")
    ))
    if not files:
        raise FileNotFoundError(f"No parquets encontrados en {cfg.data.processed_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def write_bronze_business(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    f = cfg.filters

    df["stars"]        = pd.to_numeric(df["stars"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").astype("Int64")
    df["latitude"]     = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"]    = pd.to_numeric(df["longitude"], errors="coerce")
    df["is_open"]      = pd.to_numeric(df["is_open"], errors="coerce").astype("Int64")

    df["city"]  = df["city"].str.strip().str.title()
    df["state"] = df["state"].str.strip().str.upper()

    if f.only_open:
        df = df[df["is_open"] == 1].copy()
    df = df[df["review_count"] >= f.min_review_count].copy()
    df = df[
        df["latitude"].between(f.coord_lat_min, f.coord_lat_max) &
        df["longitude"].between(f.coord_lon_min, f.coord_lon_max)
    ].copy()

    df = df.dropna(subset=["business_id", "stars", "latitude", "longitude"])
    df = df.reset_index(drop=True)

    out = cfg.bronze_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "business.parquet", index=False)
    df.to_csv(out / "business.csv", index=False)
    print(f"[bronze] business: {len(df):,} filas → {out}")
    return df


def write_bronze_reviews(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df["review_stars"]  = pd.to_numeric(df["review_stars"], errors="coerce")
    df["review_useful"] = pd.to_numeric(df["review_useful"], errors="coerce").fillna(0)
    df["review_funny"]  = pd.to_numeric(df["review_funny"], errors="coerce").fillna(0)
    df["review_cool"]   = pd.to_numeric(df["review_cool"], errors="coerce").fillna(0)

    df = df.dropna(subset=["review_id", "business_id", "review_stars", "date"])
    df = df.reset_index(drop=True)

    out = cfg.bronze_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "reviews.parquet", index=False)
    print(f"[bronze] reviews: {len(df):,} filas → {out}")
    return df
