from __future__ import annotations

import pandas as pd
from yelp_analysis.config import PipelineConfig
from yelp_analysis.storage import read_json_lines, read_parquets, write_table


def load_raw_business(cfg: PipelineConfig) -> pd.DataFrame:
    rows = read_json_lines(cfg, "yelp_academic_dataset_business.json")
    return pd.DataFrame([{
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
    } for b in rows])


def load_raw_reviews(cfg: PipelineConfig) -> pd.DataFrame:
    return read_parquets(cfg)


def write_bronze_business(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    f = cfg.filters
    df["stars"]        = pd.to_numeric(df["stars"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").astype("Int64")
    df["latitude"]     = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"]    = pd.to_numeric(df["longitude"], errors="coerce")
    df["is_open"]      = pd.to_numeric(df["is_open"], errors="coerce").astype("Int64")
    df["city"]         = df["city"].str.strip().str.title()
    df["state"]        = df["state"].str.strip().str.upper()

    if f.only_open:
        df = df[df["is_open"] == 1].copy()
    df = df[df["review_count"] >= f.min_review_count].copy()
    df = df[
        df["latitude"].between(f.coord_lat_min, f.coord_lat_max) &
        df["longitude"].between(f.coord_lon_min, f.coord_lon_max)
    ].copy()
    df = df.dropna(subset=["business_id", "stars", "latitude", "longitude"])
    df = df.reset_index(drop=True)

    write_table(df, "bronze", "business", cfg)
    return df


def write_bronze_reviews(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df["review_stars"]  = pd.to_numeric(df["review_stars"], errors="coerce")
    df["review_useful"] = pd.to_numeric(df["review_useful"], errors="coerce").fillna(0)
    df["review_funny"]  = pd.to_numeric(df["review_funny"], errors="coerce").fillna(0)
    df["review_cool"]   = pd.to_numeric(df["review_cool"], errors="coerce").fillna(0)
    df = df.dropna(subset=["review_id", "business_id", "review_stars", "date"])
    df = df.reset_index(drop=True)

    write_table(df, "bronze", "reviews", cfg)
    return df