from __future__ import annotations

import pandas as pd
from yelp_analysis.config import PipelineConfig
from yelp_analysis.storage import write_table


def build_gold_businesses(silver_biz: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    cols = [
        "business_id", "name", "city", "state",
        "latitude", "longitude", "postal_code",
        "stars", "review_count", "categories", "primary_category",
        "log_review_count", "stars_norm", "log_rc_norm",
        "divergence_score", "quadrant",
    ]
    df = silver_biz[[c for c in cols if c in silver_biz.columns]].copy()
    write_table(df, "gold", "businesses_enriched", cfg)
    return df


def build_gold_reviews_lite(bronze_rev: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    cols = [
        "review_id", "business_id", "user_id",
        "review_stars", "review_useful", "review_funny", "review_cool",
        "date", "year",
    ]
    df = bronze_rev[[c for c in cols if c in bronze_rev.columns]].copy()
    write_table(df, "gold", "reviews_lite", cfg)
    return df


def build_gold_categories(categories: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    write_table(categories, "gold", "categories_agg", cfg)
    return categories


def build_gold_monthly(monthly: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    write_table(monthly, "gold", "monthly_reviews", cfg)
    return monthly