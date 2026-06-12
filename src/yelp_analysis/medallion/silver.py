from __future__ import annotations

import numpy as np
import pandas as pd
from yelp_analysis.config import PipelineConfig
from yelp_analysis.storage import write_table


def build_silver_business(bronze_biz: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = bronze_biz.copy()

    df["primary_category"] = (
        df["categories"].fillna("").str.split(",").str[0].str.strip()
    )
    df["log_review_count"] = np.log1p(df["review_count"].astype(float))

    mn_s, mx_s = df["stars"].min(), df["stars"].max()
    mn_r, mx_r = df["log_review_count"].min(), df["log_review_count"].max()
    df["stars_norm"]  = (df["stars"] - mn_s) / (mx_s - mn_s)
    df["log_rc_norm"] = (df["log_review_count"] - mn_r) / (mx_r - mn_r)
    df["divergence_score"] = df["stars_norm"] - df["log_rc_norm"]

    thr = cfg.features.quadrant_norm_threshold
    df["quadrant"] = "Low Q / Low Pop"
    df.loc[(df["stars_norm"] >= thr) & (df["log_rc_norm"] >= thr), "quadrant"] = "High Q / High Pop"
    df.loc[(df["stars_norm"] >= thr) & (df["log_rc_norm"] <  thr), "quadrant"] = "High Q / Low Pop (Oportunidad)"
    df.loc[(df["stars_norm"] <  thr) & (df["log_rc_norm"] >= thr), "quadrant"] = "Low Q / High Pop"

    write_table(df, "silver", "business", cfg)
    return df


def build_silver_reviews(bronze_rev: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = bronze_rev.copy()
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    write_table(df, "silver", "reviews", cfg)
    return df


def build_silver_categories(silver_biz: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    for _, row in silver_biz.iterrows():
        for cat in str(row["categories"]).split(","):
            rows.append({
                "cat":          cat.strip(),
                "stars":        row["stars"],
                "review_count": row["review_count"],
                "divergence":   row["divergence_score"],
                "state":        row["state"],
                "quadrant":     row["quadrant"],
            })
    cat_df = pd.DataFrame(rows)
    cat_agg = (
        cat_df.groupby("cat")
        .agg(
            n_negocios    =("stars", "count"),
            avg_stars     =("stars", "mean"),
            avg_rc        =("review_count", "mean"),
            avg_divergence=("divergence", "mean"),
        )
        .reset_index()
    )
    result = cat_agg[cat_agg["n_negocios"] >= cfg.filters.min_category_businesses].copy()
    n_op = cat_df[cat_df["quadrant"] == "High Q / Low Pop (Oportunidad)"].groupby("cat")["stars"].count().rename("n_oportunidad")
    result = result.join(n_op, on="cat").fillna({"n_oportunidad": 0})
    result["n_oportunidad"] = result["n_oportunidad"].astype(int)
    result["pct_oportunidad"] = (result["n_oportunidad"] / result["n_negocios"] * 100).round(1)
    write_table(result, "silver", "categories", cfg)
    return result


def build_silver_monthly(silver_rev: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    monthly = (
        silver_rev.groupby(pd.to_datetime(silver_rev["date"]).dt.to_period("M"))
        .agg(reviews=("review_id", "count"), avg_stars=("review_stars", "mean"))
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()
    write_table(monthly, "silver", "monthly_reviews", cfg)
    return monthly