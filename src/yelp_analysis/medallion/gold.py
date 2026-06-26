from __future__ import annotations

import numpy as np
import pandas as pd

from yelp_analysis.config import PipelineConfig
from yelp_analysis.storage import write_table


SEGMENTS = [
    {
        "segment_key": 1,
        "segment_name": "High Q / High Pop",
        "quality_band": "High",
        "popularity_band": "High",
        "is_opportunity": False,
    },
    {
        "segment_key": 2,
        "segment_name": "High Q / Low Pop (Oportunidad)",
        "quality_band": "High",
        "popularity_band": "Low",
        "is_opportunity": True,
    },
    {
        "segment_key": 3,
        "segment_name": "Low Q / High Pop",
        "quality_band": "Low",
        "popularity_band": "High",
        "is_opportunity": False,
    },
    {
        "segment_key": 4,
        "segment_name": "Low Q / Low Pop",
        "quality_band": "Low",
        "popularity_band": "Low",
        "is_opportunity": False,
    },
]


def _build_dim_business(silver_biz: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "business_id",
        "name",
        "city",
        "state",
        "postal_code",
        "latitude",
        "longitude",
    ]
    dim = (
        silver_biz[cols]
        .drop_duplicates(subset=["business_id"])
        .sort_values("business_id")
        .reset_index(drop=True)
    )
    dim.insert(0, "business_key", np.arange(1, len(dim) + 1, dtype=np.int64))
    return dim


def _build_dim_date(review_dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(review_dates, errors="coerce").dropna().dt.normalize()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    dim = pd.DataFrame({"date": calendar})
    dim.insert(0, "date_key", dim["date"].dt.strftime("%Y%m%d").astype(int))
    dim["year"] = dim["date"].dt.year
    dim["quarter"] = "Q" + dim["date"].dt.quarter.astype(str)
    dim["month"] = dim["date"].dt.month
    dim["month_name"] = dim["date"].dt.month_name()
    dim["year_month"] = dim["date"].dt.strftime("%Y-%m")
    dim["day"] = dim["date"].dt.day
    dim["day_of_week"] = dim["date"].dt.dayofweek + 1
    dim["day_name"] = dim["date"].dt.day_name()
    dim["week_of_year"] = dim["date"].dt.isocalendar().week.astype(int)
    dim["is_weekend"] = dim["date"].dt.dayofweek >= 5
    return dim


def _build_category_model(
    silver_biz: pd.DataFrame, business_lookup: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    categories = (
        silver_biz[["business_id", "categories"]]
        .assign(category_name=lambda x: x["categories"].fillna("").str.split(","))
        .explode("category_name")
    )
    categories["category_name"] = categories["category_name"].str.strip()
    categories = categories[categories["category_name"] != ""].copy()
    categories["category_order"] = categories.groupby("business_id").cumcount() + 1
    categories = categories.drop_duplicates(
        subset=["business_id", "category_name"], keep="first"
    )

    dim_category = (
        categories[["category_name"]]
        .drop_duplicates()
        .sort_values("category_name")
        .reset_index(drop=True)
    )
    dim_category.insert(
        0, "category_key", np.arange(1, len(dim_category) + 1, dtype=np.int64)
    )

    bridge = (
        categories.merge(business_lookup, on="business_id", validate="many_to_one")
        .merge(dim_category, on="category_name", validate="many_to_one")
        [["business_key", "category_key", "category_order"]]
        .sort_values(["business_key", "category_order"])
        .reset_index(drop=True)
    )
    bridge["is_first_listed"] = bridge["category_order"] == 1
    return dim_category, bridge


def build_gold_star_schema(
    silver_biz: pd.DataFrame,
    bronze_rev: pd.DataFrame,
    cfg: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    """Build Tableau-ready facts, dimensions, and the category bridge."""
    dim_business = _build_dim_business(silver_biz)
    business_lookup = dim_business[["business_key", "business_id"]]

    dim_segment = pd.DataFrame(SEGMENTS)
    segment_lookup = dim_segment[["segment_key", "segment_name"]]

    fact_business = (
        silver_biz.merge(business_lookup, on="business_id", validate="one_to_one")
        .merge(
            segment_lookup,
            left_on="quadrant",
            right_on="segment_name",
            validate="many_to_one",
        )
        [
            [
                "business_key",
                "segment_key",
                "stars",
                "review_count",
                "log_review_count",
                "stars_norm",
                "log_rc_norm",
                "divergence_score",
            ]
        ]
        .copy()
    )
    fact_business["business_count"] = 1

    dim_category, bridge_category = _build_category_model(
        silver_biz, business_lookup
    )

    reviews = bronze_rev.merge(
        business_lookup, on="business_id", how="inner", validate="many_to_one"
    )
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews = reviews.dropna(subset=["date"]).copy()
    reviews["date_key"] = reviews["date"].dt.strftime("%Y%m%d").astype(int)
    fact_review = reviews[
        [
            "review_id",
            "business_key",
            "user_id",
            "date_key",
            "review_stars",
            "review_useful",
            "review_funny",
            "review_cool",
        ]
    ].copy()
    fact_review["review_count"] = 1
    dim_date = _build_dim_date(reviews["date"])

    tables = {
        "dim_business": dim_business,
        "dim_date": dim_date,
        "dim_category": dim_category,
        "dim_opportunity_segment": dim_segment,
        "bridge_business_category": bridge_category,
        "fact_business_opportunity": fact_business,
        "fact_review": fact_review,
    }
    for name, table in tables.items():
        write_table(table, "gold", name, cfg)

    dropped_reviews = len(bronze_rev) - len(fact_review)
    print(
        f"[gold] integridad: {dropped_reviews:,} resenas fuera del universo "
        "de negocios fueron excluidas"
    )
    return tables
