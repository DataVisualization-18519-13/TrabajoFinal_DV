from __future__ import annotations

from yelp_analysis.config import PipelineConfig
from yelp_analysis.medallion.bronze import (
    load_raw_business,
    load_raw_reviews,
    write_bronze_business,
    write_bronze_reviews,
)
from yelp_analysis.medallion.silver import (
    build_silver_business,
    build_silver_reviews,
    build_silver_categories,
    build_silver_monthly,
)
from yelp_analysis.medallion.gold import (
    build_gold_businesses,
    build_gold_reviews_lite,
    build_gold_categories,
    build_gold_monthly,
)


def run(cfg: PipelineConfig) -> None:
    print("=== PIPELINE YELP DV — ONE SHOT ===\n")

    # BRONZE
    print("── Bronze ──")
    raw_biz     = load_raw_business(cfg)
    raw_reviews = load_raw_reviews(cfg)
    bronze_biz  = write_bronze_business(raw_biz, cfg)
    bronze_rev  = write_bronze_reviews(raw_reviews, cfg)

    # SILVER
    print("\n── Silver ──")
    silver_biz  = build_silver_business(bronze_biz, cfg)
    silver_rev  = build_silver_reviews(bronze_rev, cfg)
    categories  = build_silver_categories(silver_biz, cfg)
    monthly     = build_silver_monthly(silver_rev, cfg)

    # GOLD
    print("\n── Gold ──")
    build_gold_businesses(silver_biz, cfg)
    build_gold_reviews_lite(bronze_rev, cfg)
    build_gold_categories(categories, cfg)
    build_gold_monthly(monthly, cfg)

    print("\n=== PIPELINE COMPLETO ===")


if __name__ == "__main__":
    cfg = PipelineConfig.from_yaml("configs/pipeline.yaml")
    run(cfg)