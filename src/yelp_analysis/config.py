from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class RunConfig:
    output_root: str = "outputs"
    storage_mode: str = "local"


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


@dataclass
class FilterConfig:
    min_review_count: int = 5
    only_open: bool = True
    coord_lat_min: float = 24.0
    coord_lat_max: float = 72.0
    coord_lon_min: float = -141.0
    coord_lon_max: float = -52.0
    min_category_businesses: int = 50


@dataclass
class FeatureConfig:
    divergence_score: bool = True
    quadrant_norm_threshold: float = 0.5
    log_review_count: bool = True


@dataclass
class TableauConfig:
    export_csv: bool = True
    output_dir: str = "outputs/gold"


@dataclass
class GcpConfig:
    enabled: bool = False
    bucket: str = "yelp-dv-grupo"
    prefix: str = "runs"


@dataclass
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    tableau: TableauConfig = field(default_factory=TableauConfig)
    gcp: GcpConfig = field(default_factory=GcpConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            run=RunConfig(**raw.get("run", {})),
            data=DataConfig(**raw.get("data", {})),
            filters=FilterConfig(**raw.get("filters", {})),
            features=FeatureConfig(**raw.get("features", {})),
            tableau=TableauConfig(**raw.get("tableau", {})),
            gcp=GcpConfig(**raw.get("gcp", {})),
        )

    @property
    def bronze_dir(self) -> Path:
        return Path(self.run.output_root) / "bronze"

    @property
    def silver_dir(self) -> Path:
        return Path(self.run.output_root) / "silver"

    @property
    def gold_dir(self) -> Path:
        return Path(self.run.output_root) / "gold"