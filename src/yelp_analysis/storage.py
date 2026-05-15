from __future__ import annotations

import json
import io
from pathlib import Path
import pandas as pd
from yelp_analysis.config import PipelineConfig


def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def read_json_lines(cfg: PipelineConfig, filename: str) -> list[dict]:
    """Lee un JSON lines desde local o GCS."""
    if cfg.run.storage_mode == "gcp":
        client = _gcs_client()
        bucket = client.bucket(cfg.gcp.bucket)
        blob = bucket.blob(f"raw/{filename}")
        content = blob.download_as_text(encoding="utf-8")
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        path = Path(cfg.data.raw_dir) / filename
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows


def read_parquets(cfg: PipelineConfig) -> pd.DataFrame:
    """Lee los parquets de reviews desde local o GCS."""
    if cfg.run.storage_mode == "gcp":
        import glob
        from google.cloud import storage
        client = _gcs_client()
        bucket = client.bucket(cfg.gcp.bucket)
        blobs = list(bucket.list_blobs(prefix="processed/"))
        dfs = []
        for blob in blobs:
            if blob.name.endswith(".parquet"):
                data = blob.download_as_bytes()
                dfs.append(pd.read_parquet(io.BytesIO(data)))
        if not dfs:
            raise FileNotFoundError("No parquets en GCS processed/")
        df = pd.concat(dfs, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        return df
    else:
        import glob
        files = sorted(glob.glob(
            str(Path(cfg.data.processed_dir) / "reviews_enriched_v1_part_*.parquet")
        ))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        return df


def write_table(df: pd.DataFrame, layer: str, name: str, cfg: PipelineConfig) -> None:
    """Escribe CSV y parquet en local o GCS."""
    if cfg.run.storage_mode == "gcp":
        from google.cloud import storage
        client = _gcs_client()
        bucket = client.bucket(cfg.gcp.bucket)

        # parquet
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        bucket.blob(f"{layer}/{name}.parquet").upload_from_string(
            buf.getvalue(), content_type="application/octet-stream"
        )
        # csv
        csv_buf = df.to_csv(index=False).encode("utf-8")
        bucket.blob(f"{layer}/{name}.csv").upload_from_string(
            csv_buf, content_type="text/csv"
        )
        print(f"[{layer}] {name}: {len(df):,} filas → gs://{cfg.gcp.bucket}/{layer}/")
    else:
        if layer == "bronze":
            out = cfg.bronze_dir
        elif layer == "silver":
            out = cfg.silver_dir
        else:
            out = cfg.gold_dir
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / f"{name}.parquet", index=False)
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"[{layer}] {name}: {len(df):,} filas → {out}")