"""
data_loader.py - High-Performance Cached Data Loading for CityPulse
Loads sensor metadata, graph adjacency matrix, and traffic observations.
Integrates with Hugging Face Dataset (YesadeSamiksha/trafficpulse-data).
"""

import os
import pickle
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from config import (
    resolve_data_path,
    HF_DATASET_REPO_ID,
    HF_DATASET_FILENAME
)

_SENSOR_LOCS_CACHE = None
_ADJ_MX_CACHE = None
_LATEST_SNAPSHOT_CACHE = None
_RECENT_WINDOW_CACHE = None
_RESOLVED_DATASET_PATH = None

def get_dataset_parquet_path() -> Path:
    """
    Resolves the dataset parquet file path.
    1. Checks local file 'train.parquet' via resolve_data_path. If present locally, uses it immediately.
    2. If missing locally (e.g. in Streamlit Community Cloud), downloads and caches 'train.parquet'
       from Hugging Face repository 'YesadeSamiksha/trafficpulse-data' using hf_hub_download.
    3. If Hugging Face is unreachable, checks for bundled fallback caches.
    """
    global _RESOLVED_DATASET_PATH
    if _RESOLVED_DATASET_PATH is not None and Path(_RESOLVED_DATASET_PATH).exists():
        return Path(_RESOLVED_DATASET_PATH)

    # 1. Check local file system
    local_path = resolve_data_path("train.parquet")
    if local_path.exists():
        _RESOLVED_DATASET_PATH = str(local_path)
        return local_path

    # 2. Download from Hugging Face Dataset repository
    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            filename=HF_DATASET_FILENAME,
            repo_type="dataset"
        )
        _RESOLVED_DATASET_PATH = downloaded_path
        return Path(downloaded_path)
    except Exception as hf_err:
        # 3. Fallback to bundled cache if offline
        fallback_path = resolve_data_path("snapshot_cache.parquet")
        if fallback_path.exists():
            return fallback_path
        raise FileNotFoundError(
            f"Unable to load traffic dataset '{HF_DATASET_FILENAME}' from Hugging Face repository '{HF_DATASET_REPO_ID}'. "
            f"Error: {hf_err}. Also checked local path '{local_path}'."
        )

def load_traffic_data(columns: list[str] = None, filters: list = None, nrows: int = None) -> pd.DataFrame:
    """
    Reusable loader for traffic dataset using PyArrow / Pandas with Hugging Face integration.
    Supports column projection and slice limits to optimize memory on Streamlit Cloud.
    """
    parquet_path = get_dataset_parquet_path()
    
    if filters is not None:
        table = pq.read_table(str(parquet_path), columns=columns, filters=filters)
        df = table.to_pandas()
    elif nrows is not None:
        table = pq.read_table(str(parquet_path), columns=columns)
        total_rows = len(table)
        slice_len = min(nrows, total_rows)
        df = table.slice(total_rows - slice_len, slice_len).to_pandas()
    elif columns is not None:
        table = pq.read_table(str(parquet_path), columns=columns)
        df = table.to_pandas()
    else:
        df = pd.read_parquet(str(parquet_path))
        
    return df

def get_sensor_locations() -> pd.DataFrame:
    """Returns the enriched 207 sensor stations with verified coordinates and corridors."""
    global _SENSOR_LOCS_CACHE
    if _SENSOR_LOCS_CACHE is None:
        locs_path = resolve_data_path("sensor_locations_enriched.csv")
        if not locs_path.exists():
            locs_path = resolve_data_path("sensor_locations.csv")
        if not locs_path.exists():
            raise FileNotFoundError(
                f"Sensor locations file not found. Expected at: '{locs_path}'."
            )
        df = pd.read_csv(locs_path)
        if "node_id" not in df.columns and "index" in df.columns:
            df["node_id"] = df["index"]
        _SENSOR_LOCS_CACHE = df
    return _SENSOR_LOCS_CACHE.copy()

def get_adjacency_matrix() -> tuple[list, np.ndarray]:
    """Returns sensor IDs and the 207x207 network distance adjacency matrix."""
    global _ADJ_MX_CACHE
    if _ADJ_MX_CACHE is None:
        adj_path = resolve_data_path("adj_mx.pkl")
        if not adj_path.exists():
            raise FileNotFoundError(
                f"Adjacency matrix not found. Expected at: '{adj_path}'."
            )
        with open(adj_path, "rb") as f:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")
        _ADJ_MX_CACHE = (sensor_ids, adj_mx)
    return _ADJ_MX_CACHE[0], _ADJ_MX_CACHE[1].copy()

def get_network_snapshot(timestamp: str = None) -> pd.DataFrame:
    """
    Returns a complete network snapshot (all 207 nodes) for a given timestamp,
    or the latest available timestamp if not specified.
    """
    parquet_path = get_dataset_parquet_path()
    
    cols_to_read = [
        "node_id", "t0_timestamp", "x_t+0_d0", "x_t+0_d1",
        "y_t+3_d0", "y_t+6_d0", "y_t+12_d0"
    ] + [f"x_t-{i}_d0" for i in range(1, 12)]
    
    if timestamp is None:
        table = pq.read_table(str(parquet_path), columns=cols_to_read)
        num_rows = len(table)
        slice_len = min(1500, num_rows)
        last_rows = table.slice(num_rows - slice_len, slice_len).to_pandas()
        latest_ts = last_rows["t0_timestamp"].max()
        snapshot = last_rows[last_rows["t0_timestamp"] == latest_ts].drop_duplicates(subset=["node_id"])
    else:
        table = pq.read_table(
            str(parquet_path), 
            columns=cols_to_read, 
            filters=[("t0_timestamp", "==", timestamp)]
        )
        snapshot = table.to_pandas().drop_duplicates(subset=["node_id"])
        
    # Merge with sensor location metadata
    locs = get_sensor_locations()
    merged = pd.merge(snapshot, locs, on="node_id", how="left")
    merged["dataset_source"] = f"Hugging Face: {HF_DATASET_REPO_ID}" if "snapshots" in str(parquet_path) else "Local Verified Dataset"
    return merged

def get_recent_window_cache() -> pd.DataFrame:
    """Loads and caches recent observations (~24-36h across all 207 nodes)."""
    global _RECENT_WINDOW_CACHE
    if _RECENT_WINDOW_CACHE is None:
        parquet_path = get_dataset_parquet_path()
        meta = pq.read_metadata(str(parquet_path))
        total_rows = meta.num_rows
        cols = ["node_id", "t0_timestamp", "x_t+0_d0", "x_t+0_d1", "y_t+3_d0"]
        table = pq.read_table(str(parquet_path), columns=cols)
        slice_size = min(75000, total_rows)
        df_recent = table.slice(total_rows - slice_size, slice_size).to_pandas()
        df_recent["timestamp"] = pd.to_datetime(df_recent["t0_timestamp"])
        _RECENT_WINDOW_CACHE = df_recent
    return _RECENT_WINDOW_CACHE

def get_sensor_history(node_id: int, num_samples: int = 288) -> pd.DataFrame:
    """
    Fetches the recent historical observations for a specific sensor node.
    num_samples=288 corresponds to 24 hours (288 * 5 min).
    """
    df_recent = get_recent_window_cache()
    df_node = df_recent[df_recent["node_id"] == int(node_id)].copy()
    df_node = df_node.sort_values(by="timestamp").tail(num_samples).reset_index(drop=True)
    return df_node
