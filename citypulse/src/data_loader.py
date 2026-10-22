"""
data_loader.py - High-Performance Cached Data Loading for CityPulse
Loads sensor metadata, graph adjacency matrix, and traffic observations
with graceful fallback handling for cloud deployments.
"""

import os
import pickle
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from config import resolve_data_path

_SENSOR_LOCS_CACHE = None
_ADJ_MX_CACHE = None
_LATEST_SNAPSHOT_CACHE = None
_RECENT_WINDOW_CACHE = None

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
    
    If data/train.parquet is missing (e.g. in Streamlit Cloud deployment where files > 100MB 
    are omitted from Git), automatically falls back to data/snapshot_cache.parquet.
    """
    parquet_path = resolve_data_path("train.parquet")
    using_fallback = False
    
    if not parquet_path.exists():
        fallback_path = resolve_data_path("snapshot_cache.parquet")
        if fallback_path.exists():
            parquet_path = fallback_path
            using_fallback = True
        else:
            raise FileNotFoundError(
                f"Primary highway dataset not found at expected path: '{parquet_path}'.\n"
                "Diagnostics:\n"
                "  - train.parquet (157 MB, 4,962,618 rows) exceeds GitHub's 100 MB single-file limit.\n"
                "  - It is located in your local development workspace under 'citypulse/data/train.parquet'.\n"
                "  - To deploy to Streamlit Community Cloud: track the file with Git LFS (git lfs track 'citypulse/data/train.parquet') "
                "or configure an external object store via the CITYPULSE_DATA_DIR environment variable."
            )
    
    cols_to_read = [
        "node_id", "t0_timestamp", "x_t+0_d0", "x_t+0_d1",
        "y_t+3_d0", "y_t+6_d0", "y_t+12_d0"
    ] + [f"x_t-{i}_d0" for i in range(1, 12)]
    
    if timestamp is None or using_fallback:
        table = pq.read_table(parquet_path, columns=cols_to_read)
        num_rows = len(table)
        slice_len = min(1000, num_rows)
        last_rows = table.slice(num_rows - slice_len, slice_len).to_pandas()
        latest_ts = last_rows["t0_timestamp"].max()
        snapshot = last_rows[last_rows["t0_timestamp"] == latest_ts].drop_duplicates(subset=["node_id"])
    else:
        table = pq.read_table(
            parquet_path, 
            columns=cols_to_read, 
            filters=[("t0_timestamp", "==", timestamp)]
        )
        snapshot = table.to_pandas().drop_duplicates(subset=["node_id"])
        
    # Merge with sensor location metadata
    locs = get_sensor_locations()
    merged = pd.merge(snapshot, locs, on="node_id", how="left")
    merged["is_fallback_mode"] = using_fallback
    return merged

def get_recent_window_cache() -> pd.DataFrame:
    """Loads and caches recent observations (~24-36h across all 207 nodes)."""
    global _RECENT_WINDOW_CACHE
    if _RECENT_WINDOW_CACHE is None:
        parquet_path = resolve_data_path("train.parquet")
        if not parquet_path.exists():
            fallback_path = resolve_data_path("recent_telemetry_cache.parquet")
            if fallback_path.exists():
                parquet_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Telemetry history not found. Expected: '{parquet_path}' or '{fallback_path}'."
                )
        
        meta = pq.read_metadata(parquet_path)
        total_rows = meta.num_rows
        cols = ["node_id", "t0_timestamp", "x_t+0_d0", "x_t+0_d1", "y_t+3_d0"]
        table = pq.read_table(parquet_path, columns=cols)
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
