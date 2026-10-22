"""
traffic_patterns.py - Recurring Hotspot and Rush-Hour Intelligence Engine
Computes hour-of-day x day-of-week heatmaps, morning vs evening peak comparisons,
sensor-specific recurrence frequencies, and automated evidence-backed insights.
"""

import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

_HISTORICAL_PATTERNS_CACHE = None

DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def analyze_traffic_patterns(sample_size: int = 120000, random_seed: int = 42) -> dict:
    """
    Analyzes historical patterns across hours, weekdays, and sensors.
    Uses representative chronological sampling from train.parquet to guarantee fast computation.
    """
    global _HISTORICAL_PATTERNS_CACHE
    if _HISTORICAL_PATTERNS_CACHE is not None:
        return _HISTORICAL_PATTERNS_CACHE

    import pickle
    from config import resolve_data_path

    # Check precomputed cache first for instant sub-second page loads
    cache_pkl = resolve_data_path("traffic_patterns_cache.pkl")
    if cache_pkl.exists():
        with open(cache_pkl, "rb") as f:
            _HISTORICAL_PATTERNS_CACHE = pickle.load(f)
        return _HISTORICAL_PATTERNS_CACHE

    from src.data_loader import get_dataset_parquet_path
    parquet_path = get_dataset_parquet_path()

    meta = pq.read_metadata(str(parquet_path))
    total_rows = meta.num_rows

    cols = ["node_id", "t0_timestamp", "x_t+0_d0", "x_t+0_d1"]
    table = pq.read_table(str(parquet_path), columns=cols)
    
    np.random.seed(random_seed)
    if total_rows > sample_size:
        indices = np.random.choice(total_rows, size=sample_size, replace=False)
        df = table.take(indices).to_pandas()
    else:
        df = table.to_pandas()

    # Parse timestamps and extract calendar components
    df["ts"] = pd.to_datetime(df["t0_timestamp"])
    df["hour"] = df["ts"].dt.hour
    df["day_name"] = df["ts"].dt.day_name()
    df["day_idx"] = df["ts"].dt.dayofweek
    df["speed"] = df["x_t+0_d0"]
    df["is_congested"] = (df["speed"] < 35.0).astype(int)

    # 1. Hour x Day Heatmap of Congestion Probability and Average Speed
    heatmap_speed = df.pivot_table(
        index="hour", 
        columns="day_name", 
        values="speed", 
        aggfunc="mean"
    ).reindex(columns=[d for d in DAYS_ORDER if d in df["day_name"].unique()])

    heatmap_congestion = df.pivot_table(
        index="hour", 
        columns="day_name", 
        values="is_congested", 
        aggfunc="mean"
    ).reindex(columns=[d for d in DAYS_ORDER if d in df["day_name"].unique()]) * 100.0

    # 2. Hourly Speed Profile across All Corridors
    hourly_summary = df.groupby("hour").agg(
        avg_speed=("speed", "mean"),
        p25_speed=("speed", lambda x: np.percentile(x, 25)),
        p75_speed=("speed", lambda x: np.percentile(x, 75)),
        congestion_rate=("is_congested", "mean")
    ).reset_index()

    # 3. Morning Peak vs Evening Peak Analysis
    # Morning: 07:00 - 10:00; Evening: 16:00 - 19:00; Midday: 11:00 - 14:00; Night: 22:00 - 05:00
    df_morning = df[df["hour"].between(7, 9)]
    df_evening = df[df["hour"].between(16, 18)]
    df_midday = df[df["hour"].between(11, 13)]

    morn_speed = df_morning["speed"].mean()
    eve_speed = df_evening["speed"].mean()
    midday_speed = df_midday["speed"].mean()

    # 4. Sensor-Specific Congestion Recurrence & Persistent Hotspots
    # Definition: Persistent if sensor is congested in >= 15% of all historical readings
    sensor_stats = df.groupby("node_id").agg(
        obs_count=("speed", "count"),
        avg_speed=("speed", "mean"),
        congestion_rate=("is_congested", "mean"),
        morning_speed=("speed", lambda s: s[df.loc[s.index, "hour"].between(7, 9)].mean() if len(s[df.loc[s.index, "hour"].between(7, 9)]) > 0 else np.nan),
        evening_speed=("speed", lambda s: s[df.loc[s.index, "hour"].between(16, 18)].mean() if len(s[df.loc[s.index, "hour"].between(16, 18)]) > 0 else np.nan)
    ).reset_index()

    # Map sensor corridor metadata
    locs_path = os.path.join("data", "sensor_locations_enriched.csv")
    if os.path.exists(locs_path):
        locs = pd.read_csv(locs_path)
        if "node_id" not in locs.columns and "index" in locs.columns:
            locs["node_id"] = locs["index"]
        sensor_stats = pd.merge(sensor_stats, locs[["node_id", "sensor_id", "corridor", "latitude", "longitude"]], on="node_id", how="left")
    else:
        sensor_stats["sensor_id"] = sensor_stats["node_id"]
        sensor_stats["corridor"] = "Highway Corridor"

    sensor_stats["is_persistent"] = sensor_stats["congestion_rate"] >= 0.15
    persistent_ranking = sensor_stats.sort_values(by="congestion_rate", ascending=False).reset_index(drop=True)
    persistent_ranking["rank"] = persistent_ranking.index + 1

    # Map for hotspot_ranking recurrence feature
    recurrence_map = dict(zip(sensor_stats["node_id"], sensor_stats["congestion_rate"]))

    # 5. Generate Exactly Three Evidence-Backed Insights
    # Template: "Sensor {sensor_id} recorded an average speed of {observed_speed} mph during {period}, compared with {baseline_speed} mph during {comparison_period}."
    
    top_persistent = persistent_ranking.iloc[0]
    top_eve_bottleneck = sensor_stats.dropna(subset=["evening_speed"]).sort_values(by="evening_speed", ascending=True).iloc[0]
    top_morn_bottleneck = sensor_stats.dropna(subset=["morning_speed"]).sort_values(by="morning_speed", ascending=True).iloc[0]

    insight_1 = (
        f"Sensor Station #{top_persistent['sensor_id']} ({top_persistent['corridor']}) recorded an average speed of "
        f"{top_persistent['avg_speed']:.1f} mph during overall monitoring ({top_persistent['congestion_rate']*100:.1f}% congestion recurrence), "
        f"compared with {df['speed'].mean():.1f} mph during network-wide baseline conditions (supported by {top_persistent['obs_count']:,} observations)."
    )

    insight_2 = (
        f"Sensor Station #{top_eve_bottleneck['sensor_id']} ({top_eve_bottleneck['corridor']}) recorded an average speed of "
        f"{top_eve_bottleneck['evening_speed']:.1f} mph during the evening rush hour (4:00 PM – 7:00 PM), "
        f"compared with {top_eve_bottleneck['avg_speed']:.1f} mph during 24-hour baseline operations."
    )

    insight_3 = (
        f"Sensor Station #{top_morn_bottleneck['sensor_id']} ({top_morn_bottleneck['corridor']}) recorded an average speed of "
        f"{top_morn_bottleneck['morning_speed']:.1f} mph during the morning rush hour (7:00 AM – 10:00 AM), "
        f"compared with {midday_speed:.1f} mph during midday off-peak hours (11:00 AM – 2:00 PM)."
    )

    results = {
        "heatmap_speed": heatmap_speed,
        "heatmap_congestion": heatmap_congestion,
        "hourly_summary": hourly_summary,
        "morn_speed": morn_speed,
        "eve_speed": eve_speed,
        "midday_speed": midday_speed,
        "persistent_ranking": persistent_ranking,
        "recurrence_map": recurrence_map,
        "insights": [insight_1, insight_2, insight_3]
    }

    _HISTORICAL_PATTERNS_CACHE = results
    return results
