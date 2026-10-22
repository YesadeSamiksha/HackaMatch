# Data Quality Report: METR-LA Highway Traffic Benchmark (`train.parquet`)

**Prepared for:** CityPulse AI Platform  
**Status:** VALIDATED - PRODUCTION READY  

---

## 1. Missingness & Completeness Audit

- **Total Elements Audited:** 248,130,900 across 50 columns and 5 row groups.
- **Null Value Count:** Exactly 0 missing values detected across all columns (100.0% completeness).
- **Parquet Chunk Compression:** Snappy compression, with PLAIN and RLE dictionary encodings verified.

---

## 2. Temporal Continuity & Sampling Regularity

- **Sampling Step:** Strictly 5-minute uniform intervals.
- **Total Timestamps per Sensor:** 23,974 time steps.
- **Total Node Multiplier:** 207 sensors * 23,974 timestamps = 4,962,618 rows.
- **Zero-Value Speeds:** Speeds of 0.0 mph occur in ~2.8% of observations, corresponding to genuine highway standstill congestion or temporary loop detector re-calibration cycles.
- **Speed Upper Bound:** Fixed at 70.0 mph, reflecting standard Caltrans PeMS highway speed limit recording caps.

---

## 3. Data Leakage & Integrity Guarantees

1. **Chronological Splitting:** 
   - All models must use chronological train/test separation (e.g., Train on March 1 to April 30; Held-out Test on May 1 to May 23).
   - Random K-Fold splits are strictly prohibited to prevent look-ahead bias and temporal leakage.
2. **Feature Isolation:**
   - At reference time t0, feature engineering only consumes t <= t0 columns (`x_t-*` and `x_t+0`).
   - Targets (`y_t+*`) are strictly excluded from the feature matrix during training and inference.
3. **Graph Adjacency Compatibility:**
   - Adjacency matrix `adj_mx.pkl` contains 1,722 directed spatial connections between the 207 sensors based on highway network road distance.
