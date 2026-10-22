# Feature Feasibility Report: Engineering Strategy & Leakage Prevention

**Platform:** CityPulse TrafficPulse AI  
**Dataset:** `data/train.parquet`  

---

## 1. Primary Available Raw Features

| Feature Name | Source | Description | Timing Availability | Leakage Risk |
| :--- | :--- | :--- | :--- | :---: |
| `node_id` | Metadata | Integer ID of sensor (0-206) | Present at $t_0$ | None |
| `t0_timestamp` | Header | Reference timestamp | Present at $t_0$ | None |
| `x_t+0_d0` | Sensor loop | Current speed at $t_0$ (mph) | Measured at $t_0$ | None |
| `x_t-1_d0` to `x_t-11_d0` | Historical buffer | Speeds at 5 to 55 min prior | Historical buffer | None |
| `x_t+0_d1` | Sensor loop | Current normalized day fraction | Measured at $t_0$ | None |
| `x_t-1_d1` to `x_t-11_d1` | Historical buffer | Diurnal fractions at prior steps | Historical buffer | None |

---

## 2. Engineered Temporal & Kinematic Features

To capture traffic shockwaves and congestion onset without future leakage, the following features are derived strictly from $t \le t_0$:

1. **Short-Term Speed Trend (Acceleration):**
   `Delta_v_15m = x_t+0_d0 - x_t-3_d0`
   Negative values indicate rapid deceleration / queue formation.
2. **Medium-Term Speed Trend:**
   `Delta_v_30m = x_t+0_d0 - x_t-6_d0`
3. **Rolling Mean Speed (15-min and 30-min windows):**
   `v_mean_15m = mean(x_t-3_d0 ... x_t+0_d0)`
   `v_mean_30m = mean(x_t-6_d0 ... x_t+0_d0)`
4. **Rolling Speed Volatility (Standard Deviation):**
   Measures turbulence and flow instability preceding breakdown.
5. **Diurnal Time Features:**
   - `hour_of_day`: Derived from timestamp / $d_1$ ($0$ to $23$).
   - `is_peak_morning`: Boolean flag for 07:00 - 10:00.
   - `is_peak_evening`: Boolean flag for 16:00 - 19:00.
   - `is_weekend`: Boolean flag for Saturday / Sunday.
6. **Spatial Location Attributes (from `sensor_locations_enriched.csv`):**
   - Latitude, Longitude, and Freeway Corridor categorical identifier.
