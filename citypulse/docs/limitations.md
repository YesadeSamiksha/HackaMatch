# CityPulse Known Limitations & Integrity Boundaries

In accordance with rigorous data science ethics, the following limitations are formally documented:

## 1. Primary Dataset (Mumbai Traffic Police 2024 Report)
- **Enforcement Bias:** Challan numbers measure enforcement operations, camera installations, and officer nakabandis. A higher count in Multimedia Cell or Mankhurd indicates active monitoring, not necessarily higher driver recklessness.
- **Temporal Aggregation:** The report contains annual cumulative figures for 2024. Granular time-of-day or day-of-week timestamps are not provided in the PDF tables.
- **Spatial Resolution:** Data is aggregated at the Division and Police Station level. Specific intersection coordinates, latitude/longitude points, or lane-level geometry are not present.

## 2. Secondary Dataset (Maharashtra Traffic CSV)
- **Unverified Measurement Provenance:** The acquisition methodology, sensor calibration, and sampling frequency for speed and vehicle counts are unconfirmed.
- **Prototyping Scope:** Retained exclusively for interface prototyping and filter wiring. Not suitable for municipal policy or physical road engineering decisions.

## 3. Rainfall Dataset (IMD Daily District Archive)
- **Macro Geographic Unit:** District-wide rainfall numbers cannot identify localized waterlogged underpasses (e.g. Milan Subway, King's Circle) or localized cloudbursts.
- **Limited Timeline:** Observations cover late monsoon transition dates present in the archive (August to October 2026).

## 4. Vehicle Registration Dataset
- **Missing Municipal Demarcation:** The CSV provides vehicle tallies without city-level geographical attribution.
- **Fleet Stock vs On-Road Activity:** Registrations indicate licensed vehicle ownership, which is fundamentally distinct from active peak-hour traffic volume.

## 5. Machine Learning Boundaries
- **No Training in Phase 1:** Machine learning models have deliberately NOT been trained, as verified continuous physical sensor streams with a documented target variable are required first.
