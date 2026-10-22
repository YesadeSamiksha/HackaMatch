# CityPulse Data Statement

**Product:** TrafficPulse AI — Predict. Explain. Act.  
**Geographic Scope:** Mumbai City, Maharashtra, India Only  
**Phase:** Phase 1 (Website Skeleton & Verified Dataset Integration)

---

## 1. Executive Summary

CityPulse adheres strictly to geographic integrity and empirical transparency. In compliance with the competition mandate:
1. All official analytics are strictly bounded to **Mumbai City**.
2. Benchmark files from Los Angeles and California PeMS (`PEMS-BAY.h5`, `adj_mx.pkl`) are **strictly excluded**.
3. Datasets lacking confirmed measurement provenance (`Traffic dataset.csv`) are prominently labelled as **Unverified Prototyping Data** and filtered to Mumbai City rows only.
4. Macro vehicle registration summaries lacking municipal demarcation (`babfb7a0-992e-487b-97dd-220d0290122e.csv`) are labelled **"Not suitable for Mumbai-only analysis"** and excluded from Mumbai KPI cards.
5. No machine learning forecasting models have been trained in Phase 1.

---

## 2. Dataset Specifics

### Dataset A — Mumbai Traffic Violations Data 2024 (Primary)
- **File Name:** `141d484b-bbb2-4b93-8797-6e88533fd506.pdf`
- **Source:** Mumbai Traffic Police Department (Official Government Report)
- **Reporting Period:** 1 January 2024 to 31 December 2024
- **Coverage:** 42 Traffic Divisions, 91 Police Stations, 24 Violation Categories, 26 Vehicle Types
- **Total Recorded Infractions:** 5,937,151 division challans (₹4,784,033,250 total amount) + 847,766 police station challans (₹670,772,400 total amount).
- **Extraction Protocol:** Vector table extraction via PyMuPDF (`fitz.find_tables()`). All extracted row sums match printed Grand Totals with 100% mathematical precision.
- **Administrative Constraint:** Division-level and police-station-level figures represent separate operational hierarchies and are maintained strictly separate.

### Dataset B — Maharashtra Traffic Dataset CSV (Secondary / Prototype)
- **File Name:** `Traffic dataset.csv`
- **Status:** **Unverified — source and measurement methodology not confirmed**
- **Raw Row Count:** 1,200 rows across Maharashtra
- **Mumbai Scoped Rows:** 234 rows (19.5% retained)
- **Excluded Non-Mumbai Rows:** 966 rows (Pune: 234, Thane: 242, Nashik: 262, Nagpur: 228) strictly eliminated
- **Documented Areas in Mumbai Subset:** Bandra (53), Malad (51), Andheri (48), Dadar (43), Lower Parel (39)
- **Safety Guardrail:** Kept isolated in the Traffic Data Explorer sandbox; never combined with official police statistics.

### Dataset C — IMD District-wise Daily Rainfall CSV (Environmental Context)
- **File Name:** `rainfall_districtwise_daily_imd.csv`
- **Source:** India Meteorological Department (IMD)
- **Reporting Period:** 2026-08-19 to 2026-10-22 (65 continuous daily records)
- **Coverage:** Mumbai City District only (all other districts excluded)
- **Decoupling Rule:** Maintained as an independent meteorological baseline; no unsupported causal claims linking rain to traffic congestion.

### Dataset D — Vehicle Registration Dataset (Fleet Reference)
- **File Name:** `babfb7a0-992e-487b-97dd-220d0290122e.csv`
- **Status:** **Not suitable for Mumbai-only analysis**
- **Limitation:** Does not contain a city or municipal boundary identifier. Registrations represent vehicle ownership licenses, not active on-road traffic. Excluded from Mumbai KPI metrics.

### Dataset E — METR-LA / PEMS-BAY Benchmark Files (Non-Mumbai)
- **File Names:** `PEMS-BAY.h5`, `adj_mx.pkl`, `adj_mx_bay.pkl`
- **Origin:** California Department of Transportation (Caltrans PeMS)
- **Status:** **Strictly Excluded**. Completely excluded from the UI, calculations, and APIs.
