# CityPulse Methodology & Engineering Architecture

## 1. Vector Table Extraction Methodology

CityPulse implements programmatic table extraction from the official Mumbai Police report (`141d484b-bbb2-4b93-8797-6e88533fd506.pdf`) using PyMuPDF (`fitz.find_tables()`):
- Vector bounding boxes are detected dynamically to extract exact cell text without lossy optical character recognition (OCR).
- Extracted values undergo cleaning (removal of thousand separators, newline artifacts, and whitespace).
- Each table is audited against printed Grand Totals:
  $$\sum_{i=1}^{42} \text{Challans}_{\text{Division } i} = 5,937,151$$
  $$\sum_{j=1}^{91} \text{Challans}_{\text{Police Station } j} = 847,766$$
  $$\sum_{k=1}^{24} \text{Challans}_{\text{Category } k} = 5,937,151$$
- Verified schemas are serialized to `cache/challans_cache.json` for sub-50ms API response times.

---

## 2. Geographic Filtering Protocols

To maintain geographic integrity:
- **Filtering Algorithm:** For any tabular dataset with geographic columns (e.g. `City`, `State`, `District`), rows are filtered using strict case-insensitive equality against verified Mumbai City tokens.
- **Leakage Prevention:** In `Traffic dataset.csv`, all 966 records from non-Mumbai cities (Pune, Thane, Nashik, Nagpur) are excluded before computing any distribution, summary statistic, or chart data.
- **Reporting Unit Transparency:** For rainfall data, the reporting unit is explicitly stated as *Mumbai City District*.

---

## 3. Separation of Enforcement, Traffic, and Weather

CityPulse strictly prevents cross-dataset contamination:
1. **Enforcement vs Congestion:** Recorded traffic challans reflect policing and camera density, not physical road congestion or accident risk.
2. **Registrations vs Traffic:** Vehicle registration figures reflect fleet stock, not real-time road circulation volume.
3. **Decoupled Weather:** District-level rainfall is not used as a causal explanatory variable for traffic events in Phase 1.

---

## 4. Deterministic Descriptive Insights Engine

CityPulse computes explainable descriptive insights without machine learning or large language models:
- Observations are calculated directly from verified sums and percentages.
- Inferences and guardrails are structured using standardized templates:
  - **Observed:** Factual statement computed from validated records.
  - **Interpretation:** Cautious contextual explanation.
  - **Can Infer:** Defensible domain conclusions.
  - **Cannot Infer:** Explicit warnings preventing overinterpretation.
  - **Suggested Next Step:** Follow-up data collection recommendation.
