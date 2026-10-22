import streamlit as st
import pandas as pd

st.markdown("""
<div class="citypulse-topbar">
    <div>
        <h1 style="margin: 0; font-size: 1.5rem; font-weight: 800; color: #FFFFFF;">
            Methodology, Data Provenance & Ethics Statement
        </h1>
        <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #94A3B8;">
            Formal documentation of extraction protocols, geographic boundaries, and ethical constraints governing <strong style="color: #F1F5F9;">CityPulse Phase 1</strong>
        </p>
    </div>
    <div style="background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.4); border-radius: 8px; padding: 6px 12px; font-size: 0.75rem; color: #22D3EE; font-weight: 700;">
        Open & Reproducible
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Strict Mumbai-Only Geographic Integrity")
    st.markdown("""
    A foundational tenet of CityPulse is uncompromising geographic honesty. All dashboards represent conditions in **Mumbai City only**:
    - **Primary Challans:** Covers 42 Mumbai traffic divisions and 91 local police stations.
    - **Maharashtra Traffic CSV:** Filtered to 234 Mumbai records. 966 records from Pune, Thane, Nashik, and Nagpur were strictly removed.
    - **IMD Rainfall:** Filtered strictly to `District == 'MUMBAI CITY'`.
    - **No False Substitution:** If Mumbai data is absent for any metric, an empty state is shown rather than substituting another city's records.
    """)

    st.subheader("2. Table Extraction & Reconciliation")
    st.markdown("""
    - **Vector Extraction:** Utilized PyMuPDF vector line parsing on `141d484b-bbb2-4b93-8797-6e88533fd506.pdf` to extract exact numbers without OCR artifacts.
    - **Single-Rupee Reconciliation:** The sum of all 42 division rows (5,937,151 challans, ₹4,784,033,250) matches the printed Grand Total row down to the single rupee.
    - **Separation Safeguard:** Traffic Division tallies and Police Station tallies are kept in distinct tables to eliminate double counting.
    """)

with col2:
    st.subheader("3. Deliberate METR-LA / PEMS-BAY Exclusion")
    st.markdown("""
    The provided archive contains `PEMS-BAY.h5` and sensor adjacency matrices (`adj_mx.pkl`).
    
    **Challenge Compliance Decision:**
    These files represent Los Angeles and California highway loop detectors from Caltrans PeMS. 
    They have been completely excluded from Mumbai traffic metrics, charts, and calculations. 
    Presenting California highway sensor measurements as Mumbai traffic would falsify domain reality.
    """)

    st.subheader("4. Evidence-Based Insights (No Fake ML)")
    st.markdown("""
    - **Correlation != Causation:** District rainfall observations and traffic challan numbers are kept separate. No claim is made that rainfall directly caused recorded infractions.
    - **Deterministic Explanations:** All insights use structured templates (*Observed*, *Interpretation*, *Can Infer*, *Cannot Infer*, *Next Step*) computed directly from validated numbers.
    - **Zero Hallucination:** No external LLM APIs or black-box generators are used to fabricate safety scores.
    """)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Data Provenance Matrix Table
st.subheader("📋 Data Provenance Matrix")
matrix_data = [
    {
        "Dataset": "Mumbai Traffic Violations 2024",
        "Primary Source": "Mumbai Traffic Police Annual Report (PDF)",
        "Geographic Scope": "Mumbai City (42 Divs + 91 PS)",
        "Verification Tier": "Official Government (Verified)",
        "Handling in CityPulse": "Primary empirical source for Mumbai Insights & Overview"
    },
    {
        "Dataset": "Maharashtra Traffic CSV",
        "Primary Source": "Synthetic / Exploratory CSV archive",
        "Geographic Scope": "Filtered to 234 Mumbai rows",
        "Verification Tier": "Unverified Prototype",
        "Handling in CityPulse": "Isolated in Traffic Data Explorer; excluded from official statistics"
    },
    {
        "Dataset": "IMD Daily Rainfall CSV",
        "Primary Source": "India Meteorological Department (IMD)",
        "Geographic Scope": "Mumbai City District (65 Days)",
        "Verification Tier": "Official Meteorological (Verified)",
        "Handling in CityPulse": "Environmental baseline displayed independently on Rainfall page"
    },
    {
        "Dataset": "Vehicle Registration CSV",
        "Primary Source": "Administrative registry file",
        "Geographic Scope": "State-wide / Unspecified Scope",
        "Verification Tier": "Unsuitable (Scope)",
        "Handling in CityPulse": "Excluded from Mumbai metrics; flagged with warning on Vehicle Trends"
    },
    {
        "Dataset": "METR-LA / PEMS-BAY Files",
        "Primary Source": "Caltrans PeMS Highway Network (California, USA)",
        "Geographic Scope": "Los Angeles / Bay Area (US)",
        "Verification Tier": "Strictly Excluded",
        "Handling in CityPulse": "Zero integration into Mumbai dashboards; documented for audit"
    }
]

st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)
