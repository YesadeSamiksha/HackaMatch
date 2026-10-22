import streamlit as st
import sys
from pathlib import Path

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import APP_NAME, APP_TAGLINE, GEOGRAPHIC_SCOPE, TOTAL_MONITORED_NODES, SAMPLING_INTERVAL

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Global CSS for dark navy analytics aesthetic
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background-color: #0B0F19;
        color: #F1F5F9;
    }
    
    /* Stat cards */
    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stMetricLabel"] {
        color: #94A3B8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-family: monospace;
        font-weight: 700;
        font-size: 1.6rem;
    }
</style>
""", unsafe_allow_html=True)

def get_page_path(rel_path: str) -> str:
    p = BASE_DIR / rel_path
    return str(p) if p.exists() else rel_path

# Define multipage navigation using st.navigation
pages = [
    st.Page(get_page_path("pages/1_Overview.py"), title="Overview", icon="📊", url_path="Overview", default=True),
    st.Page(get_page_path("pages/2_Hotspot_Intelligence.py"), title="Hotspot Intelligence", icon="🔍", url_path="Hotspot_Intelligence"),
    st.Page(get_page_path("pages/3_Prediction_Lab.py"), title="Prediction Lab", icon="🧪", url_path="Prediction_Lab"),
    st.Page(get_page_path("pages/4_Action_Center.py"), title="Action Center", icon="🎯", url_path="Action_Center"),
    st.Page(get_page_path("pages/5_Model_Evaluation.py"), title="Model Evaluation", icon="📈", url_path="Model_Evaluation"),
    st.Page(get_page_path("pages/6_Data_Quality.py"), title="Data Quality Hub", icon="📋", url_path="Data_Quality")
]

# Sidebar Brand Header
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1E293B; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="background: linear-gradient(135deg, #06B6D4, #0891B2); padding: 6px; border-radius: 8px;">
                <span style="font-size: 1.2rem;">🚦</span>
            </div>
            <div>
                <h2 style="margin: 0; font-size: 1.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">{APP_NAME}</h2>
                <span style="font-size: 0.65rem; background: rgba(16, 185, 129, 0.2); color: #34D399; padding: 2px 6px; border-radius: 4px; font-weight: 600;">Production ML Active</span>
            </div>
        </div>
        <p style="margin: 8px 0 0 0; font-size: 0.72rem; color: #94A3B8; line-height: 1.3;">
            {APP_TAGLINE}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 10px; margin-bottom: 16px;">
        <div style="font-size: 0.7rem; font-weight: 700; color: #38BDF8; text-transform: uppercase;">Monitored Highway Domain</div>
        <div style="font-size: 0.8rem; font-weight: 600; color: #F1F5F9; margin-top: 2px;">{GEOGRAPHIC_SCOPE}</div>
        <div style="font-size: 0.68rem; color: #94A3B8; margin-top: 4px;">{TOTAL_MONITORED_NODES} Stations • {SAMPLING_INTERVAL} Cadence</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Clear Cache & Reload", use_container_width=True, help="Force reload and re-parse all local dataset files"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style="font-size: 0.65rem; color: #64748B; text-align: center; margin-top: 20px;">
        HackAMatch 1.0 Challenge<br>Data Science + AI Platform
    </div>
    """, unsafe_allow_html=True)

# Run navigation
pg = st.navigation(pages)
pg.run()
