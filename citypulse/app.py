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

# Sidebar Brand Header using clean native Streamlit components
with st.sidebar:
    st.title(f"🚦 {APP_NAME}")
    st.caption(f"{APP_TAGLINE}")
    st.success("● Production ML Active", icon="✅")
    
    with st.container(border=True):
        st.caption("MONITORED HIGHWAY DOMAIN")
        st.markdown(f"**{GEOGRAPHIC_SCOPE}**")
        st.caption(f"{TOTAL_MONITORED_NODES} Stations • {SAMPLING_INTERVAL} Cadence")

    if st.button("🔄 Clear Cache & Reload", use_container_width=True, help="Force reload and re-parse all local dataset files"):
        st.cache_data.clear()
        st.rerun()

    st.caption("HackAMatch 1.0 Challenge • Data Science + AI Platform")

# Run navigation
pg = st.navigation(pages)
pg.run()
