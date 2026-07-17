import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"

st.subheader("📈 Training Data Distributions")

col1, col2 = st.columns(2)

# Load your actual pre-saved backend images seamlessly
if (ANALYSIS_DIR / "disease_distribution.png").exists():
    col1.image(str(ANALYSIS_DIR / "disease_distribution.png"), caption="Overall Disease Distribution", use_container_width=True)

if (ANALYSIS_DIR / "multilabel_distribution.png").exists():
    col2.image(str(ANALYSIS_DIR / "multilabel_distribution.png"), caption="Co-occurrence / Multilabel Spread", use_container_width=True)