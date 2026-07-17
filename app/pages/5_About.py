import streamlit as st
import pandas as pd

st.set_page_config(page_title="Specifications & Contact", layout="wide")

st.title(" System Specifications & Contact Portal")
st.markdown("##### Technical infrastructure metrics and engineering primary points of contact.")
st.markdown("---")

# --- SECTION 1: SYSTEM SPECIFICATIONS BRIEF ---
st.subheader("📊 Architecture & Deployment Framework")
st.markdown("A brief overview of the operational parameters and core framework modules executing within the network environment:")

# Clean, scannable specifications matrix
specs_data = {
    "Operational Layer": [
        "Analytical Core", 
        "Data Encryption & Sync", 
        "Logistics Management"
    ],
    "Technical Specification": [
        "ResNet18 Deep Residual Network Architecture", 
        "Decentralized Parameter Orchestration (FedAvg Framework)", 
        "Dynamic Capacity Allocation & Telemetry Matrix"
    ],
    "Functional Scope": [
        "Multi-class diagnostic probability vector generation across 6 targeted pathologies.",
        "Secure global update loops utilizing abstracted one-way weight tensors.",
        "Real-time tracking profiles for physical critical care inventory and personnel loading."
    ]
}

df_specs = pd.DataFrame(specs_data)
st.table(df_specs)

st.markdown("---")

# --- SECTION 2: DEVELOPER DIRECTORY / CONTACT US ---
st.subheader(" Contact Us")
st.markdown("For any architectural inquiries, system administration support, or technical documentation queries, please reach out to the development team:")

# Unified contact grid layout
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("### **Aryan Tiwari**")
    st.markdown("🏫 **Delhi Technological University**")
    st.markdown("📧 **Email:** [aryantiwari3251@gmail.com](mailto:aryantiwari3251@gmail.com)")

with col2:
    st.markdown("### **Yash Daksh**")
    st.markdown("🏫 **Delhi Technological University**")
    st.markdown("📧 **Email:** [yashdaksh_mce_25_b06061@dtu.ac.in](mailto:yashdaksh_mce_25_b06061@dtu.ac.in)")
with col3:
    st.markdown("### **Rohit Joshi**")
    st.markdown("🏫 **Delhi Technological University**")
    st.markdown("📧 **Email:** [rockrjstar04@gmail.com](mailto:rockrjstar04@gmail.com)")
    