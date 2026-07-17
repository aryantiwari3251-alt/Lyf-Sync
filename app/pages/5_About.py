import streamlit as st

st.set_page_config(page_title="About LifeSync", layout="wide")

st.title(" Platform Vision & Specifications")
st.markdown("##### Balancing advanced AI medical diagnostics with absolute patient data privacy.")
st.markdown("---")

# --- SECTION 1: THE CORE MISSION ---
st.subheader(" The Big Picture")
st.write(
    "LyfSync was built to solve a massive real-world bottleneck in healthcare technology: "
    "how can medical institutions use powerful Artificial Intelligence to diagnose diseases without "
    "risking patient privacy laws or leaking sensitive records? "
    "By bridging clinical AI diagnostics with a decentralized networking framework, LyfSync allows "
    "hospitals to collectively train and improve a global AI model while ensuring 100% of their "
    "patient images remain safely locked behind their local firewalls."
)

st.markdown("---")

# --- SECTION 2: THE CRUCIAL TECH PILLARS ---
st.subheader(" Core Functional Pillars")
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("#### 🩺 AI Diagnostics Core")
    st.write(
        "Powered by a customized **ResNet18** deep learning architecture. The system processes "
        "chest X-rays in seconds to screen for 6 major conditions: **No Finding, Atelectasis, "
        "Cardiomegaly, Effusion, Pneumothorax, and Pneumonia**, giving doctors an instant, reliable second opinion."
    )

with col2:
    st.markdown("#### 🔒 Privacy-Preserving Sync")
    st.write(
        "Instead of collecting private patient pixels into a vulnerable central cloud database, the platform "
        "uses **Federated Learning**. The AI learns from data locally at each hospital, broadcasting "
        "only encrypted mathematical model parameters to keep system updates completely anonymous."
    )

with col3:
    st.markdown("#### 🏥 Logistical Balancing")
    st.write(
        "An integrated hospital resource command center designed to balance clinical demands with physical "
        "capacity. It dynamically maps patient flows against active ICU bed limits, ventilator counts, "
        "and on-duty medical staff to eliminate emergency room bottlenecks."
    )

st.markdown("---")

# --- SECTION 3: FUTURE SCOPE ---
st.subheader(" Scalability Roadmap")
st.write(
    "To take this software from a working demonstration to an enterprise-grade hospital network, the future roadmap "
    "includes adding advanced mathematical noise parameters (Differential Privacy) to completely neutralize reverse-engineering "
    "attacks, expanding the vision framework to read high-resolution CT and MRI scans, and introducing automated inventory "
    "routing to balance patient overflows between connected city facilities."
)

# --- SECTION 4: UNHIGHLIGHTED CONTRIBUTORS FOOTER ---
st.markdown("---")
st.write("Developed by: Aryan Tiwari, Yash Daksh, Rohit Joshi")