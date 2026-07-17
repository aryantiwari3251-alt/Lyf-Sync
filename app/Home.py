import streamlit as st

st.set_page_config(
    page_title="LyfSync Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, corporate sidebar navigation branding
st.sidebar.markdown("# LyfSync ")
st.sidebar.markdown("---")
st.sidebar.caption("Enterprise Clinical Intelligence Platform v2.4")

# --- HERO SECTION ---
st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0rem;">
        <h1 style="color: #0052CC; font-size: 3rem; margin-bottom: 0.5rem;">LyfSync</h1>
        <h3 style="color: #1E293B; font-weight: 400; margin-bottom: 1.5rem;">The Future of Collaborative Clinical Intelligence</h3>
        <p style="font-size: 1.2rem; color: #64748B; max-width: 800px; margin: 0 auto; line-height: 1.6;">
            A unified healthcare SaaS ecosystem integrating deep-learning diagnostic screening, decentralized privacy protocols, and predictive asset management into a single sovereign hospital network.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- THE CORE PARADOX WE SOLVE ---
st.markdown("### The Core Mission: Overcoming the Healthcare AI Bottleneck")
st.write(
    "Modern medicine faces a critical conflict: AI models need massive amounts of patient data to become accurate, "
    "but strict privacy regulations (like HIPAA and GDPR) rightly lock that data down behind institutional firewalls. "
    "**LyfSync breaks this deadlock.** By shifting from centralized data collection to decentralized data orchestration, "
    "we deliver cutting-edge clinical insights while ensuring patient data never leaves its source."
)

st.markdown("---")

# --- THREE VALUE PILLARS ---
st.subheader(" Integrated Ecosystem Solutions")
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("####  1. AI Clinical Diagnostics")
    st.markdown(
        "Empower frontline radiologists with advanced Computer Vision. Our core imaging engine runs "
        "high-fidelity screening protocols across **6 major thoracic pathologies**, offering instant diagnostic acceleration "
        "and triage prioritization where seconds count."
    )

with col2:
    st.markdown("####  2. Privacy-Preserving Federation")
    st.markdown(
        "Advance institutional knowledge without compromising data security. By utilizing a **Federated Learning (FedAvg)** "
        "framework, the network safely trains models across isolated hospital nodes by sharing encrypted mathematical parameters "
        "rather than private patient pixels."
    )

with col3:
    st.markdown("####  3. Asset & Resource Optimization")
    st.markdown(
        "Synchronize clinical insights with physical capacity. LyfSync dynamically aligns incoming analytical diagnostics "
        "with real-time ICU bed metrics, ventilator availability, and staffing allocations to eliminate critical bottlenecks."
    )

st.markdown("---")

# --- CONCEPTUAL WORKFLOW ---
st.subheader("🔄 How LyfSync Operates in the Real World")

wf_col1, wf_col2, wf_col3 = st.columns(3)

with wf_col1:
    st.markdown(
        """
        <div style="background-color: #FFFFFF; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #0052CC; min-height: 180px;">
            <h5 style="color: #0052CC; margin-top:0;">Step 1: Localized Screening</h5>
            <p style="font-size: 0.95rem; color: #1E293B; margin-bottom:0;">
                Hospitals process chest X-rays locally. The clinical interface instantly pinpoints abnormalities to assist on-duty doctors, acting entirely behind the facility's secure firewall.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

with wf_col2:
    st.markdown(
        """
        <div style="background-color: #FFFFFF; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #34D399; min-height: 180px;">
            <h5 style="color: #34D399; margin-top:0;">Step 2: Sovereign Synchronization</h5>
            <p style="font-size: 0.95rem; color: #1E293B; margin-bottom:0;">
                Periodically, local nodes securely broadcast numerical weight adjustments—not patient images—to a secure coordinator server, generating a collective intelligence baseline.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

with wf_col3:
    st.markdown(
        """
        <div style="background-color: #FFFFFF; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #60A5FA; min-height: 180px;">
            <h5 style="color: #60A5FA; margin-top:0;">Step 3: Logistics Balancing</h5>
            <p style="font-size: 0.95rem; color: #1E293B; margin-bottom:0;">
                As localized diagnostics shift, operational dashboards update personnel and emergency resource parameters, keeping hospital management agile and prepared.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.markdown("---")

# --- CALL TO ACTION ---
st.markdown(
    """
    <div style="text-align: center; background-color: #E0EFFF; padding: 1.5rem; border-radius: 8px;">
        <h4 style="color: #0052CC; margin-top: 0;">Explore the Enterprise Suite</h4>
        <p style="color: #1E293B; margin-bottom: 0;">
            Use the <b>Sidebar Navigation Menu</b> on the left to seamlessly transition between the live 
            <b>AI Inference Engine</b>, the <b>Resource Command Center</b>, or the <b>Privacy Compliance Audit Logs</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)