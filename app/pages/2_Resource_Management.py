import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Resource Management", page_icon="🏥", layout="wide")

st.title("🏥 Hospital Resource Allocation Command Center")
st.markdown("##### Real-time tracking of critical care assets, patient admissions, and tactical staff distribution networks.")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Dashboard Controls")
facility_view = st.sidebar.selectbox(
    "Select Facility View", 
    ["All Departments", "Intensive Care Unit (ICU)", "Emergency Department", "General Medicine"]
)

# Mock Data Generation representing a typical busy metropolitan hospital
# In production, this dictionary would read from your 'resource_management/' folder files
raw_resource_data = {
    "Department": ["Intensive Care Unit (ICU)", "Emergency Department", "General Medicine", "Radiology Clinic", "Pulmonology Core"],
    "Total Beds": [50, 45, 250, 10, 40],
    "Occupied Beds": [42, 38, 195, 4, 35],
    "Ventilators Total": [25, 10, 0, 0, 15],
    "Ventilators Active": [18, 6, 0, 0, 11],
    "Doctors Available": [14, 22, 35, 8, 12],
    "Nurses Available": [45, 60, 110, 15, 38],
    "Today Admissions": [8, 42, 18, 12, 9],
    "Emergency Cases": [5, 28, 0, 0, 4]
}

df_master = pd.DataFrame(raw_resource_data)

# Filter the data frame based on user selection in the sidebar
if facility_view != "All Departments":
    df_filtered = df_master[df_master["Department"] == facility_view]
else:
    df_filtered = df_master

# --- SECTION 1: CORE OPERATIONAL METRICS ---
# Aggregating values to show in the high-impact metric cards
total_patients = int(df_filtered["Occupied Beds"].sum())
today_admissions = int(df_filtered["Today Admissions"].sum())
emergency_cases = int(df_filtered["Emergency Cases"].sum())

st.subheader("📊 Active Patient Load Telemetry")
metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(label="Total Admitted Patients", value=total_patients, delta=f"+{today_admissions} In Last 24h")
metric_col2.metric(label="Today's Admissions", value=today_admissions, delta="Stable Intake")
metric_col3.metric(label="Active Emergency Cases", value=emergency_cases, delta="+4 Awaiting Triage", delta_color="inverse")

st.markdown("---")

# --- SECTION 2: CRITICAL CARE ASSET CAPACITY ---
st.subheader("💨 Critical Care Capacity Tracker (Beds & Ventilators)")
asset_col1, asset_col2, asset_col3 = st.columns(3)

total_beds = int(df_filtered["Total Beds"].sum())
occupied_beds = int(df_filtered["Occupied Beds"].sum())
available_beds = total_beds - occupied_beds

total_vents = int(df_filtered["Ventilators Total"].sum())
active_vents = int(df_filtered["Ventilators Active"].sum())
available_vents = total_vents - active_vents

# 1. Bed Allocation Readout
asset_col1.metric(
    label="Available Beds / Total Beds", 
    value=f"{available_beds} / {total_beds}", 
    delta=f"{(available_beds/total_beds)*100:.1f}% Capacity Free"
)

# 2. Ventilator Allocation Readout
if total_vents > 0:
    asset_col2.metric(
        label="Available / Total Ventilators", 
        value=f"{available_vents} / {total_vents}", 
        delta=f"{active_vents} Currently In-Use"
    )
else:
    asset_col2.metric(label="Available / Total Ventilators", value="0 / 0", delta="N/A for Current View")

# 3. Live Staffing Ratio
total_docs = int(df_filtered["Doctors Available"].sum())
total_nurses = int(df_filtered["Nurses Available"].sum())
asset_col3.metric(label="On-Duty Clinical Staff", value=f"{total_docs + total_nurses} Active", delta=f"{total_docs} MDs | {total_nurses} RNs")

# --- SECTION 3: INTERACTIVE CHARTS & ALLOCATION GRIDS ---
st.markdown("---")
layout_col_left, layout_col_right = st.columns([1.3, 1], gap="large")

with layout_col_left:
    st.subheader("📋 Tactical Departmental Capacity Log")
    st.markdown("Detailed resource utilization breakdowns across the selected facility filters:")
    
    # Calculate utilization percentage for the progress bars
    df_grid = df_master.copy()
    df_grid["Bed Occupancy Rate (%)"] = ((df_grid["Occupied Beds"] / df_grid["Total Beds"]) * 100).round(1)
    
    # Render an interactive clean grid with built-in visual metrics bars
    st.data_editor(
        df_grid[["Department", "Total Beds", "Occupied Beds", "Doctors Available", "Nurses Available", "Bed Occupancy Rate (%)"]],
        column_config={
            "Bed Occupancy Rate (%)": st.column_config.ProgressColumn(
                "Bed Occupancy Rate (%)",
                format="%d%%",
                min_value=0,
                max_value=100
            )
        },
        disabled=True,
        use_container_width=True,
        hide_index=True
    )

with layout_col_right:
    st.subheader("📈 Bed Load Distribution Analysis")
    
    # Generate a matching clean donut chart for visual telemetry
    fig_pie = px.pie(
        names=["Occupied Beds", "Available General Beds"],
        values=[occupied_beds, available_beds],
        color_discrete_sequence=["#0052CC", "#E2E8F0"],
        hole=0.5
    )
    fig_pie.update_layout(
        margin=dict(l=20, r=20, t=10, b=10), 
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_pie, use_container_width=True)