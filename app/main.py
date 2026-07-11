import streamlit as st
import pandas as pd
import plotly.graph_objects as ob
import os
from pathlib import Path

# 1. Page Configuration
st.set_page_config(page_title="EEP & Data Center Marriage", layout="wide", page_icon="⚡")

st.title("⚡ Structural Turnaround Dashboard: EEP & Data Center Synergy")
st.markdown("""
This platform maps the macroeconomic transformation of **Ethiopian Electric Power (EEP)**. 
By hosting data centers as continuous anchor consumers, EEP flipped structural losses into record profits.
""")

# 2. Safe Data Loading
current_dir = Path(os.getcwd())
raw_data_path = current_dir / "data" / "raw" / "eep_baseline.csv"

if raw_data_path.exists():
    df = pd.read_csv(raw_data_path)
else:
    st.error("❌ Data file missing. Please check your data directory.")
    st.stop()

# 3. Sidebar Simulation Slider
st.sidebar.header("🎛️ Simulation Parameters")
expansion_slider = st.sidebar.slider(
    "Simulated Future Tech Load Growth Multiplier", 
    min_value=1.0, 
    max_value=2.5, 
    value=1.0, 
    step=0.1,
    help="Simulates scaling up permitted mega-watt capacity deployments across EEP facilities."
)

# Apply dynamic simulation changes to a copy of the dataframe
df_simulated = df.copy()
if expansion_slider > 1.0:
    last_idx = df_simulated.index[-1]
    df_simulated.loc[last_idx, 'Data_Mining_Consumption_GWh'] *= expansion_slider
    df_simulated.loc[last_idx, 'Net_Income_Loss_Billion_Birr'] *= (1.0 + (expansion_slider - 1.0) * 0.4)
    df_simulated.loc[last_idx, 'Total_Revenue_Billion_Birr'] *= (1.0 + (expansion_slider - 1.0) * 0.3)

# 4. Key Performance Metric Cards
st.subheader("📌 Key Performance Indicators (Latest Fiscal Year)")
m1, m2, m3 = st.columns(3)
latest_year = df_simulated.iloc[-1]

with m1:
    st.metric(
        label="Projected Total Revenue", 
        value=f"{latest_year['Total_Revenue_Billion_Birr']:.1f} Billion ETB"
    )
with m2:
    st.metric(
        label="Net Financial Position", 
        value=f"{latest_year['Net_Income_Loss_Billion_Birr']:.1f} Billion ETB",
        delta="Profitable Turnaround" if latest_year['Net_Income_Loss_Billion_Birr'] > 0 else "Net Loss"
    )
with m3:
    st.metric(
        label="Tech Grid Share (Simulated)", 
        value=f"{df.iloc[-1]['Data_Mining_Grid_Share_Pct'] * expansion_slider:.1f}%"
    )

st.markdown("---")

# 5. Clean, Robust Data Visualization
st.subheader("📊 Financial Trajectory: Net Income/Loss Turnaround")

fig = ob.Figure()

# Add EEP Net Income/Loss as a bar chart
fig.add_trace(
    ob.Bar(
        x=df_simulated['Fiscal_Year'], 
        y=df_simulated['Net_Income_Loss_Billion_Birr'], 
        name="Net Income/Loss (Billion ETB)",
        marker_color=['#e74c3c' if y < 0 else '#2ecc71' for y in df_simulated['Net_Income_Loss_Billion_Birr']],
        text=df_simulated['Net_Income_Loss_Billion_Birr'].round(1),
        textposition='auto'
    )
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="Fiscal Year",
    yaxis_title="Net Profit / Loss (Billion Birr)",
    margin=dict(l=40, r=40, t=20, b=40),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# 6. Executive Narrative
st.info(
    "💡 **Stakeholder Presentation Takeaway:** Adjusting the growth slider on the left demonstrates how increasing "
    "permitted data center infrastructure directly expands EEP's net margins, providing the state utility with the "
    "exact capital required to fund broad domestic grid repairs."
)