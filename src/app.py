"""
app.py — Streamlit UI for the Household Energy Agent Pipeline
==============================================================
Replaces terminal input() agents (1-5) with a Streamlit sidebar.
Runs computation agents (6-11) through the existing LangGraph workflow.

Run:  cd src && streamlit run app.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, START, END
from state import HouseholdProfileState
from agents.gross_energy_calculation_agent import GrossEnergyCalculationAgent
from agents.apply_insulation_adjustments import ApplyInsulationAdjustmentsAgent
from agents.calculate_grid_draw_and_expense_agent import CalculateGridDrawAndExpenseAgent
from agents.compare_against_similar_households_agent import CompareAgainstSimilarHouseholdsAgent
from agents.solar_roi_analysis_agent import SolarROIAnalysisAgent
from agents.energy_recommendations_agent import EnergyRecommendationsAgent

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(page_title="⚡ Energy Advisor", page_icon="⚡", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.2rem;}</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# COMPUTATION-ONLY WORKFLOW (Agents 6–11)
# Skips terminal-input agents 1–5
# ─────────────────────────────────────────────

@st.cache_resource
def init_agents():
    return {
        "gross": GrossEnergyCalculationAgent(),
        "insulation": ApplyInsulationAdjustmentsAgent(),
        "grid": CalculateGridDrawAndExpenseAgent(),
        "compare": CompareAgainstSimilarHouseholdsAgent(),
        "solar": SolarROIAnalysisAgent(),
        "recs": EnergyRecommendationsAgent(),
    }


def build_compute_workflow():
    """Build LangGraph workflow with only computation agents (6-11)."""
    agents = init_agents()

    def node_gross(state):
        return agents["gross"].process(state)

    def node_insulation(state):
        return agents["insulation"].process(state)

    def node_grid(state):
        return agents["grid"].process(state)

    def node_compare(state):
        return agents["compare"].process(state)

    def node_solar(state):
        return agents["solar"].process(state)

    def node_recs(state):
        return agents["recs"].process(state)

    workflow = StateGraph(HouseholdProfileState)

    workflow.add_node("gross_energy_calculation", node_gross)
    workflow.add_node("apply_insulation_adjustments", node_insulation)
    workflow.add_node("calculate_grid_draw_and_expense", node_grid)
    workflow.add_node("compare_against_similar_households", node_compare)
    workflow.add_node("solar_roi_analysis", node_solar)
    workflow.add_node("energy_recommendations", node_recs)

    workflow.add_edge(START, "gross_energy_calculation")
    workflow.add_edge("gross_energy_calculation", "apply_insulation_adjustments")
    workflow.add_edge("apply_insulation_adjustments", "calculate_grid_draw_and_expense")
    workflow.add_edge("calculate_grid_draw_and_expense", "compare_against_similar_households")
    workflow.add_edge("compare_against_similar_households", "solar_roi_analysis")
    workflow.add_edge("solar_roi_analysis", "energy_recommendations")
    workflow.add_edge("energy_recommendations", END)

    return workflow.compile()


# ─────────────────────────────────────────────
# SIDEBAR — Replaces Agents 1-5
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("🏠 Household Profile")

    # Agent 1: Collect Household Profile
    st.subheader("1. Home Details")
    house_type = st.selectbox("House Type", ["Apartment", "Villa", "Bungalow", "Townhouse", "Penthouse"])
    num_bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=1)
    climate_zone = st.selectbox("Climate Zone", ["Hot & Dry", "Hot & Humid", "Composite", "Temperate", "Cold"])
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    floor_area = st.number_input("Floor Area (sq ft)", min_value=100, max_value=10000, value=1000, step=50)

    st.divider()

    # Agent 2: Occupancy Details
    st.subheader("2. Occupancy")
    num_adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
    num_children = st.number_input("Children", min_value=0, max_value=10, value=1)

    st.divider()

    # Agent 3: Appliance Inventory
    st.subheader("3. Appliances")

    has_ac = st.checkbox("Air Conditioner", value=True)
    num_ac_units, ac_star, ac_hrs = 0, 3, 0.0
    if has_ac:
        c1, c2, c3 = st.columns(3)
        num_ac_units = c1.number_input("AC Units", 1, 10, 2)
        ac_star = c2.selectbox("AC Star", [1, 2, 3, 4, 5], index=4)
        ac_hrs = c3.number_input("AC Hrs/Day", 0.0, 24.0, 10.0, step=0.5)

    num_fans = st.number_input("Ceiling Fans", 0, 20, 4)

    water_heater_type = st.selectbox("Water Heater", ["None", "Electric Geyser", "Heat Pump", "Solar + Backup"], index=3)
    wh_capacity, wh_hrs = 0, 0.0
    if water_heater_type != "None":
        c1, c2 = st.columns(2)
        wh_capacity = c1.number_input("Heater Capacity (L)", 0, 200, 15)
        wh_hrs = c2.number_input("Heater Hrs/Day", 0.0, 12.0, 4.0, step=0.5)

    has_fridge = st.checkbox("Refrigerator", value=True)
    fridge_capacity, fridge_star = 0, 3
    if has_fridge:
        c1, c2 = st.columns(2)
        fridge_capacity = c1.number_input("Fridge Capacity (L)", 50, 800, 330)
        fridge_star = c2.selectbox("Fridge Star", [1, 2, 3, 4, 5], index=3)

    has_wm = st.checkbox("Washing Machine", value=True)
    wm_type, wm_cycles = "Top Load", 0
    if has_wm:
        c1, c2 = st.columns(2)
        wm_type = c1.selectbox("Washer Type", ["Top Load", "Front Load", "Semi-Automatic"])
        wm_cycles = c2.number_input("Cycles/Week", 0, 21, 9)

    c1, c2 = st.columns(2)
    num_computers = c1.number_input("Computers", 0, 10, 2)
    comp_hrs = c2.number_input("PC Hrs/Day", 0.0, 24.0, 15.0, step=0.5) if num_computers > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    num_tvs = c1.number_input("TVs", 0, 10, 1)
    tv_size = c2.number_input("TV Inch", 20, 85, 55) if num_tvs > 0 else 0
    tv_hrs = c3.number_input("TV Hrs", 0.0, 24.0, 5.0, step=0.5) if num_tvs > 0 else 0.0

    c1, c2 = st.columns(2)
    has_dishwasher = c1.checkbox("Dishwasher", value=True)
    has_microwave = c2.checkbox("Microwave", value=True)
    dw_cycles = st.number_input("Dishwasher Cycles/Week", 0, 21, 9) if has_dishwasher else 0

    st.divider()

    # Agent 4: Building Envelope
    st.subheader("4. Building Envelope")
    insulation = st.selectbox("Insulation Quality", ["Excellent", "Good", "Average", "Poor"], index=2)
    window_type = st.selectbox("Window Type", ["Single Pane", "Double Pane", "Triple Pane"], index=1)
    roof_type = st.selectbox("Roof Type", ["Flat RCC", "Insulated RCC", "Sloped Tiled", "Green Roof"])

    st.divider()

    # Agent 5: Renewable Energy
    st.subheader("5. Renewable Energy")
    has_solar = st.checkbox("Solar Panels", value=True)
    solar_kwp = st.number_input("Solar kWp", 0.0, 20.0, 2.0, step=0.5) if has_solar else 0.0
    has_battery = st.checkbox("Battery Storage")
    battery_kwh = st.number_input("Battery kWh", 0.0, 50.0, 0.0, step=1.0) if has_battery else 0.0

    st.divider()
    run_btn = st.button("⚡ Run Analysis", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# BUILD PROFILE DATA (matches existing state shape)
# ─────────────────────────────────────────────

profile_data = {
    "house_type": house_type,
    "num_bedrooms": num_bedrooms,
    "climate_zone": climate_zone,
    "city_tier": city_tier,
    "floor_area_sqft": floor_area,
    "num_adults": num_adults,
    "num_children": num_children,
    "num_occupants": num_adults + num_children,
    "has_ac": 1 if has_ac else 0,
    "num_ac_units": num_ac_units if has_ac else 0,
    "ac_start_rating": ac_star if has_ac else 0,
    "ac_usage_hrs_per_day": ac_hrs if has_ac else 0.0,
    "num_ceiling_fans": num_fans,
    "water_heater_type": water_heater_type if water_heater_type != "None" else None,
    "water_heater_capacity_L": wh_capacity,
    "water_heater_usage_hrs_per_day": wh_hrs,
    "has_refrigerator": 1 if has_fridge else 0,
    "fridge_capacity_L": fridge_capacity if has_fridge else 0,
    "fridge_star_rating": fridge_star if has_fridge else 0,
    "has_washing_machine": 1 if has_wm else 0,
    "washing_machine_type": wm_type if has_wm else None,
    "washing_cycles_per_week": wm_cycles if has_wm else 0,
    "num_computers": num_computers,
    "computer_usage_hrs_per_day": comp_hrs,
    "num_tvs": num_tvs,
    "tv_screen_size_inch": tv_size,
    "tv_usage_hrs_per_day": tv_hrs,
    "has_dishwasher": 1 if has_dishwasher else 0,
    "dishwasher_cycles_per_week": dw_cycles,
    "has_microwave": 1 if has_microwave else 0,
    "has_solar_panels": 1 if has_solar else 0,
    "solar_capacity_kWp": solar_kwp,
    "has_battery_storage": 1 if has_battery else 0,
    "battery_capacity_kWh": battery_kwh,
    "has_dryer": 0,
    "num_floors": 1,
    "insulation_quality": insulation,
    "window_type": window_type,
    "roof_type": roof_type,
}


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.title("⚡ Household Energy Advisor")
st.caption("LangGraph Agent Pipeline — fill the sidebar, hit Run, see results")

if not run_btn and "final_state" not in st.session_state:
    st.info("👈 Fill in your household profile in the sidebar and click **Run Analysis**.")
    st.stop()


# ─────────────────────────────────────────────
# RUN LANGGRAPH PIPELINE
# ─────────────────────────────────────────────

if run_btn:
    initial_state: HouseholdProfileState = {
        "profile_data": profile_data,
        "messages": [],
        "errors": [],
        "current_agent": "",
        "workflow_stage": "Start",
    }

    graph = build_compute_workflow()

    progress = st.progress(0, text="Starting pipeline...")
    agent_names = [
        "Gross Energy Calculation",
        "Insulation Adjustments",
        "Grid Draw & Expense",
        "Compare Similar Households",
        "Solar ROI Analysis",
        "Energy Recommendations",
    ]

    with st.spinner("Running LangGraph agent pipeline..."):
        final_state = graph.invoke(initial_state)

    progress.progress(100, text="Pipeline complete!")
    st.session_state["final_state"] = final_state

# Get state
state = st.session_state.get("final_state", {})
pd_data = state.get("profile_data", {})

# Show agent messages
msgs = state.get("messages", [])
errors = state.get("errors", [])

if errors:
    for e in errors:
        st.error(f"❌ {e}")

# ─────────────────────────────────────────────
# AGENT PIPELINE LOG
# ─────────────────────────────────────────────

with st.expander("🛠️ Agent Pipeline Log", expanded=False):
    for msg in msgs:
        if "[SUCCESS]" in msg:
            st.success(msg)
        elif "[ERROR]" in msg:
            st.error(msg)
        else:
            st.info(msg)


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────

st.markdown("---")

gross_kwh = pd_data.get("daily_energy_consumption_kWh", 0)
adjusted_kwh = pd_data.get("adjusted_daily_energy_kWh", 0)
solar_gen = pd_data.get("solar_generation_kWh_per_day", 0)
net_grid = pd_data.get("net_grid_draw_kWh_per_day", 0)
monthly_bill = pd_data.get("monthly_grid_bill_kWh", 0)
adj_factors = pd_data.get("energy_adjustment_factors", {})

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Gross Daily", f"{gross_kwh:.1f} kWh")
k2.metric("Adjusted Daily", f"{adjusted_kwh:.1f} kWh",
          f"×{adj_factors.get('combined_factor', 1.0):.2f}")
k3.metric("Solar Gen", f"{solar_gen:.1f} kWh/day")
k4.metric("Net Grid Draw", f"{net_grid:.1f} kWh/day")
k5.metric("Monthly Bill", f"₹{monthly_bill:,.0f}")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Consumption",
    "🔧 Adjustments",
    "📈 Benchmark",
    "☀️ Solar ROI",
    "💡 Recommendations",
])


# TAB 1: Consumption Breakdown
with tab1:
    st.subheader("Appliance-Level Energy Breakdown")

    appliance_data = pd_data.get("appliance_energy_kWh", {})
    if appliance_data:
        names = [k.replace("_kWh_per_day", "").replace("_", " ").title() for k in appliance_data.keys()]
        values = list(appliance_data.values())

        c1, c2 = st.columns(2)

        with c1:
            fig = px.pie(
                names=names, values=values,
                title="Daily Consumption Share", hole=0.4,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(
                x=values, y=names, orientation="h",
                title="Daily Consumption (kWh)",
                labels={"x": "kWh/day", "y": "Appliance"},
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        st.metric("Total Gross Daily Consumption", f"{gross_kwh:.2f} kWh")

    # Assumptions
    assumptions = pd_data.get("energy_assumptions", [])
    if assumptions:
        with st.expander("📋 Calculation Assumptions"):
            for a in assumptions:
                st.markdown(f"- {a}")


# TAB 2: Adjustments
with tab2:
    st.subheader("Climate & Insulation Adjustments")

    if adj_factors:
        c1, c2, c3 = st.columns(3)
        c1.metric("Climate Factor", f"{adj_factors.get('climate_factor', 1.0):.2f}",
                   pd_data.get("energy_adjustment_climate_zone", ""))
        c2.metric("Insulation Factor", f"{adj_factors.get('insulation_factor', 1.0):.2f}",
                   pd_data.get("energy_adjustment_insulation_quality", ""))
        c3.metric("Combined Factor", f"{adj_factors.get('combined_factor', 1.0):.2f}")

        fig = go.Figure(go.Waterfall(
            x=["Gross", "Climate Adj", "Insulation Adj", "Adjusted"],
            y=[gross_kwh,
               gross_kwh * (adj_factors.get("climate_factor", 1) - 1),
               gross_kwh * adj_factors.get("climate_factor", 1) * (adj_factors.get("insulation_factor", 1) - 1),
               0],
            measure=["absolute", "relative", "relative", "total"],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(title="Energy Adjustment Waterfall")
        st.plotly_chart(fig, use_container_width=True)

    # Grid draw
    st.subheader("Net Grid Draw")
    c1, c2, c3 = st.columns(3)
    c1.metric("Adjusted Daily", f"{adjusted_kwh:.1f} kWh")
    c2.metric("Solar Generation", f"-{solar_gen:.1f} kWh")
    c3.metric("Net Grid Draw", f"{net_grid:.1f} kWh")

    grid_assumptions = pd_data.get("grid_draw_assumptions", [])
    if grid_assumptions:
        with st.expander("📋 Grid Draw Assumptions"):
            for a in grid_assumptions:
                st.markdown(f"- {a}")


# TAB 3: Benchmark
with tab3:
    st.subheader("Comparison Against Similar Households")

    comparison = state.get("similar_households_comparison", {})
    if comparison:
        st.info(f"Found **{comparison.get('similar_households_found', 0)}** similar households")

        peer_avg = comparison.get("peer_average_daily_kWh", 0)
        target = comparison.get("target_daily_kWh") or adjusted_kwh

        c1, c2, c3 = st.columns(3)
        c1.metric("Your Daily", f"{target:.1f} kWh")
        c2.metric("Peer Average", f"{peer_avg:.1f} kWh")
        diff = ((target - peer_avg) / peer_avg * 100) if peer_avg > 0 else 0
        c3.metric("Difference", f"{diff:+.1f}%",
                   "Above peers" if diff > 0 else "Below peers")

        # Top matches table
        matches = comparison.get("top_matches", [])
        if matches:
            match_df = pd.DataFrame(matches)
            display_cols = ["house_type", "num_bedrooms", "num_occupants",
                            "climate_zone", "city_tier",
                            "daily_energy_consumption_kWh", "monthly_energy_consumption_kWh"]
            display_cols = [c for c in display_cols if c in match_df.columns]
            st.dataframe(match_df[display_cols], use_container_width=True, hide_index=True)

            # Bar chart comparison
            labels = [f"Match {i+1}" for i in range(len(matches))]
            peer_values = [m["daily_energy_consumption_kWh"] for m in matches]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=peer_values, name="Similar Households", marker_color="steelblue"))
            fig.add_hline(y=target, line_dash="dash", line_color="red",
                          annotation_text=f"Your Home: {target:.1f} kWh")
            fig.add_hline(y=peer_avg, line_dash="dot", line_color="green",
                          annotation_text=f"Peer Avg: {peer_avg:.1f} kWh")
            fig.update_layout(title="Your Consumption vs Similar Households",
                              yaxis_title="Daily kWh")
            st.plotly_chart(fig, use_container_width=True)

        if comparison.get("notes"):
            st.caption(comparison["notes"])

    else:
        st.warning("No benchmark data available. Run the analysis first.")


# TAB 4: Solar ROI
with tab4:
    st.subheader("☀️ Solar ROI Analysis")

    solar_roi = pd_data.get("solar_roi_analysis", {})
    if solar_roi:
        st.markdown(f"**Base Daily Energy:** {solar_roi.get('base_daily_energy_kWh', 0):.1f} kWh  |  "
                     f"**Grid Cost:** ₹{solar_roi.get('grid_cost_per_kWh', 0):.2f}/kWh  |  "
                     f"**Capital Cost:** ₹{solar_roi.get('capital_cost_per_kWp', 0):,.0f}/kWp")

        scenarios = solar_roi.get("scenarios", [])
        if scenarios:
            solar_df = pd.DataFrame(scenarios)
            st.dataframe(solar_df, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)

            with c1:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(
                    x=solar_df["solar_capacity_kWp"].astype(str) + " kWp",
                    y=solar_df["annual_savings"],
                    name="Annual Savings",
                    marker_color="gold",
                ), secondary_y=False)
                if "payback_years" in solar_df.columns:
                    fig.add_trace(go.Scatter(
                        x=solar_df["solar_capacity_kWp"].astype(str) + " kWp",
                        y=solar_df["payback_years"],
                        name="Payback (Years)",
                        marker_color="crimson",
                        mode="lines+markers",
                    ), secondary_y=True)
                fig.update_layout(title="Annual Savings vs Payback")
                fig.update_yaxes(title_text="Annual Savings", secondary_y=False)
                fig.update_yaxes(title_text="Payback (Years)", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.bar(
                    solar_df,
                    x=solar_df["solar_capacity_kWp"].astype(str) + " kWp",
                    y="net_savings_25y",
                    title="25-Year Net Savings",
                    color="net_savings_25y",
                    color_continuous_scale="Greens",
                )
                fig.update_layout(yaxis_title="Net Savings (25 yr)")
                st.plotly_chart(fig, use_container_width=True)

        roi_assumptions = solar_roi.get("assumptions", [])
        if roi_assumptions:
            with st.expander("📋 Solar ROI Assumptions"):
                for a in roi_assumptions:
                    st.markdown(f"- {a}")

    else:
        st.warning("No solar ROI data. Run the analysis first.")


# TAB 5: Recommendations
with tab5:
    st.subheader("💡 AI-Generated Energy Recommendations")

    recs_data = state.get("energy_recommendations", {})
    if recs_data:
        recs_text = recs_data.get("recommendations", "")
        if recs_text:
            st.markdown(recs_text)

        model_used = recs_data.get("model", "")
        if model_used:
            st.caption(f"Generated by: {model_used}")

        with st.expander("📋 Prompt Sent to LLM"):
            st.code(recs_data.get("prompt", ""), language="text")
    else:
        st.warning("No recommendations available. Make sure OPENAI_API_KEY is set and run the analysis.")


# ─────────────────────────────────────────────
# RAW STATE (Debug)
# ─────────────────────────────────────────────

with st.expander("🔍 Raw State (Debug)", expanded=False):
    # Filter out non-serializable items
    debug_state = {
        "profile_data": pd_data,
        "messages": msgs,
        "errors": errors,
        "workflow_stage": state.get("workflow_stage"),
        "current_agent": state.get("current_agent"),
        "similar_households_comparison": state.get("similar_households_comparison"),
        "energy_recommendations": state.get("energy_recommendations"),
    }
    st.json(debug_state)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.caption(
    "**Architecture:** LangGraph StateGraph → 11 Sequential Agents → OpenAI GPT-3.5-turbo  \n"
    "**Pipeline:** Profile (UI) → Occupancy (UI) → Appliances (UI) → Envelope (UI) → Renewables (UI) "
    "→ Gross Energy → Insulation Adj → Grid Draw → Benchmark → Solar ROI → Recommendations  \n"
    "**Pattern:** Planner-Executor with Human-in-the-Loop (Streamlit sidebar)"
)
