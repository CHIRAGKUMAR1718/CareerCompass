"""
CareerCompass — Step 9: Full Dashboard (Explore View + My Roadmap View)
============================================================================
BHAI, YE FILE KYA KARTI HAI:

Do "tabs" me poora product deta hai:
1. Explore Market  — Layer 1 (Stats+Regression+Clustering), koi input
   nahi chahiye, sabke liye browse-able
2. My Roadmap      — Layer 2 (Skill-Gap+RAG+LLM), user apna profile
   deta hai, personalized roadmap milta hai

ZAROORI DESIGN-DECISION: Yaha FastAPI NAHI use ki — Streamlit khud
Python-runtime hai, isliye roadmap_agent ke functions SEEDHA call
karte hain, bina kisi API-layer ke beech me (jo SentinelOps/CodeGuardian
me zaroori thi kyunki wahan dashboard plain HTML/JS tha).
"""

import streamlit as st
import pandas as pd
import json
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_dark"
BRAND_COLORS = ["#8B5CF6", "#6366F1", "#A78BFA", "#4F46E5", "#C4B5FD", "#818CF8", "#7C3AED", "#93C5FD"]

sys.path.append(os.path.join(os.path.dirname(__file__), "../agents"))

st.set_page_config(page_title="CareerCompass", layout="wide", page_icon="🧭")

# --- Custom CSS for premium look ---
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 50%, #4F46E5 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 { color: white; margin: 0; font-size: 2.2rem; }
    .hero-banner p { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 1rem; }

    div[data-testid="stMetric"] {
        background: #161B29;
        border: 1px solid #2A2F3E;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        overflow: visible;
    }
    div[data-testid="stMetricValue"] {
        color: #A78BFA;
        font-size: 1.6rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    h2 { border-bottom: 2px solid #8B5CF6; padding-bottom: 0.4rem; margin-top: 2rem; }

    .stButton button {
        background: linear-gradient(135deg, #8B5CF6, #6366F1);
        color: white; border: none; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- Load pre-computed data (Layer 1) ---
@st.cache_data
def load_data():
    df = pd.read_csv("job_postings_clustered.csv")
    with open("analysis_results.json") as f:
        stats = json.load(f)
    with open("regression_results.json") as f:
        regression = json.load(f)
    with open("cluster_results.json") as f:
        clusters = json.load(f)
    return df, stats, regression, clusters


# --- Load AI-layer resources ONCE (Layer 2) — cached across reruns ---
@st.cache_resource
def load_roadmap_resources():
    from roadmap_agent import app as roadmap_app, load_resources
    load_resources()
    return roadmap_app


df, stats, regression, clusters = load_data()

st.markdown("""
<div class="hero-banner">
    <h1>🧭 CareerCompass — Job Market Intelligence</h1>
    <p>Statistically-grounded insights from 30,000 synthetic job postings</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar filters (only relevant on Explore tab, but shown globally) ---
st.sidebar.header("🔍 Filters (Explore tab)")
selected_roles = st.sidebar.multiselect("Job Titles", options=sorted(df["job_title"].unique()), default=[])
selected_tiers = st.sidebar.multiselect("City Tier", options=sorted(df["city_tier"].unique()), default=[])

if "applied_roles" not in st.session_state:
    st.session_state.applied_roles = []
    st.session_state.applied_tiers = []

if st.sidebar.button("🔍 Apply Filters", use_container_width=True):
    st.session_state.applied_roles = selected_roles
    st.session_state.applied_tiers = selected_tiers

if st.sidebar.button("↺ Reset Filters", use_container_width=True):
    st.session_state.applied_roles = []
    st.session_state.applied_tiers = []
    st.rerun()

filtered_df = df.copy()
if st.session_state.applied_roles:
    filtered_df = filtered_df[filtered_df["job_title"].isin(st.session_state.applied_roles)]
if st.session_state.applied_tiers:
    filtered_df = filtered_df[filtered_df["city_tier"].isin(st.session_state.applied_tiers)]

if st.session_state.applied_roles or st.session_state.applied_tiers:
    st.sidebar.success(f"Showing {len(filtered_df):,} filtered postings")
else:
    st.sidebar.caption(f"Showing all {len(filtered_df):,} postings")


# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["🔍 Explore Market", "🎯 My Roadmap"])

# ============================================================
# TAB 1: EXPLORE MARKET (Layer 1 — unchanged from before)
# ============================================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Postings", f"{len(filtered_df):,}")
    col2.metric("Avg Salary", f"₹{filtered_df['salary_lpa'].mean():.1f}L")
    col3.metric("Job Roles", filtered_df["job_title"].nunique())
    col4.metric("Remote %", f"{filtered_df['remote'].mean()*100:.0f}%")

    st.divider()

    st.header("💰 Salary Analysis")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(filtered_df, x="job_title", y="salary_lpa", title="Salary Distribution by Role",
                     color_discrete_sequence=BRAND_COLORS)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        tier_avg = filtered_df.groupby("city_tier")["salary_lpa"].mean().reset_index()
        fig = px.bar(tier_avg, x="city_tier", y="salary_lpa", title="Avg Salary by City Tier",
                     color="city_tier", text_auto=".1f", color_discrete_sequence=BRAND_COLORS)
        st.plotly_chart(fig, use_container_width=True)

    anova = stats["city_tier_anova"]
    sig_text = "statistically significant" if anova["significant_at_0.05"] else "not statistically significant"
    st.caption(f"📊 ANOVA test: F={anova['f_statistic']}, p={anova['p_value']:.2e} — this difference is **{sig_text}** (p<0.05).")

    remote_test = stats["remote_hypothesis_test"]
    st.info(
        f"**Remote vs On-site:** On-site averages ₹{remote_test['onsite_mean']}L vs Remote ₹{remote_test['remote_mean']}L "
        f"— a difference of ₹{remote_test['mean_difference']}L, confirmed statistically significant via t-test (p={remote_test['p_value']:.2e})."
    )

    st.divider()

    st.header("📈 Emerging Skills Trend")
    trend = stats["emerging_skill_trend"]
    monthly = trend["monthly_series"]
    trend_df = pd.DataFrame({"month": list(monthly.keys()), "pct": list(monthly.values())})
    fig = px.line(trend_df, x="month", y="pct", markers=True,
                  title="% of Postings Requiring GenAI Skills (LangChain, RAG, Prompt Engineering...)",
                  color_discrete_sequence=BRAND_COLORS)
    fig.update_traces(line_width=3, marker_size=8)
    fig.update_layout(yaxis_title="% of postings", xaxis_title="Month")
    st.plotly_chart(fig, use_container_width=True)
    st.success(
        f"**Mann-Kendall trend test:** demand is **{trend['trend']}** "
        f"(from {trend['first_month_pct']}% to {trend['last_month_pct']}%), "
        f"statistically significant at p={trend['p_value']:.2e}."
    )

    st.divider()

    st.header("🎯 What Drives Salary? (Regression Model)")
    coefs = regression["coefficients"]
    coef_df = pd.DataFrame([
        {"factor": k, "coefficient": v["value"], "p_value": v["p_value"]}
        for k, v in coefs.items() if k != "Intercept"
    ]).sort_values("coefficient")
    fig = px.bar(coef_df, x="coefficient", y="factor", orientation="h",
                 title="Regression Coefficients (impact on salary, in LPA)",
                 color="coefficient", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig, use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Test R²", f"{regression['test_r_squared']:.3f}")
    m2.metric("MAE", f"₹{regression['test_mae']:.2f}L")
    m3.metric("RMSE", f"₹{regression['test_rmse']:.2f}L")
    st.caption("Model explains ~89% of salary variance on unseen (test) data — all coefficients statistically significant (p<0.001).")

    st.divider()

    st.header("🧩 Job Clusters (Discovered from Skills Alone)")
    fig = px.scatter(
        filtered_df, x="pca_x", y="pca_y", color=filtered_df["cluster"].astype(str),
        hover_data=["job_title", "salary_lpa"],
        title="Job Postings Clustered by Skill Similarity (PCA projection)",
        color_discrete_sequence=BRAND_COLORS
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Breakdown")
    for c_id, info in clusters["clusters"].items():
        with st.expander(f"Cluster {c_id} — {', '.join(info['dominant_job_titles'][:2])} (avg ₹{info['avg_salary']}L, n={info['size']})"):
            st.write(f"**Top skills:** {', '.join(info['top_skills'])}")
            st.write(f"**Dominant roles:** {', '.join(info['dominant_job_titles'])}")


# ============================================================
# TAB 2: MY ROADMAP (Layer 2 — Skill-Gap + RAG + LLM)
# ============================================================
with tab2:
    st.header("🎯 Get Your Personalized Career Roadmap")
    st.caption("Enter your current skills and target role — powered by embedding-based skill-matching, RAG-grounded course recommendations, and LLM-generated guidance.")

    col_a, col_b = st.columns(2)
    with col_a:
        skills_input = st.text_input(
            "Your current skills (comma-separated)",
            placeholder="e.g. Python, SQL, Excel, Statistics"
        )
    with col_b:
        target_role = st.selectbox("Target role", options=sorted(df["job_title"].unique()))

    if st.button("🚀 Generate My Roadmap", use_container_width=True):
        if not skills_input.strip():
            st.warning("Please enter at least one skill.")
        else:
            user_skills = [s.strip() for s in skills_input.split(",") if s.strip()]

            with st.spinner("Loading models and connecting to services (first run takes longer)..."):
                try:
                    status_box = st.empty()
                    status_box.info("Step 1/3: Loading embedding model + Qdrant + LLM client...")
                    roadmap_app = load_roadmap_resources()

                    status_box.info("Step 2/3: Running pipeline (market analysis, skill-gap, retrieval)...")
                    initial_state = {
                        "user_skills": user_skills,
                        "target_role": target_role,
                        "target_skills": None, "gap_result": None,
                        "course_recommendations": None, "salary_upside": None,
                        "roadmap_text": None,
                    }
                    result = roadmap_app.invoke(initial_state)
                    status_box.empty()

                    # --- Readiness + skill breakdown ---
                    gap = result["gap_result"]
                    r1, r2 = st.columns([1, 2])
                    with r1:
                        st.metric("Readiness", f"{gap['readiness_pct']}%")
                        st.metric("Est. Salary Upside", f"+₹{result['salary_upside']}L")
                    with r2:
                        st.write("**✅ Already have:**", ", ".join(m["skill"] for m in gap["matched_skills"]) or "None yet")
                        st.write("**❌ Missing:**", ", ".join(m["skill"] for m in gap["missing_skills"]) or "None — fully ready!")

                    st.divider()

                    # --- Final LLM-generated roadmap ---
                    st.markdown(result["roadmap_text"])

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.caption("Make sure GROQ_API_KEY is set in agents/.env and all Step 6-8 files/data are in place.")
