import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Unemployment Rate Analysis", page_icon="💎", layout="wide")

# --- CYBER NEON UI STYLING ---
st.markdown("""
    <style>
    .stApp { background: #0f172a; color: #e2e8f0; }
    div[data-testid="stMetric"] { 
        background: rgba(30, 41, 59, 0.7); 
        border: 1px solid #38bdf8; 
        border-radius: 15px; 
    }
    h1, h2, h3 { color: #38bdf8; text-shadow: 0 0 10px rgba(56, 189, 248, 0.3); }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color: #38bdf8 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING & CLEANING ---
@st.cache_data
def load_clean_data():
    try:
        data = pd.read_csv('Survay_Responses.csv', encoding='latin1')
        # FIX: Fill empty strings with 'Unknown' to prevent Sunburst/Treemap 'leaf' errors
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].fillna("Unknown").str.strip()
        return data
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_assets():
    try:
        with open('unemployment_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

df = load_clean_data()
assets = load_assets()

# --- SIDEBAR (Persistent Control Panel) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3281/3281306.png", width=80)
    st.title("📂 Control Panel")
    st.divider()
    
    if not df.empty:
        st.metric("Total Sample", len(df))
        unemp_rate = (df['employment_status'] == 'Unemployed').mean() * 100
        st.metric("Unemployment Rate", f"{unemp_rate:.1f}%")
    
    st.divider()
    st.info("System Status: Operational 🟢")
    st.caption("Research Project 2026")

# --- MAIN DASHBOARD ---
st.title("💎 Unemployment Rate Analysis")
tab1, tab2, tab3 = st.tabs(["🏠 Summary", "📊 Advanced Analytics", "🤖 Prediction Engine"])

with tab1:
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.subheader("Structural Findings")
        # Treemap: Clean data to prevent ValueError
        tree_df = df[df['proposed_solution'] != "Unknown"]
        fig_tree = px.treemap(tree_df, path=['proposed_solution'], color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_tree.update_layout(margin=dict(t=0, l=0, r=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_tree, use_container_width=True)

    with c2:
        st.subheader("Education Distribution")
        fig_edu = px.pie(df, names='education_level', hole=0.7, color_discrete_sequence=px.colors.sequential.Darkmint)
        fig_edu.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_edu, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    # Corrected Indentation for Column A
    with col_a:
        st.subheader("Gender vs. Employment Hierarchy")
        # Sunburst: Ensuring valid leaf rows
        sun_df = df[['gender', 'employment_status']].replace("", "Unknown").dropna()
        fig_sun = px.sunburst(sun_df, path=['gender', 'employment_status'], color_discrete_sequence=px.colors.qualitative.T10)
        fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_sun, use_container_width=True)
        
    with col_b:
        st.subheader("Employment Density Heatmap")
        heatmap_data = pd.crosstab(df['education_level'], df['employment_status'])
        fig_heat = px.imshow(heatmap_data, text_auto=True, color_continuous_scale='Blues')
        fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    if assets:
        st.header("Crisis Perception AI")
        cin, cout = st.columns([0.4, 0.6])
        with cin:
            age = st.selectbox("Age Group", list(assets['age_map'].keys()))
            edu = st.selectbox("Education", list(assets['edu_map'].keys()))
            emp = st.selectbox("Status", list(assets['emp_map'].keys()))
            gen = st.selectbox("Gender", assets['encoders']['gender'].classes_)
            skl = st.selectbox("Skills Match?", assets['encoders']['skill_alignment'].classes_)
            trn = st.selectbox("Training?", assets['encoders']['skill_training'].classes_)
            sk  = st.selectbox("Job Seeking?", assets['encoders']['job_seeking_status'].classes_)
            
            if st.button("Generate Prediction", use_container_width=True):
                x_in = [[assets['age_map'][age], assets['edu_map'][edu], assets['emp_map'][emp],
                         assets['encoders']['gender'].transform([gen])[0],
                         assets['encoders']['skill_alignment'].transform([skl])[0],
                         assets['encoders']['skill_training'].transform([trn])[0],
                         assets['encoders']['job_seeking_status'].transform([sk])[0]]]
                prediction = assets['model'].predict(x_in)[0]
                
                with cout:
                    st.metric("Predicted Severity Score", f"{prediction:.2f}/5.0")
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number", value = prediction,
                        gauge = {'axis': {'range': [1, 5]}, 'bar': {'color': "#38bdf8"}}
                    ))
                    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
                    st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.error("Model assets not found. Run the training script first.")