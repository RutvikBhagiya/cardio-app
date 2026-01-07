import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(
    page_title="CardioShield AI | Neural Diagnostic",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("cardio_rf_model.pkl")
    except:
        st.error("🚨 Model File Missing: 'cardio_rf_model.pkl' not found.")
        st.stop()

model = load_model()

# --- CALLBACK TO CLEAR RESULTS ---
def clear_results():
    st.session_state.results = None
    st.session_state.static_spider_fig = None
    st.session_state.prediction_made = False

# --- CSS (ULTRA-RESPONSIVE PURE BLACK & RED THEME) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');

    /* --- GLOBAL BACKGROUND --- */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #000000 !important;
        color: #f8fafc;
    }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at 20% 20%, rgba(255, 0, 0, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(255, 0, 0, 0.05) 0%, transparent 50%);
        z-index: -1;
    }

    /* --- GLASS CARDS & CONTAINERS --- */
    .glass-card, [data-testid="stContainer"][data-border="true"] {
        background: rgba(17, 24, 39, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 25px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
        width: 100%;
    }

    /* --- BRANDING --- */
    .brand-logo {
        font-size: clamp(1.5rem, 5vw, 2.2rem);
        font-weight: 700;
        letter-spacing: -1.5px;
        background: linear-gradient(90deg, #ff0000, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s ease-in-out infinite alternate;
        text-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
        white-space: nowrap;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(255, 0, 0, 0.3)); }
        to { filter: drop-shadow(0 0 25px rgba(255, 0, 0, 0.6)); }
    }
    
    label, .stMarkdown p {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    .section-header {
        color: #ff0000;
        font-size: clamp(1.1rem, 3vw, 1.4rem);
        font-weight: 600;
        margin-bottom: 20px;
    }

    /* --- NAVBAR BUTTONS --- */
    div[data-testid="column"] button p {
        color: #ff0000 !important;
        font-weight: 700;
        font-size: clamp(0.7rem, 2vw, 1rem);
    }

    /* --- BUTTON STYLES --- */
    .stButton > button {
        background: linear-gradient(135deg, #991b1b 0%, #000000 100%) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 0, 0, 0.3) !important;
        height: 3.5rem !important;
        font-weight: 700 !important;
        transition: 0.4s all ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 20px 40px rgba(255, 0, 0, 0.2) !important;
    }

    /* --- INPUT FIELDS --- */
    input, textarea, [data-baseweb="select"] {
        background-color: #000000 !important;
        color: #ff0000 !important;
        border: 1px solid rgba(255, 0, 0, 0.35) !important;
    }

    div[data-baseweb="select"] input {
        caret-color: transparent !important;
    }
    
    div[data-baseweb="pill"] span {
        color: #ff0000 !important; /* Forces Red text */
        font-weight: 700 !important;
    }

    /* --- MODEL PERFORMANCE METRIC CARDS --- */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.15), rgba(0, 0, 0, 0.5)) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(15px) !important;
        transition: all 0.3s ease !important;
        text-align: center !important;
        padding: 1.25rem !important;
        border-radius: 16px !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 100px;
    }

    .metric-card:hover {
        transform: translateY(-8px) !important;
        box-shadow: 0 15px 30px rgba(255, 0, 0, 0.2) !important;
    }

    /* --- STATIC CONFUSION MATRIX CELLS --- */
    .conf-cell {
        background: rgba(17, 24, 39, 0.6) !important;
        text-align: center !important;
        padding: 1.5rem 0.5rem !important;
        border-radius: 16px !important;
        transition: none !important;
    }
    .conf-cell h2 {
        font-size: clamp(1.5rem, 4vw, 2.8rem) !important;
        margin: 0 !important;
        font-weight: 800 !important;
        line-height: 1 !important;
    }
    .conf-cell p {
        font-size: clamp(0.6rem, 1.5vw, 0.9rem) !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px !important;
    }

    /* --- SLIDERS --- */
    [data-testid="stSlider"] .stSlider > div > div > div:last-child {
        background: transparent !important;
        border: 2px solid #ff0000 !important;
        border-radius: 50% !important;
        box-shadow: 0 0 0 4px rgba(255, 0, 0, 0.2) !important;
    }

    /* --- MOBILE & TABLET OPTIMIZATIONS --- */
    @media (max-width: 1024px) {
        .glass-card { padding: 1.5rem; }
    }

    @media (max-width: 768px) {
        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        .metric-card, .conf-cell {
            padding: 1rem !important;
        }
    }

    @media (max-width: 480px) {
        .glass-card { padding: 1rem; border-radius: 16px; }
        .brand-logo { text-align: center; margin-bottom: 1rem; }
        h1 { font-size: 2.5rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
if "page" not in st.session_state:
    st.session_state.page = "Diagnostic"
if "results" not in st.session_state:
    st.session_state.results = None
if "static_spider_fig" not in st.session_state:
    st.session_state.static_spider_fig = None
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# --- NAVIGATION ---
def draw_nav():
    c1, c2, c3, c4 = st.columns([2.5, 1, 1, 1])
    with c1:
        st.markdown('<div class="brand-logo animate__animated animate__pulse">CARDIO.SHIELD AI</div>', unsafe_allow_html=True)
    with c2:
        if st.button("DIAGNOSTIC", width='stretch', key="nav_diag"):
            st.session_state.page = "Diagnostic"
            st.session_state.prediction_made = False
    with c3:
        if st.button("ANALYTICS", width='stretch', key="nav_ana"):
            st.session_state.page = "Analytics"
    with c4:
        if st.button("ABOUT", width='stretch', key="nav_about"):
            st.session_state.page = "About"

draw_nav()
st.markdown("<br>", unsafe_allow_html=True)

# --- SPIDER CHART FUNCTION ---
def create_spider_chart(data):
    categories = ['Age', 'BMI', 'Systolic', 'Diastolic', 'Cholesterol']
    values = [data['age']/100, min(data['bmi']/45, 1), min(data['sbp']/220, 1), min(data['dbp']/140, 1), (data['chol']-1)/2]
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        line=dict(color='#ff0000', width=3),
        fillcolor='rgba(255, 0, 0, 0.25)',
        marker=dict(color='#ff0000', size=8)
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False), gridshape='linear', bgcolor="rgba(0,0,0,0)"),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=30), height=380
    )
    return fig

# --- DIAGNOSTIC PAGE ---
if st.session_state.page == "Diagnostic":
    col_in, col_res = st.columns([1.2, 1], gap="large")

    with col_in:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; color:#ff0000;'>🧬 Patient Vitals</h3>", unsafe_allow_html=True)
            disabled = st.session_state.prediction_made

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p style="color:#F2F2F2; font-weight:bold;">Age</p>', unsafe_allow_html=True)
                age = st.slider("Age", 18, 100, 50, on_change=clear_results, disabled=disabled, label_visibility="collapsed")
                st.markdown('<p style="color:#F2F2F2; font-weight:bold;">Height (cm)</p>', unsafe_allow_html=True)
                height = st.number_input("Height", 100, 250, 175, on_change=clear_results, disabled=disabled, label_visibility="collapsed")
                st.markdown('<p style="color:#F2F2F2; font-weight:bold;">Systolic BP</p>', unsafe_allow_html=True)
                sbp = st.slider("SBP", 80, 220, 120, on_change=clear_results, disabled=disabled, label_visibility="collapsed")
            with c2:
                st.markdown('<p style="color:#F2F2F2; font-weight:bold;">Gender</p>', unsafe_allow_html=True)
                gender_val = st.pills("Gender", ["Female", "Male"], selection_mode="single", default="Female", on_change=clear_results, disabled=disabled,label_visibility="collapsed")
                gender = 1 if gender_val == "Male" else 2
                st.markdown('<p style="color:#F2F2F2; font-weight:bold;">Weight (kg)</p>', unsafe_allow_html=True)
                weight = st.number_input("Weight", 30, 200, 75, on_change=clear_results, disabled=disabled, label_visibility="collapsed")
                st.markdown('<p style="color:#F2F2F2; font-weight:bold;">Diastolic BP</p>', unsafe_allow_html=True)
                dbp = st.slider("DBP", 40, 140, 80, on_change=clear_results, disabled=disabled, label_visibility="collapsed")

            st.markdown("<hr style='opacity:0.1'>", unsafe_allow_html=True)
            st.markdown('<p style="color:#F2F2F2; font-size:18px; font-weight:bold;">Lab & Lifestyle</p>', unsafe_allow_html=True)
            
            lab_c1, lab_c2 = st.columns(2)
            with lab_c1:
                chol = st.select_slider("Cholesterol", [1, 2, 3], value=1, on_change=clear_results, disabled=disabled)
            with lab_c2:
                gluc = st.select_slider("Glucose", [1, 2, 3], value=1, on_change=clear_results, disabled=disabled)

            life_c1, life_c2, life_c3 = st.columns(3)
            active = life_c1.toggle("Active", True, on_change=clear_results, disabled=disabled)
            smoke = life_c2.toggle("Smoking", on_change=clear_results, disabled=disabled)
            alco = life_c3.toggle("Alcohol", on_change=clear_results, disabled=disabled)

            if st.button("🔬 COMPUTE RISK PROJECTION", width='stretch', disabled=st.session_state.prediction_made):
                bmi = weight / ((height / 100) ** 2)
                features = pd.DataFrame([[gender, sbp, dbp, chol, gluc, int(smoke), int(alco), int(active), age, bmi, sbp-dbp]], 
                                     columns=['gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years', 'bmi', 'bp_diff'])
                prob = model.predict_proba(features)[0][1] * 100
                st.session_state.results = {"prob": prob, "bmi": bmi, "age": age, "sbp": sbp, "dbp": dbp, "chol": chol}
                st.session_state.static_spider_fig = create_spider_chart(st.session_state.results)
                st.session_state.prediction_made = True

    with col_res:
        if st.session_state.prediction_made:
            res = st.session_state.results
            color = "#ef4444" if res['prob'] > 70 else "#facc15" if res['prob'] > 35 else "#22c55e"
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <h1 style='color: {color}; font-size: clamp(3rem, 10vw, 4.5rem); margin:0;'>{res['prob']:.1f}%</h1>
                <p style='opacity:0.6; letter-spacing: 3px;'>CUMULATIVE RISK</p>
                <div style='width: 100%; height: 12px; background: rgba(255,255,255,0.05); border-radius: 20px; margin: 30px 0;'>
                    <div style='width: {res['prob']}%; height: 100%; background: linear-gradient(90deg, {color}, #fff); border-radius: 20px; box-shadow: 0 0 20px {color};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(st.session_state.static_spider_fig, width='stretch', config={'staticPlot': True})
            st.markdown(f"""
            <div class="glass-card" style='display:flex; justify-content:space-around; background: rgba(255,255,255,0.05); padding: 15px; flex-wrap: wrap; gap: 10px;'>
                <div style="flex: 1; text-align: center;"><p style='margin:0; opacity:0.5; font-size:0.8rem;'>BMI</p><b>{res['bmi']:.1f}</b></div>
                <div style="flex: 1; text-align: center;"><p style='margin:0; opacity:0.5; font-size:0.8rem;'>AGE</p><b>{res['age']}</b></div>
                <div style="flex: 1; text-align: center;"><p style='margin:0; opacity:0.5; font-size:0.8rem;'>BP</p><b>{res['sbp']}/{res['dbp']}</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass-card" style="min-height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; border-style: dashed; border-color: #ff000033;"><div style="font-size: 5rem;">🔬</div><h3 style="color: #ff0000;">Waiting for Data</h3><p style="opacity: 0.6;">Enter patient vitals to generate profile.</p></div>', unsafe_allow_html=True)

# --- ANALYTICS PAGE ---
elif st.session_state.page == "Analytics":
    with st.container():
        st.markdown('<div class="glass-card"><div class="section-header">🧠 Neural Feature Sensitivity</div>', unsafe_allow_html=True)
        if hasattr(model, 'feature_importances_'):
            feat_names = ['Gender', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'Glucose', 'Smoking', 'Alcohol', 'Activity', 'Age', 'BMI', 'Pulse Pressure']
            imp_df = pd.DataFrame({'Factor': feat_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
            fig = go.Figure()
            for i in range(len(imp_df)):
                fig.add_shape(type='line', x0=0, y0=i, x1=imp_df['Importance'].iloc[i], y1=i, line=dict(color="#ff0000", width=3))
            fig.add_trace(go.Scatter(x=imp_df['Importance'], y=imp_df['Factor'], mode='markers', marker=dict(size=18, color=imp_df['Importance'], colorscale='Reds', line=dict(color="#ff0000", width=2))))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450, margin=dict(l=20, r=20, t=20, b=10), xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig, width='stretch', config={"staticPlot": True})
        st.markdown("</div>", unsafe_allow_html=True)

    col_perf, col_conf = st.columns(2, gap="large")
    with col_perf:
        st.markdown(f"""
        <div class="glass-card">
            <div class="section-header">🎯 Model Performance</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px;">
                <div class="metric-card"><h2 style='color:#ff0000; margin:0;'>73%</h2><p>ACCURACY</p></div>
                <div class="metric-card"><h2 style='color:#ff0000; margin:0;'>68.1%</h2><p>RECALL</p></div>
                <div class="metric-card"><h2 style='color:#ff0000; margin:0;'>71.2%</h2><p>F1-SCORE</p></div>
                <div class="metric-card"><h2 style='color:#ff0000; margin:0;'>74.7%</h2><p>PRECISION</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_conf:
        st.markdown(f"""
        <div class="glass-card">
            <div class="section-header">📈 Confusion Matrix</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div class="conf-cell" style="border: 2px solid #10b981;">
                    <h2 style='color:#10b981;'>5366</h2><p>True Positive</p>
                </div>
                <div class="conf-cell" style="border: 2px solid #ef4444;">
                    <h2 style='color:#ef4444;'>1536</h2><p>False Positive</p>
                </div>
                <div class="conf-cell" style="border: 2px solid #ef4444;">
                    <h2 style='color:#ef4444;'>2126</h2><p>False Negative</p>
                </div>
                <div class="conf-cell" style="border: 2px solid #10b981;">
                    <h2 style='color:#10b981;'>4546</h2><p>True Negative</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- ABOUT PAGE ---
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="glass-card" style="height: 420px;">
                    <div class="section-header">🧠 Neural Engine</div>
                    <p>This system utilizes a <b>Random Forest Ensemble</b> trained on 54,296 records.</p><p style="opacity:0.8;">Unlike linear regression, this architecture handles complex non-linear feature interactions.</p>
                    <p style='color:#818cf8; font-weight:600;'>Validation Method: Cross-Validation Recall</p>
                    <p style='opacity:0.8;'><b>Feature Handling:</b> No feature scaling required. Tree-based splits operate directly on raw clinical values.</p>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="glass-card" style="height: 420px; border-left: 5px solid #ef4444;">
                    <div class="section-header" style="color:#ef4444;">⚠️ Project Protocol</div>
                    <p>Academic project for placement demonstration. Not for medical diagnosis.</p>
                    <p style='opacity:0.8;'><b>Classification:</b> Beginner Level Academic Project.</p>
                    <p style='opacity:0.8;'><b>Accuracy:</b> Clinically simulated for educational demonstration. Not for medical diagnosis.</p>
                    <p style='opacity:0.8;'><b>Privacy:</b> Zero-Retention. No data is stored on the server.</p>
                    <p style='opacity:0.8;'><b>Model Behavior:</b> Balanced learning observed with minimal overfitting between training and testing data.</p>
            </div>""", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; opacity: 0.3; font-size: 0.75rem; margin-top: 50px;'>CARDIO.SHIELD • 2026 © Rutvik Bhagiya</p>", unsafe_allow_html=True)