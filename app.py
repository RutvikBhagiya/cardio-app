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
    if "results" in st.session_state:
        st.session_state.results = None

# --- CSS (glass + app theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #030712;
        color: #f8fafc;
    }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at 20% 20%, rgba(6, 182, 212, 0.15) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%);
        z-index: -1;
    }

    .glass-card {
        background: rgba(17, 24, 39, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 35px;
        margin-bottom: 25px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
    }

    .brand-logo {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -1.5px;
        background: linear-gradient(90deg, #22d3ee, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .section-header {
        color: #22d3ee;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 20px;
    }

    /* make bordered containers look like glass cards */
    [data-testid="stContainer"][data-border="true"] {
        background: rgba(17, 24, 39, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 35px;
        margin-bottom: 25px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
    }

    .stSlider > div > div > div { background: #22d3ee !important; }
    .stButton > button {
        background: linear-gradient(135deg, #0891b2 0%, #4f46e5 100%) !important;
        border-radius: 12px !important;
        border: none !important;
        height: 3.5rem !important;
        font-weight: 700 !important;
        transition: 0.4s all ease !important;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE ---
if "page" not in st.session_state:
    st.session_state.page = "Diagnostic"
if "results" not in st.session_state:
    st.session_state.results = None

# --- NAVIGATION ---
def draw_nav():
    c1, c2, c3, c4 = st.columns([2.5, 1, 1, 1])
    with c1:
        st.markdown('<div class="brand-logo">CARDIO.SHIELD AI</div>', unsafe_allow_html=True)
    with c2:
        if st.button("DIAGNOSTIC", use_container_width=True):
            st.session_state.page = "Diagnostic"
    with c3:
        if st.button("ANALYTICS", use_container_width=True):
            st.session_state.page = "Analytics"
    with c4:
        if st.button("INTELLIGENCE", use_container_width=True):
            st.session_state.page = "Intelligence"

draw_nav()
st.markdown("<br>", unsafe_allow_html=True)

# --- SPIDER CHART ---
def create_spider_chart(data):
    categories = ['Age', 'BMI', 'Systolic', 'Diastolic', 'Cholesterol']
    values = [
        data['age'] / 100,
        min(data['bmi'] / 45, 1),
        min(data['sbp'] / 220, 1),
        min(data['dbp'] / 140, 1),
        data['chol'] / 3
    ]
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line=dict(color='#22d3ee', width=3),
        fillcolor='rgba(34, 211, 238, 0.25)',
        marker=dict(color='#818cf8', size=8)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False),
            gridshape='linear',
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=30),
        height=380
    )
    return fig

# --- PAGE: DIAGNOSTIC ---
if st.session_state.page == "Diagnostic":
    col_in, col_res = st.columns([1.2, 1], gap="large")

    # LEFT: inputs inside a bordered container, styled as glass by CSS
    with col_in:
        with st.container(border=True):
            st.markdown(
                "<h3 style='margin-top:0; color:#22d3ee;'>🧬 Patient Vitals</h3>",
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 18, 100, 50, on_change=clear_results)
                height = st.number_input("Height (cm)", 100, 250, 175, on_change=clear_results)
                sbp = st.slider("Systolic BP", 80, 220, 120, on_change=clear_results)
            with c2:
                gender_val = st.selectbox("Gender", ["Female", "Male"], on_change=clear_results)
                gender = 1 if gender_val == "Female" else 2
                weight = st.number_input("Weight (kg)", 30, 200, 75, on_change=clear_results)
                dbp = st.slider("Diastolic BP", 40, 140, 80, on_change=clear_results)

            st.markdown("<hr style='opacity:0.1'>", unsafe_allow_html=True)

            st.markdown(
                "<p style='font-size:0.85rem; opacity:0.6; margin-bottom:5px;'>Lab Markers</p>",
                unsafe_allow_html=True
            )
            lab_c1, lab_c2 = st.columns(2)
            with lab_c1:
                chol = st.select_slider("Cholesterol", [1, 2, 3], value=1, on_change=clear_results)
            with lab_c2:
                gluc = st.select_slider("Glucose", [1, 2, 3], value=1, on_change=clear_results)

            st.markdown(
                "<p style='font-size:0.85rem; opacity:0.6; margin-top:15px; margin-bottom:5px;'>Lifestyle Factors</p>",
                unsafe_allow_html=True
            )
            life_c1, life_c2, life_c3 = st.columns(3)
            with life_c1:
                active = st.toggle("Active", True, on_change=clear_results)
            with life_c2:
                smoke = st.toggle("Smoking", on_change=clear_results)
            with life_c3:
                alco = st.toggle("Alcohol Usage", on_change=clear_results)

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("COMPUTE RISK PROJECTION", use_container_width=True):
                bmi = weight / ((height / 100) ** 2)
                bp_diff = sbp - dbp
                features = pd.DataFrame(
                    [[gender, sbp, dbp, chol, gluc, int(smoke), int(alco), int(active), age, bmi, bp_diff]],
                    columns=[
                        'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
                        'smoke', 'alco', 'active', 'age_years', 'bmi', 'bp_diff'
                    ]
                )
                prob = model.predict_proba(features)[0][1] * 100
                st.session_state.results = {
                    "prob": prob,
                    "bmi": bmi,
                    "age": age,
                    "sbp": sbp,
                    "dbp": dbp,
                    "chol": chol
                }
                st.rerun()

    # RIGHT: result cards
    with col_res:
        if st.session_state.results:
            res = st.session_state.results
            color = "#ef4444" if res['prob'] > 70 else "#facc15" if res['prob'] > 35 else "#22c55e"

            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <h1 style='color: {color}; font-size: 4.5rem; margin:0; line-height:1;'>{res['prob']:.1f}%</h1>
                <p style='opacity:0.6; letter-spacing: 3px; font-weight:500;'>CUMULATIVE RISK</p>
                <div style='width: 100%; height: 12px; background: rgba(255,255,255,0.05); border-radius: 20px; margin: 30px 0;'>
                    <div style='width: {res['prob']}%; height: 100%; background: linear-gradient(90deg, {color}, #fff); border-radius: 20px; box-shadow: 0 0 20px {color};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(create_spider_chart(res), use_container_width=True, config={'displayModeBar': False})

            st.markdown(f"""
            <div class="glass-card" style='display:flex; justify-content:space-around; background: rgba(255,255,255,0.05); padding: 15px; border-radius: 15px;'>
                <div><p style='margin:0; opacity:0.5; font-size:0.8rem;'>BMI</p><b>{res['bmi']:.1f}</b></div>
                <div><p style='margin:0; opacity:0.5; font-size:0.8rem;'>AGE</p><b>{res['age']}</b></div>
                <div><p style='margin:0; opacity:0.5; font-size:0.8rem;'>BP</p><b>{res['sbp']}/{res['dbp']}</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="height: 580px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; border-style: dashed; border-color: rgba(34, 211, 238, 0.2);">
                <div style='font-size: 5rem; margin-bottom: 20px; filter: drop-shadow(0 0 10px rgba(34, 211, 238, 0.4));'>🔬</div>
                <h3 style="color: #22d3ee; margin-bottom: 10px;">Waiting for Data</h3>
                <p style="opacity: 0.6; max-width: 250px;">Enter patient vitals to generate a neural health profile.</p>
            </div>
            """, unsafe_allow_html=True)

# --- PAGE: ANALYTICS ---
elif st.session_state.page == "Analytics":
    if hasattr(model, 'feature_importances_'):
        feat_names = [
            'Gender', 'Systolic BP', 'Diastolic BP', 'Cholesterol',
            'Glucose', 'Smoking', 'Alcohol', 'Activity', 'Age', 'BMI', 'Pulse Pressure'
        ]
        imp_df = pd.DataFrame(
            {'Factor': feat_names, 'Importance': model.feature_importances_}
        ).sort_values('Importance', ascending=True)

        fig = go.Figure()
        for i in range(len(imp_df)):
            fig.add_shape(
                type='line',
                x0=0, y0=i,
                x1=imp_df['Importance'].iloc[i], y1=i,
                line=dict(color="#19D5DC", width=3)
            )

        fig.add_trace(go.Scatter(
            x=imp_df['Importance'],
            y=imp_df['Factor'],
            mode='markers',
            marker=dict(
                size=18,
                color=imp_df['Importance'],
                colorscale=[[0, "#E09999"], [1, "#f90101"]],
                line=dict(color="#09cbfb", width=3)
            ),
            hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>"
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=False)
        )

        st.markdown("""
        <div class="glass-card">
            <div class="section-header">📊 Neural Feature Sensitivity</div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="glass-card" style="margin-top:20px; padding:20px; background:rgba(129, 140, 248, 0.1); border-radius:15px; border-left:4px solid #818cf8;">
            <p style="margin:0; font-size:0.95rem; opacity:0.9;">
                <b>Insight:</b> Patients with consistently elevated <b>systolic pressure</b> and <b>BMI</b> above the healthy range show a significantly higher predicted risk, even when laboratory markers like <b>cholesterol</b> and <b>glucose</b> are only mildly abnorma
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">📊 Neural Feature Sensitivity</div>
            <p style="opacity:0.7;">Feature importances are not available for this model.</p>
        </div>
        """, unsafe_allow_html=True)

# --- PAGE: INTELLIGENCE ---
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card" style="height: 350px;">
            <div class="section-header">🧠 Neural Engine</div>
            <p style='opacity:0.8;'>This system utilizes a <b>Random Forest Ensemble</b>. Unlike linear regression, this architecture handles complex feature interactions, such as the relationship between high BMI and increased Systolic Pressure over time.</p>
            <p style='color:#818cf8; font-weight:600;'>Validation Method: Cross-Entropy Recall</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card" style="height: 350px; border-left: 5px solid #ef4444;">
            <div class="section-header" style="color:#ef4444;">⚠️ Project Protocol</div>
            <p style='opacity:0.8;'><b>Classification:</b> Placement Level Academic Project.</p>
            <p style='opacity:0.8;'><b>Accuracy:</b> Clinically simulated for educational demonstration. Not for medical diagnosis.</p>
            <p style='opacity:0.8;'><b>Privacy:</b> Zero-Retention. No data is stored on the server.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; opacity: 0.3; font-size: 0.75rem; margin-top: 50px; line-height: 1.4;'>
    CARDIO.SHIELD ENGINE v2.6 • 2026 © Rutvik Bhagiya<br>
    <span style='font-size: 0.65rem;'>
        Academic Research Project • ML-Powered Cardiovascular Risk Assessment
    </span>
</p>
""", unsafe_allow_html=True)
