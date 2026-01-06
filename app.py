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

# --- CSS (RESPONSIVE TWEAKS ADDED) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');

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
        animation: glow 2s ease-in-out infinite alternate;
        text-shadow: 0 0 20px rgba(34, 211, 238, 0.5);
        white-space: nowrap;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(34, 211, 238, 0.3)); }
        to { filter: drop-shadow(0 0 25px rgba(129, 140, 248, 0.6)); }
    }
    
    label, .stMarkdown p {
        color: #e5e7eb !important;
        opacity: 1 !important;
    }

    [data-testid="stWidgetLabel"] {
        color: #f1f5f9 !important;
        font-weight: 500;
    }


    .stButton > button {
        background: linear-gradient(135deg, #0891b2 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        height: 3.5rem !important;
        font-weight: 700 !important;
        transition: 0.4s all ease !important;
        padding: 0.5rem 0.75rem !important;
        min-width: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-shadow: none !important;
        z-index: 5;
    }
    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 20px 40px rgba(34, 211, 238, 0.12) !important;
    }
    .stButton > button:active {
        transform: translateY(-2px) !important;
    }

    .nav-btn {
        background: rgba(17, 24, 39, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        transform: translateY(0);
        position: relative;
        overflow: hidden;
    }
    .nav-btn:hover {
        background: rgba(34, 211, 238, 0.15) !important;
        border-color: #22d3ee !important;
        transform: translateY(-4px) !important;
        box-shadow: 0 20px 40px rgba(34, 211, 238, 0.2) !important;
    }
    .nav-btn::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
        transition: left 0.5s;
    }
    .nav-btn:hover::before {
        left: 100%;
    }

    .section-header {
        color: #22d3ee;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 20px;
    }

    [data-testid="stSlider"] .stSlider > div > div > div:last-child {
        background: transparent !important;
        border: 2px solid #22d3ee !important;
        border-radius: 50% !important;
        box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.2) !important;
    }

    [data-testid="stSlider"] input[type=range]::-webkit-slider-thumb {
        background: #22d3ee !important;
        border: none !important;
    }

    [data-testid="NumberInput"] input {
        background: rgba(17, 24, 39, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        backdrop-filter: blur(10px) !important;
    }

    [data-testid="stContainer"][data-border="true"] {
        background: rgba(17, 24, 39, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 35px;
        margin-bottom: 25px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.1), rgba(129, 140, 248, 0.1)) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(15px) !important;
        transition: all 0.3s ease !important;
        text-align: center !important;
        padding: 25px !important;
        border-radius: 16px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.05), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-8px) !important;
        box-shadow: 0 25px 50px rgba(34, 211, 238, 0.2) !important;
        border-color: #22d3ee !important;
    }
    .metric-card:hover::before {
        opacity: 1;
    }

    /* ------------------ RESPONSIVE RULES ------------------ */

    /* Medium and small devices */
    @media (max-width: 768px) {
        .brand-logo { font-size: 1.4rem; }
        .glass-card { padding: 20px; border-radius: 16px; }
        .section-header { font-size: 1.05rem; }
        .stButton > button { font-size: 0.95rem; height: 48px !important; }
        [data-testid="stSlider"], [data-testid="NumberInput"] input { font-size: 0.95rem !important; }
        .glass-card p, .glass-card h3, .metric-card { opacity: 0.98 !important; }
        [data-testid="stAppViewContainer"] { padding: 12px !important; }
        .stButton > button { display: block !important; width: 100% !important; }
        .stButton > button:focus { outline: none !important; box-shadow: 0 6px 18px rgba(34,211,238,0.18) !important; }
        /* make columns stack more naturally */
        .css-1lcbmhc.e1fqkh3o2, .css-1d391kg { flex-direction: column; gap: 12px; }
        /* make spider/chart scale */
        .stPlotlyChart > div, .stPlotlyChart { max-width: 100% !important; height: auto !important; }
    }

    /* Very small phones */
    @media (max-width: 420px) {
        .brand-logo { font-size: 1.1rem; letter-spacing:-0.5px; }
        .glass-card { padding: 14px; border-radius: 12px; }
        .metric-card { padding: 16px !important; }
        h1 { font-size: 3.2rem !important; }
        .glass-card[style*="height: 580px"] { height: auto !important; min-height: 350px; }
        .stButton > button { font-size: 0.9rem; padding: 8px 10px; height: 44px !important; }
        .brand-logo.animate__animated { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        /* increase contrast for inline faded texts on tiny screens */
        [style*="opacity:0.6"], [style*="opacity:0.7"], [style*="opacity:0.75"], [style*="opacity:0.8"] {
            opacity: 0.95 !important;
            color: #e6eef8 !important;
        }
    }

    /* Accessibility + touch targets */
    button, .stButton > button {
        -webkit-tap-highlight-color: rgba(255,255,255,0.04);
        touch-action: manipulation;
    }
            
    /* ---------- FORCE BUTTON TEXT VISIBILITY ---------- */
    .stButton > button,
    .stButton > button span,
    .stButton > button div {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    /* ---------- INPUT FIELD VISIBILITY ---------- */
    input, textarea, select {
        background-color: rgba(15, 23, 42, 0.95) !important;
        color: #ffffff !important;
        border: 1px solid rgba(148, 163, 184, 0.35) !important;
    }

    /* Slider & number labels */
    [data-testid="stWidgetLabel"] {
        color: #e6eef8 !important;
        opacity: 1 !important;
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
        if st.button("DIAGNOSTIC", use_container_width=True, key="nav_diag", help="Neural Risk Assessment"):
            st.session_state.page = "Diagnostic"
            if "prediction_made" in st.session_state:
                st.session_state.prediction_made = False
    with c3:
        if st.button("ANALYTICS", use_container_width=True, key="nav_ana", help="Feature Analysis"):
            st.session_state.page = "Analytics"
    with c4:
        if st.button("ABOUT", use_container_width=True, key="nav_about", help="Project Information"):
            st.session_state.page = "About"

draw_nav()
st.markdown("<br>", unsafe_allow_html=True)

# --- SPIDER CHART FUNCTION ---
def create_spider_chart(data):
    categories = ['Age', 'BMI', 'Systolic', 'Diastolic', 'Cholesterol']

    values = [
        data['age'] / 100,
        min(data['bmi'] / 45, 1),
        min(data['sbp'] / 220, 1),
        min(data['dbp'] / 140, 1),
        (data['chol'] - 1) / 2
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

# --- DIAGNOSTIC PAGE ---
if st.session_state.page == "Diagnostic":
    col_in, col_res = st.columns([1.2, 1], gap="large")

    with col_in:
        with st.container(border=True):
            st.markdown(
                "<h3 style='margin-top:0; color:#22d3ee;'>🧬 Patient Vitals</h3>",
                unsafe_allow_html=True
            )

            # **DISABLE SLIDERS WHEN PREDICTION MADE** - Key fix!
            disabled = st.session_state.prediction_made

            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 18, 100, 50, on_change=clear_results, disabled=disabled)
                height = st.number_input("Height (cm)", 100, 250, 175, on_change=clear_results, disabled=disabled)
                sbp = st.slider("Systolic BP", 80, 220, 120, on_change=clear_results, disabled=disabled)
            with c2:
                gender_val = st.selectbox("Gender", ["Female", "Male"], on_change=clear_results, disabled=disabled)
                gender = 1 if gender_val == "Female" else 2
                weight = st.number_input("Weight (kg)", 30, 200, 75, on_change=clear_results, disabled=disabled)
                dbp = st.slider("Diastolic BP", 40, 140, 80, on_change=clear_results, disabled=disabled)

            st.markdown("<hr style='opacity:0.1'>", unsafe_allow_html=True)

            st.markdown(
                "<p style='font-size:0.85rem; opacity:0.6; margin-bottom:5px;'>Lab Markers</p>",
                unsafe_allow_html=True
            )
            lab_c1, lab_c2 = st.columns(2)
            with lab_c1:
                chol = st.select_slider("Cholesterol", [1, 2, 3], value=1, on_change=clear_results, disabled=disabled)
            with lab_c2:
                gluc = st.select_slider("Glucose", [1, 2, 3], value=1, on_change=clear_results, disabled=disabled)

            st.markdown(
                "<p style='font-size:0.85rem; opacity:0.6; margin-top:15px; margin-bottom:5px;'>Lifestyle Factors</p>",
                unsafe_allow_html=True
            )
            life_c1, life_c2, life_c3 = st.columns(3)
            with life_c1:
                active = st.toggle("Active", True, on_change=clear_results, disabled=disabled)
            with life_c2:
                smoke = st.toggle("Smoking", on_change=clear_results, disabled=disabled)
            with life_c3:
                alco = st.toggle("Alcohol Usage", on_change=clear_results, disabled=disabled)

            st.markdown("<br>", unsafe_allow_html=True)

            # **PREDICTION BUTTON - NO RERUN()**
            if st.button("🔬 COMPUTE RISK PROJECTION", use_container_width=True, disabled=st.session_state.prediction_made):
                bmi = weight / ((height / 100) ** 2)
                bp_diff = sbp - dbp
                features = pd.DataFrame(
                    [[gender, sbp, dbp, chol, gluc, int(smoke), int(alco),
                    int(active), age, bmi, bp_diff]],
                    columns=[
                        'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
                        'smoke', 'alco', 'active', 'age_years', 'bmi', 'bp_diff'
                    ]
                )
                prob = model.predict_proba(features)[0][1] * 100
                
                final_results = {
                    "prob": prob,
                    "bmi": bmi,
                    "age": age,
                    "sbp": sbp,
                    "dbp": dbp,
                    "chol": chol
                }
                st.session_state.results = final_results
                st.session_state.static_spider_fig = create_spider_chart(final_results)
                st.session_state.prediction_made = True

    with col_res:
        if st.session_state.prediction_made and st.session_state.results and st.session_state.static_spider_fig:
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

            st.plotly_chart(
                st.session_state.static_spider_fig,
                use_container_width=True,
                config={
                    'displayModeBar': False,    
                    'scrollZoom': False,
                    'staticPlot': True
                }
            )

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

# --- ANALYTICS PAGE ---
elif st.session_state.page == "Analytics":
    # ROW 1: full-width feature sensitivity
    with st.container():
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">🧠 Neural Feature Sensitivity</div>
        """, unsafe_allow_html=True)

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
                height=450,
                margin=dict(l=20, r=20, t=20, b=10),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(showgrid=False)
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": False,
                    "staticPlot": True
                }
            )

        else:
            st.markdown("""
            <div style="height: 350px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 20px;">📊</div>
                    <div style="font-size: 1.2rem; color: #22d3ee; margin-bottom: 10px;">Feature Analysis</div>
                    <p style="opacity: 0.7;">Feature importances are not available for this model.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close glass-card

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 2: two cards side by side
    col_perf, col_conf = st.columns(2, gap="large")

    # MODEL PERFORMANCE CARD
    with col_perf:
        st.markdown("""
        <div class="glass-card" style="height: 520px; padding: 30px;">
            <div class="section-header" style="margin-bottom: 25px;">🎯 Model Performance</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <div class="metric-card">
                    <div style="font-size: 1.8rem; font-weight: 700; color: #22d3ee;">73%</div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px;">ACCURACY</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 1.8rem; font-weight: 700; color: #10b981;">68.1%</div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px;">RECALL</div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div class="metric-card">
                    <div style="font-size: 1.8rem; font-weight: 700; color: #f59e0b;">71.2%</div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px;">F1-SCORE</div>
                </div>
                <div class="metric-card">
                    <div style="font-size: 1.8rem; font-weight: 700; color: #ef4444;">74.7%</div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px;">PRECISION</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # CONFUSION MATRIX CARD
    with col_conf:
        st.markdown("""
        <div class="glass-card" style="height: 520px; padding: 30px;">
            <div class="section-header" style="margin-bottom: 25px;">📈 Confusion Matrix</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="text-align: center; padding: 20px; background: rgba(16, 185, 129, 0.15); border-radius: 16px; border: 2px solid rgba(16, 185, 129, 0.4);">
                    <div style="font-size: 2.2rem; font-weight: 800; color: #10b981; margin-bottom: 8px;">5366</div>
                    <div style="font-size: 0.9rem; opacity: 0.9; font-weight: 600; letter-spacing: 1px;">True Positive</div>
                </div>
                <div style="text-align: center; padding: 20px; background: rgba(239, 68, 68, 0.15); border-radius: 16px; border: 2px solid rgba(239, 68, 68, 0.4);">
                    <div style="font-size: 2.2rem; font-weight: 800; color: #ef4444; margin-bottom: 8px;">1536</div>
                    <div style="font-size: 0.9rem; opacity: 0.9; font-weight: 600; letter-spacing: 1px;">False Positive</div>
                </div>
                <div style="text-align: center; padding: 20px; background: rgba(239, 68, 68, 0.15); border-radius: 16px; border: 2px solid rgba(239, 68, 68, 0.4);">
                    <div style="font-size: 2.2rem; font-weight: 800; color: #ef4444; margin-bottom: 8px;">2126</div>
                    <div style="font-size: 0.9rem; opacity: 0.9; font-weight: 600; letter-spacing: 1px;">False Negative</div>
                </div>
                <div style="text-align: center; padding: 20px; background: rgba(16, 185, 129, 0.15); border-radius: 16px; border: 2px solid rgba(16, 185, 129, 0.4);">
                    <div style="font-size: 2.2rem; font-weight: 800; color: #10b981; margin-bottom: 8px;">4546</div>
                    <div style="font-size: 0.9rem; opacity: 0.9; font-weight: 600; letter-spacing: 1px;">True Negative</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # bottom insight
    st.markdown("""
    <div class="glass-card" style="margin-top: 25px; padding: 25px; background: rgba(129, 140, 248, 0.08); border-radius: 15px; border-left: 4px solid #818cf8;">
        <p style="margin: 0; font-size: 0.95rem; opacity: 0.9; line-height: 1.6;">
            <b>Key Insight:</b> Patients with consistently elevated <b>systolic pressure</b> and <b>BMI</b> above the healthy range show a significantly higher predicted risk, even when laboratory markers like <b>cholesterol</b> and <b>glucose</b> are only mildly abnormal.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- ABOUT PAGE ---
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card" style="height: 420px;">
            <div class="section-header">🧠 Neural Engine</div>
            <p style='opacity:0.8;'>This system utilizes a <b>Random Forest Ensemble</b>. Unlike linear regression, this architecture handles complex feature interactions, such as the relationship between high BMI and increased Systolic Pressure over time.</p>
            <p style='opacity:0.8;'>Arroundly on <b>54296</b> records trained and tested with <b>13574</b> records</p>
            <p style='color:#818cf8; font-weight:600;'>Validation Method: Cross-Validation Recall</p>
            <p style='opacity:0.8;'><b>Feature Handling:</b> No feature scaling required. Tree-based splits operate directly on raw clinical values.</p>
            <p style='opacity:0.75; font-size:13px;'>Outlier filtering applied to BMI and Blood Pressure ranges to improve data reliability and model stability.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card" style="height: 420px; border-left: 5px solid #ef4444;">
            <div class="section-header" style="color:#ef4444;">⚠️ Project Protocol</div>
            <p style='opacity:0.8;'><b>Classification:</b> Beginner Level Academic Project.</p>
            <p style='opacity:0.8;'><b>Accuracy:</b> Clinically simulated for educational demonstration. Not for medical diagnosis.</p>
            <p style='opacity:0.8;'><b>Privacy:</b> Zero-Retention. No data is stored on the server.</p>
            <p style='opacity:0.8;'><b>Model Behavior:</b> Balanced learning observed with minimal overfitting between training and testing data.</p>
            <p style='opacity:0.75; font-size:13px;'>This system is designed to demonstrate machine-learning workflows including feature engineering, hyperparameter tuning, and cross-validation.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; opacity: 0.3; font-size: 0.75rem; margin-top: 50px; line-height: 1.4;'>
    CARDIO.SHIELD • 2026 © Rutvik Bhagiya<br>
    <span style='font-size: 0.65rem;'>
        Academic Research Project • ML-Powered Cardiovascular Risk Assessment
    </span>
</p>
""", unsafe_allow_html=True)
