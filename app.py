import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RxPredict · Drug Recommendation AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #0a1628 70%, #0f0f1e 100%);
    min-height: 100vh;
}

/* Hide default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem 2.5rem; max-width: 1400px; }

/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.8rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 1.2rem;
}

.glass-card-glow {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 1.8rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 1.2rem;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    margin-bottom: 2rem;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0.4rem;
    line-height: 1.1;
}

.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.45);
    font-weight: 400;
    letter-spacing: 0.5px;
}

.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    color: #a78bfa;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    margin-bottom: 1rem;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(167,139,250,0.7);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(167,139,250,0.3), transparent);
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}

.metric-tile {
    flex: 1;
    min-width: 130px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
}

.metric-tile .val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #a78bfa;
    line-height: 1;
}

.metric-tile .lbl {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    margin-top: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Drug result card ── */
.drug-result-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(52,211,153,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.drug-result-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(167,139,250,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.drug-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
}

.drug-emoji {
    font-size: 2.8rem;
    margin-bottom: 0.5rem;
    display: block;
}

.confidence-bar-wrap {
    background: rgba(255,255,255,0.07);
    border-radius: 50px;
    height: 6px;
    margin: 0.6rem 0 0.3rem;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #6366f1, #a78bfa, #34d399);
    transition: width 0.8s ease;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,10,26,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stSidebar"] .css-1d391kg { padding: 1.5rem 1rem; }

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
}

/* ── Select boxes ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
    gap: 4px;
}

[data-testid="stTabs"] button[role="tab"] {
    border-radius: 9px;
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    background: rgba(99,102,241,0.25) !important;
    color: #a78bfa !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.8rem 2rem !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    box-shadow: 0 6px 30px rgba(99,102,241,0.6) !important;
    transform: translateY(-1px) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    margin-bottom: 0.6rem !important;
}

/* ── Input labels ── */
.stSlider label, .stSelectbox label, .stNumberInput label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 1.5rem 0;
}

/* ── Drug info pills ── */
.drug-pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.2rem;
    letter-spacing: 0.5px;
}

/* ── Profile strip ── */
.profile-strip {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    padding: 0.8rem 0;
}

.profile-chip {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.7);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.profile-chip span { color: #a78bfa; font-weight: 600; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─── Load Assets ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("logistic_regression_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("drug200.csv")

model = load_model()
df = load_data()

# Drug metadata
DRUG_INFO = {
    "DrugY": {"color": "#6366f1", "emoji": "🔵", "desc": "Indicated for high Na/K ratio patients", "tag": "First-line", "tag_color": "#6366f1"},
    "drugA": {"color": "#f59e0b", "emoji": "🟡", "desc": "Recommended for high BP, older patients", "tag": "High BP", "tag_color": "#d97706"},
    "drugB": {"color": "#ef4444", "emoji": "🔴", "desc": "Indicated for low BP patients", "tag": "Low BP",  "tag_color": "#dc2626"},
    "drugC": {"color": "#10b981", "emoji": "🟢", "desc": "Suitable for low Na/K ratio & low BP", "tag": "Low Na/K", "tag_color": "#059669"},
    "drugX": {"color": "#8b5cf6", "emoji": "🟣", "desc": "General-purpose, normal BP/cholesterol", "tag": "General", "tag_color": "#7c3aed"},
}

FEATURE_ORDER = ['Age', 'Na_to_K', 'Sex_F', 'Sex_M', 'BP_HIGH', 'BP_LOW',
                 'BP_NORMAL', 'Cholesterol_HIGH', 'Cholesterol_NORMAL']

def build_features(age, sex, bp, chol, na_k):
    row = {'Age': age, 'Sex': sex, 'BP': bp, 'Cholesterol': chol, 'Na_to_K': na_k}
    enc = pd.get_dummies(pd.DataFrame([row]), columns=['Sex', 'BP', 'Cholesterol'])
    for c in FEATURE_ORDER:
        if c not in enc.columns:
            enc[c] = 0
    return enc[FEATURE_ORDER].astype(float)


# ─── Hero Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">✦ AI Drug Recommendation System</div>
    <div class="hero-title">RxPredict</div>
    <div class="hero-sub">Logistic Regression · Drug200 Dataset · Clinical Decision Support</div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar — Patient Input ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700;
             background:linear-gradient(135deg,#a78bfa,#60a5fa); -webkit-background-clip:text;
             -webkit-text-fill-color:transparent; background-clip:text; letter-spacing:-0.5px;">
            Patient Profile
        </div>
        <div style="font-size:0.72rem; color:rgba(255,255,255,0.35); margin-top:0.2rem; letter-spacing:1px; text-transform:uppercase;">
            Enter patient details below
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🧬  Demographics", expanded=True):
        age = st.slider("Age (years)", min_value=15, max_value=75, value=35, step=1)
        sex = st.selectbox("Biological Sex", ["F", "M"], format_func=lambda x: "Female" if x == "F" else "Male")

    with st.expander("🩺  Clinical Measurements", expanded=True):
        bp = st.selectbox("Blood Pressure Level", ["HIGH", "LOW", "NORMAL"],
                          format_func=lambda x: x.capitalize())
        chol = st.selectbox("Cholesterol Level", ["HIGH", "NORMAL"],
                            format_func=lambda x: x.capitalize())
        na_k = st.slider("Na-to-K Ratio", min_value=6.0, max_value=40.0, value=16.0, step=0.1,
                         help="Sodium-to-Potassium ratio in blood")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡  Predict Drug", use_container_width=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem; color:rgba(255,255,255,0.2); text-align:center; line-height:1.6;">
        Model: Logistic Regression (L2)<br>
        Solver: liblinear · C=1.0<br>
        Classes: 5 drugs · Features: 9
    </div>
    """, unsafe_allow_html=True)


# ─── Session state for prediction ───────────────────────────────────────────
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None

if predict_clicked:
    X = build_features(age, sex, bp, chol, na_k)
    drug_pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    st.session_state.pred_result = {
        "drug": drug_pred, "proba": proba,
        "age": age, "sex": sex, "bp": bp, "chol": chol, "na_k": na_k, "X": X
    }


# ─── Top-row KPI tiles ────────────────────────────────────────────────────────
drug_counts = df['Drug'].str.upper().str.replace('DRUG', 'Drug').value_counts()
total = len(df)
st.markdown(f"""
<div class="metric-row">
    <div class="metric-tile">
        <div class="val">{total}</div>
        <div class="lbl">Patients</div>
    </div>
    <div class="metric-tile">
        <div class="val">5</div>
        <div class="lbl">Drug Classes</div>
    </div>
    <div class="metric-tile">
        <div class="val">9</div>
        <div class="lbl">Features</div>
    </div>
    <div class="metric-tile">
        <div class="val">L2</div>
        <div class="lbl">Regularisation</div>
    </div>
    <div class="metric-tile">
        <div class="val">6</div>
        <div class="lbl">Iterations</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Main content area ───────────────────────────────────────────────────────
tabs = st.tabs(["🎯  Prediction", "📊  Data Explorer", "🧠  Model Insights", "📋  Patient History"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if st.session_state.pred_result:
        res = st.session_state.pred_result
        drug = res["drug"]
        proba = res["proba"]
        info = DRUG_INFO.get(drug, DRUG_INFO["DrugY"])
        conf = proba[list(model.classes_).index(drug)] * 100

        col_result, col_breakdown = st.columns([1, 1.4], gap="large")

        with col_result:
            # Patient profile strip
            st.markdown(f"""
            <div class="glass-card">
                <div class="section-label">Patient Profile</div>
                <div class="profile-strip">
                    <div class="profile-chip">Age <span>{res['age']} yrs</span></div>
                    <div class="profile-chip">Sex <span>{'Female' if res['sex']=='F' else 'Male'}</span></div>
                    <div class="profile-chip">BP <span>{res['bp'].capitalize()}</span></div>
                    <div class="profile-chip">Cholesterol <span>{res['chol'].capitalize()}</span></div>
                    <div class="profile-chip">Na/K <span>{res['na_k']:.1f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Drug result
            st.markdown(f"""
            <div class="drug-result-card">
                <span class="drug-emoji">{info['emoji']}</span>
                <div style="font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
                     color:rgba(255,255,255,0.4); margin-bottom:0.3rem;">Recommended Drug</div>
                <div class="drug-name">{drug.upper()}</div>
                <div style="margin:0.8rem auto 0.3rem; color:rgba(255,255,255,0.5); font-size:0.82rem;">
                    {info['desc']}
                </div>
                <div style="margin-top:1rem;">
                    <div style="display:flex;justify-content:space-between;font-size:0.75rem;
                         color:rgba(255,255,255,0.4);margin-bottom:0.3rem;">
                        <span>Confidence</span><span style="color:#a78bfa;font-weight:600;">{conf:.1f}%</span>
                    </div>
                    <div class="confidence-bar-wrap">
                        <div class="confidence-bar-fill" style="width:{min(conf,100):.0f}%"></div>
                    </div>
                </div>
                <div style="margin-top:0.8rem;">
                    <span class="drug-pill" style="background:rgba(99,102,241,0.15);
                          border:1px solid rgba(99,102,241,0.3);color:#a78bfa;">
                        {info['tag']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_breakdown:
            # Probability chart for all drugs
            st.markdown('<div class="section-label">Class Probability Distribution</div>', unsafe_allow_html=True)
            classes = list(model.classes_)
            colors_map = {k: v["color"] for k, v in DRUG_INFO.items()}
            bar_colors = [colors_map.get(c, "#6366f1") for c in classes]

            fig_proba = go.Figure(go.Bar(
                x=classes,
                y=proba * 100,
                marker=dict(
                    color=bar_colors,
                    opacity=0.85,
                    line=dict(color='rgba(255,255,255,0.1)', width=1)
                ),
                text=[f"{p*100:.1f}%" for p in proba],
                textposition='outside',
                textfont=dict(color='rgba(255,255,255,0.7)', size=11, family='Inter'),
            ))
            fig_proba.add_annotation(
                x=drug, y=max(proba)*100,
                text="▲ Predicted", showarrow=False,
                yshift=28, font=dict(color='#a78bfa', size=10, family='Inter'),
            )
            fig_proba.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
                margin=dict(t=30, b=10, l=10, r=10),
                height=220,
                xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                           ticksuffix='%', range=[0, max(proba)*120]),
                showlegend=False,
            )
            st.plotly_chart(fig_proba, use_container_width=True, config={"displayModeBar": False})

            # Decision score gauge
            st.markdown('<div class="section-label">Decision Confidence Gauge</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=conf,
                number=dict(suffix="%", font=dict(size=28, color="#a78bfa", family="Space Grotesk")),
                delta=dict(reference=50, valueformat=".1f",
                           increasing=dict(color="#34d399"),
                           decreasing=dict(color="#ef4444")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1,
                              tickcolor="rgba(255,255,255,0.2)",
                              tickfont=dict(color="rgba(255,255,255,0.4)", size=9)),
                    bar=dict(color="#6366f1", thickness=0.6),
                    bgcolor="rgba(255,255,255,0.04)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 40], color="rgba(239,68,68,0.1)"),
                        dict(range=[40, 70], color="rgba(245,158,11,0.1)"),
                        dict(range=[70, 100], color="rgba(52,211,153,0.1)"),
                    ],
                    threshold=dict(line=dict(color="#34d399", width=2), thickness=0.8, value=70)
                )
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
                margin=dict(t=10, b=10, l=30, r=30),
                height=200,
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        # ── Feature contribution heatmap ──
        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Feature Contribution to Prediction</div>',
                    unsafe_allow_html=True)
        X_val = res["X"].values[0]
        drug_idx = classes.index(drug)
        coef = model.coef_[drug_idx]
        contributions = coef * X_val

        feat_labels = ['Age', 'Na/K Ratio', 'Sex: Female', 'Sex: Male',
                       'BP: High', 'BP: Low', 'BP: Normal', 'Chol: High', 'Chol: Normal']
        feat_df = pd.DataFrame({
            'Feature': feat_labels,
            'Raw Value': X_val,
            'Contribution': contributions,
            'Coefficient': coef,
        }).sort_values('Contribution', key=abs, ascending=True)

        fig_contrib = go.Figure(go.Bar(
            y=feat_df['Feature'],
            x=feat_df['Contribution'],
            orientation='h',
            marker=dict(
                color=feat_df['Contribution'],
                colorscale=[[0, '#ef4444'], [0.5, '#374151'], [1, '#34d399']],
                cmin=-abs(feat_df['Contribution']).max(),
                cmax=abs(feat_df['Contribution']).max(),
                line=dict(width=0),
            ),
            text=[f"{v:+.3f}" for v in feat_df['Contribution']],
            textposition='outside',
            textfont=dict(color='rgba(255,255,255,0.6)', size=10),
        ))
        fig_contrib.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
            margin=dict(t=10, b=10, l=10, r=80),
            height=300,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                       zerolinecolor='rgba(255,255,255,0.15)'),
            yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_contrib, use_container_width=True, config={"displayModeBar": False})

    else:
        # Empty state
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:4rem 2rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">💊</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem; font-weight:600;
                 color:rgba(255,255,255,0.7); margin-bottom:0.5rem;">
                No Prediction Yet
            </div>
            <div style="color:rgba(255,255,255,0.35); font-size:0.9rem; max-width:360px; margin:0 auto;">
                Enter patient details in the sidebar and click <strong style="color:#a78bfa;">Predict Drug</strong> to see the AI recommendation.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="section-label">Drug Class Distribution</div>', unsafe_allow_html=True)
        drug_counts = df['Drug'].value_counts().reset_index()
        drug_counts.columns = ['Drug', 'Count']
        colors_list = [DRUG_INFO.get(d, {}).get("color", "#6366f1") for d in drug_counts['Drug']]

        fig_pie = go.Figure(go.Pie(
            labels=drug_counts['Drug'],
            values=drug_counts['Count'],
            hole=0.55,
            marker=dict(colors=colors_list, line=dict(color='rgba(0,0,0,0.3)', width=2)),
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>%{label}</b><br>%{value} patients<br>%{percent}<extra></extra>',
        ))
        fig_pie.add_annotation(text=f"<b>{total}</b><br><span style='font-size:10'>patients</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color='white', family='Space Grotesk'))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
            margin=dict(t=10, b=10, l=10, r=10), height=280,
            legend=dict(font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown('<div class="section-label">Na/K Ratio by Drug Class</div>', unsafe_allow_html=True)
        fig_box = go.Figure()
        for d, info in DRUG_INFO.items():
            sub = df[df['Drug'].str.upper() == d.upper()]['Na_to_K']
            if len(sub):
                # Convert hex to rgba for fillcolor (Plotly doesn't support 8-digit hex)
                hex_c = info['color'].lstrip('#')
                r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
                fill_rgba = f'rgba({r},{g},{b},0.2)'
                fig_box.add_trace(go.Box(
                    y=sub, name=d, marker_color=info['color'],
                    line=dict(color=info['color']),
                    fillcolor=fill_rgba,
                    boxpoints='outliers',
                ))
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
            margin=dict(t=10, b=10, l=10, r=10), height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                       title=dict(text='Na/K Ratio', font=dict(size=11))),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    # Age distribution + BP heatmap
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown('<div class="section-label">Age Distribution by Drug</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        for d, info in DRUG_INFO.items():
            sub = df[df['Drug'].str.upper() == d.upper()]['Age']
            if len(sub):
                fig_hist.add_trace(go.Histogram(
                    x=sub, name=d, nbinsx=12,
                    marker_color=info['color'],
                    opacity=0.7,
                ))
        fig_hist.update_layout(
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
            margin=dict(t=10, b=10, l=10, r=10), height=260,
            xaxis=dict(title=dict(text='Age', font=dict(size=11)), showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with c4:
        st.markdown('<div class="section-label">Drug vs Blood Pressure Heatmap</div>', unsafe_allow_html=True)
        heat_df = df.groupby(['Drug', 'BP']).size().unstack(fill_value=0)
        fig_heat = go.Figure(go.Heatmap(
            z=heat_df.values,
            x=heat_df.columns.tolist(),
            y=heat_df.index.tolist(),
            colorscale=[[0, 'rgba(99,102,241,0.05)'], [0.5, 'rgba(99,102,241,0.4)'],
                        [1, 'rgba(167,139,250,0.9)']],
            text=heat_df.values,
            texttemplate='%{text}',
            textfont=dict(size=13, color='white'),
            showscale=False,
        ))
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
            margin=dict(t=10, b=10, l=10, r=10), height=260,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # Scatter: Age vs Na_to_K
    st.markdown('<div class="section-label" style="margin-top:0.5rem;">Age vs Na/K Ratio — Patient Scatter</div>',
                unsafe_allow_html=True)
    scatter_df = df.copy()
    scatter_df['color'] = scatter_df['Drug'].map(lambda d: DRUG_INFO.get(d, {}).get("color", "#888"))

    fig_scatter = go.Figure()
    for d, info in DRUG_INFO.items():
        sub = df[df['Drug'].str.upper() == d.upper()]
        if len(sub):
            fig_scatter.add_trace(go.Scatter(
                x=sub['Age'], y=sub['Na_to_K'],
                mode='markers', name=d,
                marker=dict(color=info['color'], size=8, opacity=0.75,
                            line=dict(color='rgba(255,255,255,0.15)', width=1)),
                hovertemplate=f'<b>{d}</b><br>Age: %{{x}}<br>Na/K: %{{y:.2f}}<extra></extra>',
            ))
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
        xaxis=dict(title='Age', showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
        yaxis=dict(title='Na/K Ratio', showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
        legend=dict(font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-label">Model Coefficients — All Drug Classes</div>', unsafe_allow_html=True)
    feat_labels = ['Age', 'Na/K Ratio', 'Sex: Female', 'Sex: Male',
                   'BP: High', 'BP: Low', 'BP: Normal', 'Chol: High', 'Chol: Normal']
    classes = list(model.classes_)

    fig_coef = go.Figure()
    for i, (cls, info) in enumerate(zip(classes, DRUG_INFO.values())):
        fig_coef.add_trace(go.Bar(
            name=cls,
            x=feat_labels,
            y=model.coef_[i],
            marker_color=info['color'],
            opacity=0.8,
        ))
    fig_coef.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
        margin=dict(t=10, b=10, l=10, r=10), height=340,
        xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                   title=dict(text='Coefficient Value', font=dict(size=11)),
                   zerolinecolor='rgba(255,255,255,0.15)'),
        legend=dict(font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig_coef, use_container_width=True, config={"displayModeBar": False})

    c5, c6 = st.columns(2, gap="large")

    with c5:
        st.markdown('<div class="section-label">Coefficient Heatmap</div>', unsafe_allow_html=True)
        fig_cheat = go.Figure(go.Heatmap(
            z=model.coef_,
            x=feat_labels,
            y=classes,
            colorscale=[[0,'#ef4444'],[0.5,'#1e1e3a'],[1,'#34d399']],
            text=np.round(model.coef_, 2),
            texttemplate='%{text}',
            textfont=dict(size=9, color='rgba(255,255,255,0.8)'),
            showscale=True,
            colorbar=dict(tickfont=dict(color='rgba(255,255,255,0.5)', size=9),
                          thickness=10, len=0.8),
        ))
        fig_cheat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.5)', family='Inter'),
            margin=dict(t=10, b=60, l=10, r=10), height=280,
            xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_cheat, use_container_width=True, config={"displayModeBar": False})

    with c6:
        st.markdown('<div class="section-label">Intercepts per Drug Class</div>', unsafe_allow_html=True)
        intercept_colors = [list(DRUG_INFO.values())[i]['color'] for i in range(len(classes))]
        fig_int = go.Figure(go.Bar(
            x=classes,
            y=model.intercept_,
            marker=dict(color=intercept_colors, opacity=0.8,
                        line=dict(color='rgba(255,255,255,0.1)', width=1)),
            text=[f"{v:.3f}" for v in model.intercept_],
            textposition='outside',
            textfont=dict(color='rgba(255,255,255,0.6)', size=11),
        ))
        fig_int.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
            margin=dict(t=10, b=10, l=10, r=10), height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                       title=dict(text='Intercept', font=dict(size=11)),
                       zerolinecolor='rgba(255,255,255,0.15)'),
            showlegend=False,
        )
        st.plotly_chart(fig_int, use_container_width=True, config={"displayModeBar": False})

    # Model parameters table
    st.markdown('<div class="section-label" style="margin-top:0.5rem;">Model Hyperparameters</div>',
                unsafe_allow_html=True)
    params = {
        "Penalty": model.penalty,
        "C (Inverse Regularisation)": model.C,
        "Solver": model.solver,
        "Max Iterations": model.max_iter,
        "Iterations Used": str(model.n_iter_.tolist()),
        "Multi-class Strategy": getattr(model, 'multi_class', 'auto'),
        "Fit Intercept": model.fit_intercept,
        "Classes": ", ".join(model.classes_),
        "Features In": model.n_features_in_,
    }
    st.markdown("""
    <div class="glass-card">
    """, unsafe_allow_html=True)
    for k, v in params.items():
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
             padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="color:rgba(255,255,255,0.45); font-size:0.82rem;">{k}</span>
            <span style="color:#a78bfa; font-weight:600; font-size:0.85rem;
                  font-family:'Space Grotesk',sans-serif;">{v}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PATIENT HISTORY (batch predict on dataset sample)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-label">Batch Prediction — Dataset Sample</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="padding:1rem 1.5rem; margin-bottom:1rem;">
        <div style="color:rgba(255,255,255,0.5); font-size:0.82rem; line-height:1.8;">
            Running the model across the full <strong style="color:#a78bfa;">Drug200</strong> dataset.
            Displays actual vs predicted labels with confidence scores.
        </div>
    </div>
    """, unsafe_allow_html=True)

    sample = df.sample(min(30, len(df)), random_state=42).reset_index(drop=True)
    results = []
    for _, row in sample.iterrows():
        X_r = build_features(row['Age'], row['Sex'], row['BP'], row['Cholesterol'], row['Na_to_K'])
        pred = model.predict(X_r)[0]
        proba_r = model.predict_proba(X_r)[0]
        conf_r = proba_r[list(model.classes_).index(pred)] * 100
        actual_norm = 'Drug' + row['Drug'][4:].upper() if row['Drug'].startswith('drug') else row['Drug']
        match = pred == actual_norm
        results.append({
            "Age": row['Age'],
            "Sex": row['Sex'],
            "BP": row['BP'],
            "Cholesterol": row['Cholesterol'],
            "Na/K": round(row['Na_to_K'], 2),
            "Actual": row['Drug'],
            "Predicted": pred,
            "Confidence": f"{conf_r:.1f}%",
            "✓": "✅" if match else "❌",
        })

    results_df = pd.DataFrame(results)
    acc_rate = (results_df['✓'] == '✅').mean() * 100

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="val">{acc_rate:.0f}%</div>
            <div class="lbl">Sample Accuracy</div>
        </div>
        <div class="metric-tile">
            <div class="val">{(results_df['✓']=='✅').sum()}</div>
            <div class="lbl">Correct</div>
        </div>
        <div class="metric-tile">
            <div class="val">{(results_df['✓']=='❌').sum()}</div>
            <div class="lbl">Incorrect</div>
        </div>
        <div class="metric-tile">
            <div class="val">{len(results_df)}</div>
            <div class="lbl">Sampled</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", format="%s", min_value=0, max_value=100,
            ),
        }
    )