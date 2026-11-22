import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="NEXUS | APT Detection & Investigation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "Saved_Models"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#080d1b; --panel:#101a2d; --cyan:#35e0d0; --purple:#9b7bff; --muted:#9aabc5; }
.stApp { background: radial-gradient(circle at 80% 0%, #18264a 0, #080d1b 42%); color:#edf4ff; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; }
.block-container { padding-top:1.6rem; max-width:1500px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#101a30,#0a1020); border-right:1px solid #253653; }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(23,39,66,.95),rgba(13,23,42,.95)); border:1px solid #29415f; padding:17px 18px; border-radius:16px; }
[data-testid="stMetricLabel"] { color:#9aabc5 !important; }
[data-testid="stMetricValue"] { color:#f3f7ff !important; }
.stButton>button, .stDownloadButton>button { border:1px solid #35e0d0; border-radius:10px; background:linear-gradient(90deg,#126d78,#51419b); color:white; font-weight:700; }
.stTabs [data-baseweb="tab"] { color:#b8c7df; }
.stTabs [aria-selected="true"] { color:#35e0d0 !important; }
div[data-testid="stDataFrame"] { border:1px solid #29415f; border-radius:12px; }
.hero { padding:26px 30px; border:1px solid #29415f; border-radius:22px; background:linear-gradient(115deg,rgba(21,48,75,.92),rgba(41,30,82,.78)); margin-bottom:20px; }
.hero-kicker { color:#35e0d0; font-size:12px; letter-spacing:2px; font-weight:800; }
.hero-title { font-size:34px; font-weight:800; line-height:1.15; margin:8px 0; color:#f5f8ff; }
.hero-sub { color:#b5c4dc; font-size:14px; }
.pill { display:inline-block; padding:5px 10px; border-radius:30px; background:#173c4b; color:#62f0df; font-size:11px; font-weight:700; margin-right:6px; }
.note { padding:12px 15px; border-left:3px solid #9b7bff; background:#151c35; border-radius:8px; color:#c3cde0; font-size:13px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    model_files = {
        "MLP": "MLP.joblib",
        "Logistic Regression": "Logistic_Regression.joblib",
        "XGBoost": "XGBoost.joblib",
        "Random Forest": "Random_Forest.joblib",
        "HistGradientBoosting": "HistGradientBoosting.joblib",
    }
    models = {name: joblib.load(MODEL_DIR / filename) for name, filename in model_files.items()}
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    return preprocessor, models, metadata

def clean_text(series):
    s = series.fillna("").astype(str).str.strip()
    return s.mask(s.str.lower().isin(["nan", "none", "null", "<na>"]), "")

def combine_clean_columns(data, columns):
    available = [c for c in columns if c in data.columns]
    if not available:
        return pd.Series("", index=data.index)
    temp = pd.DataFrame({c: clean_text(data[c]) for c in available})
    temp = temp.replace("", np.nan)
    return temp.bfill(axis=1).iloc[:, 0].fillna("")

def build_model_features(raw):
    """Reproduce the notebook's 21 model-input features from raw Windows/Wazuh CSV columns."""
    f = pd.DataFrame(index=raw.index)
    def col(name, default=""):
        return raw[name] if name in raw.columns else pd.Series(default, index=raw.index)

    f["Event_ID"] = clean_text(col("_source.data.win.system.eventID")).replace("", "Unknown")
    f["Provider"] = clean_text(col("_source.data.win.system.providerName")).replace("", "Unknown")
    f["Channel"] = clean_text(col("_source.data.win.system.channel")).replace("", "Unknown")
    f["Process"] = combine_clean_columns(raw, [
        "_source.data.win.eventdata.image",
        "_source.data.win.eventdata.originalFileName",
        "_source.data.win.eventdata.processName",
        "_source.data.process.name",
    ])
    f["Parent_Process"] = combine_clean_columns(raw, ["_source.data.win.eventdata.parentImage"])
    f["Integrity"] = clean_text(col("_source.data.win.eventdata.integrityLevel")).replace("", "Unknown")
    f["Logon_Type"] = clean_text(col("_source.data.win.eventdata.logonType")).replace("", "Unknown")
    f["Protocol"] = clean_text(col("_source.data.win.eventdata.protocol")).replace("", "Unknown")
    f["Decoder"] = clean_text(col("_source.decoder.name")).replace("", "Unknown")
    f["Command"] = combine_clean_columns(raw, [
        "_source.data.win.eventdata.commandLine",
        "_source.data.win.eventdata.parentCommandLine",
        "_source.data.process.cmd",
        "_source.data.command",
    ])
    f["Destination_IP"] = clean_text(col("_source.data.win.eventdata.destinationIp")).replace("", "Unknown")
    f["Source_IP"] = clean_text(col("_source.data.win.eventdata.sourceIp")).replace("", "Unknown")
    f["Destination_Port"] = pd.to_numeric(col("_source.data.win.eventdata.destinationPort"), errors="coerce").fillna(-1)
    f["Source_Port"] = pd.to_numeric(col("_source.data.win.eventdata.sourcePort"), errors="coerce").fillna(-1)
    f["Target_File"] = combine_clean_columns(raw, [
        "_source.data.win.eventdata.targetFilename", "_source.syscheck.path"
    ])
    f["Registry_Object"] = clean_text(col("_source.data.win.eventdata.targetObject"))
    f["DNS_Query"] = clean_text(col("_source.data.win.eventdata.queryName"))
    f["Event_Type"] = clean_text(col("_source.data.win.eventdata.eventType")).replace("", "Unknown")

    ts = pd.to_datetime(col("_source.@timestamp"), format="%b %d, %Y @ %H:%M:%S.%f", errors="coerce", utc=True)
    # Accept already ISO-formatted timestamps too.
    missing_ts = ts.isna()
    if missing_ts.any():
        ts.loc[missing_ts] = pd.to_datetime(col("_source.@timestamp").loc[missing_ts], errors="coerce", utc=True)
    f["Timestamp"] = ts
    f["Hour"] = ts.dt.hour.fillna(-1)
    f["Day_of_Week"] = ts.dt.dayofweek.fillna(-1)
    f["Command_Length"] = f["Command"].str.len()
    f["Has_Command"] = (f["Command"].str.len() > 0).astype(int)
    f["Has_Network_Activity"] = ((f["Destination_IP"] != "Unknown") | (f["Source_IP"] != "Unknown")).astype(int)
    f["Has_File_Activity"] = (f["Target_File"].str.len() > 0).astype(int)
    f["Has_Registry_Activity"] = (f["Registry_Object"].str.len() > 0).astype(int)
    f["Has_DNS_Activity"] = (f["DNS_Query"].str.len() > 0).astype(int)

    terms = [
        "powershell", "encodedcommand", "frombase64string",
        "downloadstring", "invoke-expression", " -enc ",
        "cmd.exe", "wscript", "cscript", " rundll32",
        "regsvr32", "schtasks", "wmic", "bitsadmin"
    ]
    command_lower = f["Command"].str.lower()
    f["Suspicious_Term_Count"] = sum(
        command_lower.str.contains(term, regex=False, na=False).astype(int)
        for term in terms
    )

    model_input = pd.DataFrame(index=f.index)
    categorical = [
        "Event_ID", "Provider", "Channel", "Integrity",
        "Logon_Type", "Protocol", "Decoder", "Event_Type"
    ]
    for c in categorical:
        model_input[c] = clean_text(f[c]).replace("", "Unknown")
    model_input["Process_Name"] = (
        f["Process"].str.replace("/", "\\", regex=False)
        .str.split("\\", regex=False).str[-1].str.lower().replace("", "Unknown")
    )
    model_input["Parent_Process_Name"] = (
        f["Parent_Process"].str.replace("/", "\\", regex=False)
        .str.split("\\", regex=False).str[-1].str.lower().replace("", "Unknown")
    )
    numeric = [
        "Hour", "Day_of_Week", "Command_Length", "Destination_Port",
        "Source_Port", "Has_Command", "Has_Network_Activity",
        "Has_File_Activity", "Has_Registry_Activity",
        "Has_DNS_Activity", "Suspicious_Term_Count"
    ]
    for c in numeric:
        model_input[c] = pd.to_numeric(f[c], errors="coerce").fillna(-1)
    return f, model_input

def risk_label(score):
    if score >= 0.80:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"

st.markdown("""
<div class="hero">
  <div class="hero-kicker">SECURITY OPERATIONS • MACHINE LEARNING</div>
  <div class="hero-title">NEXUS <span style="color:#35e0d0">/</span> APT DETECTION & INVESTIGATION</div>
  <div class="hero-sub">AI-assisted Windows telemetry triage, behavioral signals and incident exploration.</div>
  <div style="margin-top:14px"><span class="pill">5 ML MODELS</span><span class="pill">WINDOWS TELEMETRY</span><span class="pill">SOC WORKFLOW</span></div>
</div>
""", unsafe_allow_html=True)

try:
    preprocessor, models, metadata = load_assets()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

with st.sidebar:
    st.markdown("## 🛡️ NEXUS")
    st.caption("APT Detection Console")
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview", "Detection Lab", "Investigation", "Model Intelligence", "About & Limitations"])
    st.markdown("---")
    st.caption("MODEL STATUS")
    st.success("5 trained models loaded")
    st.caption("Dataset: Windows-APT 2025 v4")
    st.caption("Inference: uploaded CSV telemetry")

if "results" not in st.session_state:
    st.session_state.results = None
if "raw_preview" not in st.session_state:
    st.session_state.raw_preview = None

if page == "Overview":
    st.subheader("Command Center")
    st.markdown("Upload a Windows telemetry CSV in **Detection Lab** to start a new analysis.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Available ML models", "05", "Ready")
    c2.metric("Input features", "21", "10 categorical • 11 numeric")
    c3.metric("Training records", f"{metadata['training_records']:,}")
    c4.metric("Held-out records", f"{metadata['testing_records']:,}")
    st.markdown("### Operational workflow")
    cols = st.columns(4)
    cards = [
        ("01","Ingest","Upload event telemetry","📥"),
        ("02","Detect","Run a selected ML model","🧠"),
        ("03","Prioritize","Review score-based triage","🚨"),
        ("04","Investigate","Explore event context and timelines","🔎"),
    ]
    for col,(n,title,desc,ico) in zip(cols,cards):
        with col:
            st.markdown(f"""<div style="background:#101a2d;border:1px solid #29415f;border-radius:15px;padding:18px;min-height:150px">
            <div style="font-size:25px">{ico}</div><div style="color:#35e0d0;font-weight:800;font-size:12px">{n}</div>
            <div style="font-size:18px;font-weight:700;margin:5px 0">{title}</div>
            <div style="color:#9aabc5;font-size:12px">{desc}</div></div>""", unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="note">Prototype note: model scores represent confidence for the project’s weak-label classification target. They are not calibrated probabilities of a real-world APT compromise.</div>', unsafe_allow_html=True)

elif page == "Detection Lab":
    st.subheader("Detection Lab")
    st.write("Upload a CSV exported from the Windows-APT/Wazuh telemetry schema used during training.")
    uploaded = st.file_uploader("Drop telemetry CSV here", type=["csv"], help="Upload one source log CSV. Large files may take a few seconds.")
    left,right = st.columns([2,1])
    with left:
        selected_model = st.selectbox("Detection model", list(models.keys()), index=0)
    with right:
        threshold = st.slider("Alert threshold", 0.10, 0.95, 0.50, 0.05)
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, low_memory=False)
            st.caption(f"Loaded {len(raw):,} records • {len(raw.columns)} raw columns")
            with st.expander("Preview uploaded telemetry"):
                st.dataframe(raw.head(8), use_container_width=True)
            if st.button("⚡ RUN DETECTION", type="primary", use_container_width=True):
                with st.spinner("Engineering features and scoring telemetry..."):
                    engineered, model_input = build_model_features(raw)
                    transformed = preprocessor.transform(model_input)
                    model = models[selected_model]

                    # Score in batches to keep memory usage predictable.
                    # HistGradientBoosting requires dense input; the other
                    # models can consume the preprocessor's sparse matrix.
                    batch_size = 5000
                    score_batches = []

                    for start in range(0, transformed.shape[0], batch_size):
                        end = min(start + batch_size, transformed.shape[0])
                        batch = transformed[start:end]

                        if selected_model == "HistGradientBoosting" and hasattr(batch, "toarray"):
                            batch = batch.toarray()

                        batch_scores = model.predict_proba(batch)[:, 1]
                        score_batches.append(batch_scores)

                    scores = np.concatenate(score_batches) if score_batches else np.array([], dtype=float)
                    predictions = (scores >= threshold).astype(int)
                    output = pd.DataFrame({
                        "Timestamp": engineered["Timestamp"],
                        "Event_ID": engineered["Event_ID"],
                        "Process": engineered["Process"],
                        "Parent_Process": engineered["Parent_Process"],
                        "User": combine_clean_columns(raw, [
                            "_source.data.win.eventdata.user",
                            "_source.data.win.eventdata.subjectUserName",
                            "_source.data.win.eventdata.targetUserName"
                        ]),
                        "Command": engineered["Command"],
                        "Destination_IP": engineered["Destination_IP"],
                        "Destination_Port": engineered["Destination_Port"],
                        "Detection_Score": scores,
                        "Prediction": predictions,
                    })
                    output["Priority"] = output["Detection_Score"].apply(risk_label)
                    output["Priority"] = np.where(output["Prediction"] == 1, output["Priority"], "Below threshold")
                    st.session_state.results = output
                    st.session_state.raw_preview = raw.head(1000)
                    st.session_state.model_name = selected_model
                    st.session_state.threshold = threshold
                st.success(f"Analysis complete: {len(output):,} records scored.")
        except Exception as e:
            st.error(f"Could not process this CSV: {e}")
            st.info("Check that the file contains the raw Windows/Wazuh event columns used in the training dataset.")
    if st.session_state.results is not None:
        out = st.session_state.results
        positives = int(out["Prediction"].sum())
        a,b,c,d = st.columns(4)
        a.metric("Records analyzed", f"{len(out):,}")
        b.metric("Above threshold", f"{positives:,}")
        c.metric("High priority", f"{int((out['Priority']=='High').sum()):,}")
        d.metric("Selected model", st.session_state.model_name)
        fig = px.histogram(out, x="Detection_Score", color="Prediction", nbins=35,
                           color_discrete_map={0:"#65748b",1:"#35e0d0"},
                           title="Detection score distribution")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Alert queue")
        show = out[out["Prediction"]==1].sort_values("Detection_Score",ascending=False)
        st.dataframe(show.head(500), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download scored events CSV", out.to_csv(index=False).encode("utf-8"), "nexus_detection_results.csv", "text/csv")
        st.download_button("⬇️ Download alert queue CSV", show.to_csv(index=False).encode("utf-8"), "nexus_alert_queue.csv", "text/csv")

elif page == "Investigation":
    st.subheader("Incident Investigation")
    out = st.session_state.results
    if out is None:
        st.info("Run a detection first in Detection Lab. Investigation tools will appear here.")
    else:
        alerts = out[out["Prediction"]==1].copy()
        if alerts.empty:
            st.warning("No records crossed the selected threshold. Lower the threshold to inspect more events.")
        else:
            st.markdown("### Alert explorer")
            priorities = st.multiselect("Priority filter", ["High","Medium","Low"], default=["High","Medium","Low"])
            filtered = alerts[alerts["Priority"].isin(priorities)]
            search = st.text_input("Search process, command, user or destination")
            if search:
                mask = filtered.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
                filtered = filtered[mask]
            st.dataframe(filtered.sort_values("Detection_Score",ascending=False), use_container_width=True, hide_index=True)
            if not filtered.empty:
                idx = st.selectbox("Select event for details", filtered.index.tolist(), format_func=lambda i: f"{filtered.loc[i,'Priority']} • {filtered.loc[i,'Process']} • score {filtered.loc[i,'Detection_Score']:.3f}")
                row = filtered.loc[idx]
                st.markdown("### Event details")
                st.json({k:(None if pd.isna(v) else str(v)) for k,v in row.to_dict().items()})
                if pd.notna(row["Timestamp"]):
                    st.caption(f"Event timestamp: {row['Timestamp']}")

elif page == "Model Intelligence":
    st.subheader("Model Intelligence")
    st.markdown("Held-out evaluation results from the project notebook.")
    eval_path = ROOT / "project_results" / "model_evaluation_results.csv"
    if eval_path.exists():
        ev = pd.read_csv(eval_path)
        st.dataframe(ev, use_container_width=True, hide_index=True)
        if {"Model","F1-Score"}.issubset(ev.columns):
            fig = px.bar(ev, x="Model", y="F1-Score", color="Model", title="Held-out F1-score comparison")
            fig.update_layout(template="plotly_dark", showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Evaluation CSV not found in project_results.")
    st.markdown("### Feature importance")
    imp_path = ROOT / "project_results" / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        st.dataframe(imp.head(20), use_container_width=True, hide_index=True)
    st.markdown('<div class="note">The project’s ablation and context-vs-behavior experiments showed strong dependence on telemetry-context fields. Treat very high held-out scores cautiously.</div>', unsafe_allow_html=True)

elif page == "About & Limitations":
    st.subheader("System Notes & Responsible Interpretation")
    st.markdown("""
    **Project:** AI-Powered APT Detection and Investigation System  
    **Training dataset:** Windows-APT 2025 Version 4  
    **Model inputs:** 21 telemetry-derived features (10 categorical, 11 numeric)  
    **Models:** Logistic Regression, Random Forest, XGBoost, HistGradientBoosting and MLP.

    **Important limitations**
    - The binary target is a weak-label proxy derived from dataset-provided technique attribution fields, not independently verified ground truth.
    - Detection scores are not calibrated probabilities of compromise or threat severity.
    - Dataset-provided MITRE technique mappings are not independently inferred by the classifier.
    - A command being present in logs does not prove that it executed successfully.
    - This application is a research/demo prototype, not a production SOC replacement.
    """)
    st.markdown("### Expected raw input fields")
    st.code("""
_source.data.win.system.eventID
_source.data.win.system.providerName
_source.data.win.system.channel
_source.data.win.eventdata.image
_source.data.win.eventdata.parentImage
_source.data.win.eventdata.commandLine
_source.data.win.eventdata.user
_source.data.win.eventdata.destinationIp
_source.data.win.eventdata.sourceIp
_source.data.win.eventdata.destinationPort
_source.data.win.eventdata.sourcePort
_source.@timestamp
    """, language="text")
    st.caption("Missing optional fields are filled with safe defaults; the application still requires a compatible event schema for meaningful inference.")
