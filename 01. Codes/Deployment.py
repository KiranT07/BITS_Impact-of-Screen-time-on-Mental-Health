import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="Mental Health Score Prediction", page_icon="🧠", layout="wide")

def set_background(image_file):
    if not os.path.exists(image_file):
        return
    with open(image_file, "rb") as f:
        data = f.read()
    b64_img = base64.b64encode(data).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64_img}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .mode-btn {{
        display: inline-flex; align-items: center;
        padding: 10px 20px;
        border-radius: 20px;
        background-color: #2471E3;
        color: white;
        font-weight: 600;
        font-size: 1.2em;
        margin: 5px 15px 10px 0;
        cursor: pointer;
        border: none;
        transition: background-color 0.3s ease;
    }}
    .mode-btn:hover, .mode-btn:focus {{
        background-color: #0e59c6;
        outline: none;
    }}
    .mode-btn.activated {{
        background-color: #0e59c6;
    }}
    .emoji {{
        font-size: 1.4em;
        margin-right: 8px;
    }}
    .predict-btn {{
        display: flex; align-items: center; justify-content: center;
        background-color: #2471E3;
        color: white;
        font-weight: 600;
        font-size: 1.2em;
        padding: 12px 36px;
        border-radius: 32px;
        cursor: pointer;
        border: none;
        margin: 20px 0;
        transition: background-color 0.3s ease;
        width: fit-content;
    }}
    .predict-btn:hover, .predict-btn:focus {{
        background-color: #0e59c6;
        outline: none;
    }}
    .predict-btn .icon {{
        margin-right: 10px;
        font-size: 1.5em;
    }}
    .custom-score {{
        margin-top: 15px;
        padding: 15px;
        font-size: 1.8em;
        font-weight: 700;
        text-align: center;
        border-radius: 12px;
        width: 70%;
        margin-left: auto;
        margin-right: auto;
    }}
    .score-red {{
        color: #dc3545;
        border: 2px solid #dc3545;
        background-color: #ffe9eccc;
    }}
    .score-orange {{
        color: #fd7e14;
        border: 2px solid #fd7e14;
        background-color: #fff4e5cc;
    }}
    .score-green {{
        color: #198754;
        border: 2px solid #198754;
        background-color: #e9fcd7cc;
    }}
    .disclaimer-box {{
        background-color: #fff3cd;
        border-left: 6px solid #ffeeba;
        padding: 17px;
        border-radius: 9px;
        margin-bottom: 22px;
        color: #856404;
        font-weight: 600;
        font-size: 1.1em;
    }}
    .footer-note {{
        margin-top: 2em;
        margin-bottom: 2em;
        color: #56636a;
        font-size: 1em;
        text-align: left;
        opacity: 0.85;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)

set_background("back1.png")

st.title("🧠 Mental Health Score Prediction")
st.markdown("Welcome to the AI Mental Wellness Assessment Portal — estimate how your lifestyle impacts mental health using advanced machine learning models.")

# Load model artifacts
try:
    ensemble = joblib.load("champion_voting_ensemble.joblib")
    scaler = joblib.load("deployment_scaler.joblib")
    features = joblib.load("deployment_feature_names.joblib")
except Exception as e:
    st.error(f"Failed loading model files: {e}")
    st.stop()

_gender_map = {'Male': 0, 'Female': 1, 'Others': 2}
_location_map = {'Urban': 0, 'Semiurban': 1, 'Rural': 2, 'Others': 3}

def preprocess_input(df):
    df = df.copy()
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(_gender_map)
    if 'location_type' in df.columns:
        df['location_type'] = df['location_type'].map(_location_map)
    for c in ['uses_wellness_apps', 'eats_healthy']:
        if c in df.columns:
            df[c] = df[c].astype(int)
    for col in features:
        if col in df.columns:
            if df[col].dtype.kind in "biufc":
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    df_filled = df.reindex(columns=features)
    scaled = scaler.transform(df_filled)
    scaled_df = pd.DataFrame(scaled, columns=features)
    return scaled_df, df_filled

# Input Mode Buttons
col1, col2 = st.columns(2)
if "input_mode" not in st.session_state:
    st.session_state.input_mode = None

with col1:
    is_manual = st.button("🧍 Manual Input", key="manual_input_button")
with col2:
    is_batch = st.button("📂 Batch Prediction", key="batch_prediction_button")

if is_manual:
    st.session_state.input_mode = "manual"
if is_batch:
    st.session_state.input_mode = "batch"

input_mode = st.session_state.input_mode

# Manual Input Form
if input_mode == "manual":
    with st.form("manual_form"):
        colL, colR = st.columns(2)
        inp = {}
        with colL:
            inp['age'] = st.number_input("Age", 10, 100, 30)
            inp['gender'] = st.selectbox("Gender", ["Male", "Female", "Others"])
            inp['location_type'] = st.selectbox("Location Type", ["Urban", "Semiurban", "Rural", "Others"])
            inp['uses_wellness_apps'] = 1 if st.selectbox("Uses Wellness Apps", ["Yes", "No"]) == "Yes" else 0
            inp['eats_healthy'] = 1 if st.selectbox("Eats Healthy", ["Yes", "No"]) == "Yes" else 0
            inp['sleep_quality'] = st.slider("Sleep Quality (1 to 10)", 1, 10, 5)
            inp['stress_level'] = st.slider("Stress Level (1 to 10)", 1, 10, 5)
            inp['social_media_hours'] = st.number_input("Social Media Hours", 0.0, 24.0, 1.0, 0.01)

        with colR:
            inp['daily_screen_time_hours'] = st.number_input("Daily Screen Time (hrs)", 0.0, 24.0, 5.0, 0.01)
            inp['phone_usage_hours'] = st.number_input("Phone Usage (hrs)", 0.0, 24.0, 2.0, 0.01)
            inp['laptop_usage_hours'] = st.number_input("Laptop Usage (hrs)", 0.0, 24.0, 2.0, 0.01)
            inp['tablet_usage_hours'] = st.number_input("Tablet Usage (hrs)", 0.0, 24.0, 1.0, 0.01)
            inp['tv_usage_hours'] = st.number_input("TV Usage (hrs)", 0.0, 24.0, 1.0, 0.01)
            inp['work_related_hours'] = st.number_input("Work Related Hours", 0.0, 24.0, 3.0, 0.01)
            inp['entertainment_hours'] = st.number_input("Entertainment Hours", 0.0, 24.0, 2.0, 0.01)
            inp['gaming_hours'] = st.number_input("Gaming Hours", 0.0, 24.0, 1.0, 0.01)
            inp['sleep_duration_hours'] = st.number_input("Sleep Duration (hrs)", 0.0, 24.0, 7.0, 0.01)
            inp['mood_rating'] = st.slider("Mood Rating (1 to 10)", 1, 10, 5)
            inp['physical_activity_hours_per_week'] = st.number_input("Physical Activity Hours/Week", 0.0, 50.0, 5.0, 0.01)
            inp['caffeine_intake_mg_per_day'] = st.number_input("Caffeine Intake (mg/day)", 0.0, 500.0, 100.0, 0.01)
            inp['weekly_anxiety_score'] = st.number_input("Weekly Anxiety Score", 0.0, 50.0, 10.0, 0.01)
            inp['weekly_depression_score'] = st.number_input("Weekly Depression Score", 0.0, 50.0, 10.0, 0.01)
            inp['mindfulness_minutes_per_day'] = st.number_input("Mindfulness (min/day)", 0.0, 1440.0, 10.0, 0.01)

        submit = st.form_submit_button("🚀 PREDICT SCORE")
        if submit:
            try:
                X_scaled, _ = preprocess_input(pd.DataFrame([inp]))
                pred = float(ensemble.predict(X_scaled)[0])
                color = "score-red" if pred < 50 else "score-orange" if pred < 70 else "score-green"
                st.markdown(f'<div class="custom-score {color}"><span class="icon-rocket"></span>Predicted Mental Health Score: {pred:.2f}</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in prediction: {e}")

# Batch Prediction Section
elif input_mode == "batch":
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv","xlsx"])
    if uploaded is not None:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.dataframe(df.head())
        predict = st.button("📂 PREDICT BATCH SCORES")
        if predict:
            try:
                X_scaled, df_filled = preprocess_input(df)
                preds = ensemble.predict(X_scaled)
                df_show = df.copy()
                df_show["Predicted_Score"] = preds
                if "mental_health_score" in df_show.columns:
                    actual = df_show["mental_health_score"].values
                    predicted = df_show["Predicted_Score"].values
                    mape = mean_absolute_percentage_error(actual, predicted)
                    smape = (100 / len(actual)) * np.sum(
                        2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))
                    )
                    rmse = np.sqrt(mean_squared_error(actual, predicted))
                    try:
                        cv = cross_val_score(
                            ensemble,
                            X_scaled,
                            df_show["mental_health_score"],
                            cv=5,
                            scoring=make_scorer(mean_squared_error),
                        ).mean() if len(df_show) > 5 else np.nan
                    except:
                        cv = np.nan
                else:
                    actual = predicted = mape = smape = rmse = cv = np.nan
                cols_to_show = (
                    ["user_id", "mental_health_score", "Predicted_Score"]
                    if "user_id" in df_show.columns
                    else ["mental_health_score", "Predicted_Score"]
                )
                st.markdown("#### Batch Prediction Results")
                st.dataframe(df_show[cols_to_show], width="stretch")
                st.markdown(
                    f"""
                <div style='background:#F6FCFF;border-radius:7px;padding:10px 20px; display:inline-block'>
                    <b>MAPE:</b> {mape:.3f} &nbsp;
                    <b>SMAPE:</b> {smape:.3f} &nbsp;
                    <b>RMSE:</b> {rmse:.3f} &nbsp;
                    <b>CV Score:</b> {cv:.3f}
                </div>
                """,
                    unsafe_allow_html=True,
                )
                if "mental_health_score" in df_show.columns:
                    colA, colB = st.columns(2)
                    # Scatter plot (bright scarlet points, indigo regression line, black axis)
                    fig1 = go.Figure()
                    fig1.add_trace(
                        go.Scatter(
                            x=df_show["mental_health_score"],
                            y=df_show["Predicted_Score"],
                            mode="markers",
                            marker=dict(color="#ff2400"),
                            name="Predictions",
                        )
                    )
                    X = np.array(df_show["mental_health_score"]).reshape(-1, 1)
                    y = np.array(df_show["Predicted_Score"])
                    if len(X) > 1:
                        linreg = LinearRegression().fit(X, y)
                        line_x = np.linspace(min(X), max(X), 100)
                        pred_line = linreg.predict(line_x)
                        fig1.add_trace(
                            go.Scatter(
                                x=line_x.flatten(),
                                y=pred_line,
                                mode="lines",
                                line=dict(color="#4b0082", dash="dash", width=3),
                                name="Regression Line",
                            )
                        )
                    fig1.update_layout(
                        title="Actual vs Predicted Score",
                        xaxis_title="mental_health_score",
                        yaxis_title="Predicted_Score",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="black"),
                        width=700,
                        height=400,
                    )
                    colA.plotly_chart(fig1, use_container_width=True)

                    # Horizontal error bin bar (crystal teal)
                    errors = np.abs(df_show["mental_health_score"] - df_show["Predicted_Score"])
                    bins = [0, 5, 10, 20, 30, 100]
                    labels = ["0-5", "5-10", "10-20", "20-30", "30+"]
                    error_bins = pd.cut(errors, bins, labels=labels, include_lowest=True)
                    error_counts = error_bins.value_counts(sort=False)
                    fig2 = go.Figure(
                        go.Bar(
                            x=error_counts.values,
                            y=labels,
                            orientation="h",
                            marker_color="#00CED1",
                            opacity=0.77,
                        )
                    )
                    fig2.update_layout(
                        title="Error Bin Distribution (Horizontal)",
                        xaxis_title="Count",
                        yaxis_title="Absolute Error Bin",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="black"),
                        width=460,
                        height=400,
                    )
                    colB.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Batch prediction error: {e}")

# Footer and other text
st.markdown(
    """
---
### 📌 Project Conclusion and Recommendations
Our capstone highlights that digital and lifestyle choices directly and interactively influence mental health outcomes. Through comprehensive feature engineering—spanning device use, sleep, activity, environmental context, and wellbeing strategies—our models demonstrate:
- Prolonged, unbalanced screen time, especially when coupled with poor sleep or stressful routines, raises mental health risk factors.
- Positive wellness app usage, regular mindfulness, healthy eating, and physical activity are associated with higher predicted wellness scores.
- Urban and semi-urban contexts show moderate risk, but personalized habits within those environments are stronger predictors than setting alone.

**Recommendations:**
- Practice regular digital detox and structured device breaks.
- Aim for consistent sleep routines and strong sleep quality.
- Incorporate mindfulness and physical activity into daily schedules.
- Leverage supportive digital tools, but set healthy boundaries for tech use.
- Periodic self-assessment and professional help are encouraged whenever stress, anxiety, or digital fatigue feel unmanageable.

Ongoing development will integrate real-world wearable and longitudinal behavioral data to refine and personalize predictions further.
"""
)

st.markdown(
    """
<div class="disclaimer-box">
⚠️ <b>Disclaimer</b><br>
This tool is for educational and experimental purposes only. Predictions use models trained on synthetic data and should never be used for any real medical decisions. If you’re in crisis, seek professional help immediately.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="footer-note">
Project by:<br>
<strong>BITS Pilani – Group 04, Cohort 12 | AIML Capstone Project</strong>
</div>
""",
    unsafe_allow_html=True,
)
