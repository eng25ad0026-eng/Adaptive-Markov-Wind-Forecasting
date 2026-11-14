import json
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Adaptive Markov Wind Forecasting",
    page_icon="🌬️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# --------------------------------------------------
# LOAD MODEL OUTPUT FILES
# --------------------------------------------------

@st.cache_data
def load_data():
    test_data = pd.read_csv(DATA_DIR / "test_data.csv")
    forecast = pd.read_csv(
        DATA_DIR / "adaptive_forecast_results.csv"
    )

    with open(DATA_DIR / "model_metrics.json", "r") as file:
        metrics = json.load(file)

    return test_data, forecast, metrics


st.title("🌬️ Adaptive Markov Wind Energy Forecasting")
st.caption(
    "Probabilistic wind-state forecasting under uncertainty "
    "and changing conditions"
)

try:
    test_data, forecast, metrics = load_data()

except Exception as error:
    st.error("Unable to load the model output files.")
    st.write("Check that all three files are inside the data folder.")
    st.code(str(error))
    st.stop()

# --------------------------------------------------
# CHECK DATA
# --------------------------------------------------

if len(test_data) != len(forecast):
    st.error("Test data and forecast records have different lengths.")
    st.stop()

required_columns = [
    "P_Low", "P_Medium", "P_High",
    "Predicted_State", "Actual_Next_State", "Entropy"
]

if not all(col in forecast.columns for col in required_columns):
    st.error("Some required forecast columns are missing.")
    st.stop()

# Combine test data and forecast results
combined = pd.concat(
    [
        test_data.reset_index(drop=True),
        forecast.reset_index(drop=True)
    ],
    axis=1
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Choose a section",
    [
        "Dashboard",
        "Forecast Results",
        "Uncertainty Analysis",
        "Model Performance"
    ]
)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Model Dashboard")

    st.info(
        "This dashboard displays the saved outputs "
        "of the trained Adaptive Markov forecasting model."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training Records",
        f'{metrics["training_records"]:,}'
    )

    col2.metric(
        "Testing Records",
        f'{metrics["testing_records"]:,}'
    )

    col3.metric(
        "Study Latitude",
        metrics["study_latitude"]
    )

    st.subheader("Model Accuracy")

    col1, col2 = st.columns(2)

    col1.metric(
        "Baseline Accuracy",
        f'{metrics["baseline_accuracy"] * 100:.2f}%'
    )

    col2.metric(
        "Adaptive Accuracy",
        f'{metrics["adaptive_accuracy"] * 100:.2f}%'
    )

    st.subheader("Wind Speed Data")

    st.line_chart(
        test_data["Wind_Speed"],
        x_label="Test Record",
        y_label="Wind Speed"
    )

    st.subheader("Wind State Distribution")

    st.bar_chart(test_data["Wind_State"].value_counts())

# --------------------------------------------------
# FORECAST RESULTS
# --------------------------------------------------

elif page == "Forecast Results":

    st.header("Adaptive Forecast Results")

    selected_state = st.selectbox(
        "Filter by predicted wind state",
        ["All", "Low", "Medium", "High"]
    )

    if selected_state == "All":
        filtered = combined
    else:
        filtered = combined[
            combined["Predicted_State"] == selected_state
        ]

    st.write("Forecast records:", len(filtered))

    st.dataframe(
        filtered,
        use_container_width=True
    )

    st.subheader("Individual Forecast Probabilities")

    record_number = st.number_input(
        "Select forecast record",
        min_value=1,
        max_value=len(combined),
        value=1
    )

    selected = forecast.iloc[int(record_number) - 1]

    probability_data = pd.DataFrame(
        {"Probability": [
            selected["P_Low"],
            selected["P_Medium"],
            selected["P_High"]
        ]},
        index=["Low", "Medium", "High"]
    )

    st.bar_chart(probability_data)

    st.write("Predicted state:", selected["Predicted_State"])
    st.write("Actual next state:", selected["Actual_Next_State"])

# --------------------------------------------------
# UNCERTAINTY ANALYSIS
# --------------------------------------------------

elif page == "Uncertainty Analysis":

    st.header("Forecast Uncertainty Analysis")

    st.metric(
        "Mean Forecast Entropy",
        f'{metrics["mean_forecast_entropy"]:.5f}'
    )

    st.write(
        "Entropy measures how spread out the forecast "
        "probabilities are. Higher entropy indicates greater "
        "uncertainty."
    )

    st.subheader("Entropy Across Forecast Records")

    st.line_chart(forecast["Entropy"])

    st.subheader("Entropy Distribution")

    st.bar_chart(
        forecast["Entropy"].value_counts(
            bins=20, sort=False
        )
    )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Baseline vs Adaptive Model")

    performance = pd.DataFrame({
        "Model": ["Baseline", "Adaptive"],
        "Accuracy (%)": [
            metrics["baseline_accuracy"] * 100,
            metrics["adaptive_accuracy"] * 100
        ],
        "Brier Score": [
            metrics["baseline_brier_score"],
            metrics["adaptive_brier_score"]
        ],
        "Log Loss": [
            metrics["baseline_log_loss"],
            metrics["adaptive_log_loss"]
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True
    )

    st.subheader("Accuracy Comparison")

    st.bar_chart(
        performance.set_index("Model")["Accuracy (%)"]
    )

    st.subheader("Brier Score Comparison")

    st.bar_chart(
        performance.set_index("Model")["Brier Score"]
    )

    st.subheader("Log Loss Comparison")

    st.bar_chart(
        performance.set_index("Model")["Log Loss"]
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Adaptive Markov-Based Probabilistic Forecasting "
    "of Indian Wind Energy Under Concept Drift and Uncertainty"
)