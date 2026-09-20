import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Forest Fire Prediction",
    page_icon="🔥",
    layout="centered"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🔥 Forest Fire Prediction")

st.write(
    "This application uses Machine Learning to predict "
    "forest fire occurrence and estimate the burned area."
)

st.divider()

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("forestfires.csv")


# ---------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------

@st.cache_resource
def train_models():

    df = load_data()

    # Create binary target
    # 1 = Fire occurred
    # 0 = No fire
    df["fire_occurred"] = (df["area"] > 0).astype(int)

    # Convert categorical columns into numerical columns
    df_encoded = pd.get_dummies(
        df,
        columns=["month", "day"],
        drop_first=True
    )

    # Features
    feature_columns = [
        column
        for column in df_encoded.columns
        if column not in ["area", "fire_occurred"]
    ]

    X = df_encoded[feature_columns]

    # ------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------

    y_classification = df_encoded["fire_occurred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_classification,
        test_size=0.2,
        random_state=42,
        stratify=y_classification
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    classifier.fit(X_train, y_train)

    classification_prediction = classifier.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        classification_prediction
    )

    # ------------------------------------------------
    # REGRESSION
    # ------------------------------------------------

    # Log transformation of burned area
    y_regression = np.log1p(df_encoded["area"])

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X,
        y_regression,
        test_size=0.2,
        random_state=42
    )

    regressor = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    regressor.fit(
        X_train_reg,
        y_train_reg
    )

    regression_prediction = regressor.predict(X_test_reg)

    mae = mean_absolute_error(
        y_test_reg,
        regression_prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_reg,
            regression_prediction
        )
    )

    r2 = r2_score(
        y_test_reg,
        regression_prediction
    )

    return (
        df,
        feature_columns,
        classifier,
        regressor,
        accuracy,
        mae,
        rmse,
        r2
    )


# ---------------------------------------------------
# LOAD TRAINED MODELS
# ---------------------------------------------------

try:

    (
        df,
        feature_columns,
        classifier,
        regressor,
        accuracy,
        mae,
        rmse,
        r2
    ) = train_models()

except Exception as e:

    st.error("Error loading the dataset or training the model.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.header("🌲 Enter Environmental Conditions")

col1, col2 = st.columns(2)

with col1:

    X_coordinate = st.number_input(
        "X Coordinate",
        min_value=1,
        max_value=9,
        value=7
    )

    Y_coordinate = st.number_input(
        "Y Coordinate",
        min_value=1,
        max_value=9,
        value=5
    )

    month = st.selectbox(
        "Month",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec"
        ],
        index=7
    )

    day = st.selectbox(
        "Day",
        [
            "mon",
            "tue",
            "wed",
            "thu",
            "fri",
            "sat",
            "sun"
        ],
        index=4
    )

    FFMC = st.number_input(
        "FFMC",
        min_value=0.0,
        value=90.0,
        step=0.1
    )

    DMC = st.number_input(
        "DMC",
        min_value=0.0,
        value=50.0,
        step=0.1
    )


with col2:

    DC = st.number_input(
        "DC",
        min_value=0.0,
        value=400.0,
        step=0.1
    )

    ISI = st.number_input(
        "ISI",
        min_value=0.0,
        value=9.0,
        step=0.1
    )

    temperature = st.number_input(
        "Temperature",
        value=25.0,
        step=0.1
    )

    humidity = st.number_input(
        "Relative Humidity (RH)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

    wind = st.number_input(
        "Wind Speed",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    rain = st.number_input(
        "Rain",
        min_value=0.0,
        value=0.0,
        step=0.1
    )


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button(
    "🔥 Predict Forest Fire",
    use_container_width=True
):

    # Create input DataFrame
    input_data = pd.DataFrame(
        [{
            "X": X_coordinate,
            "Y": Y_coordinate,
            "month": month,
            "day": day,
            "FFMC": FFMC,
            "DMC": DMC,
            "DC": DC,
            "ISI": ISI,
            "temp": temperature,
            "RH": humidity,
            "wind": wind,
            "rain": rain
        }]
    )

    # One-hot encoding
    input_encoded = pd.get_dummies(
        input_data,
        columns=["month", "day"],
        drop_first=True
    )

    # Make sure input columns are exactly the same
    # as training columns
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # ------------------------------------------------
    # CLASSIFICATION PREDICTION
    # ------------------------------------------------

    fire_prediction = classifier.predict(
        input_encoded
    )[0]

    fire_probability = classifier.predict_proba(
        input_encoded
    )[0][1]

    # ------------------------------------------------
    # BURNED AREA PREDICTION
    # ------------------------------------------------

    predicted_log_area = regressor.predict(
        input_encoded
    )[0]

    predicted_area = np.expm1(
        predicted_log_area
    )

    predicted_area = max(
        0,
        predicted_area
    )

    # ------------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------------

    st.divider()

    st.header("📊 Prediction Result")

    if fire_prediction == 1:

        st.error(
            "🔥 Forest Fire Predicted"
        )

    else:

        st.success(
            "🌳 No Forest Fire Predicted"
        )

    # Probability
    st.metric(
        "🔥 Fire Probability",
        f"{fire_probability * 100:.2f}%"
    )

    # Estimated area
    st.metric(
        "🌲 Estimated Burned Area",
        f"{predicted_area:.2f} hectares"
    )


# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------

st.divider()

st.header("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Classification Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:

    st.metric(
        "MAE",
        f"{mae:.3f}"
    )

with col3:

    st.metric(
        "RMSE",
        f"{rmse:.3f}"
    )

with col4:

    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )


# ---------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------

st.divider()

st.header("📊 Dataset Information")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Number of Records",
        len(df)
    )

with col2:

    st.metric(
        "Number of Features",
        len(df.columns)
    )


with st.expander("🔍 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "🔥 Forest Fire Prediction | "
    "Machine Learning + Streamlit"
)

st.caption(
    "This project is intended for educational purposes "
    "and uses historical forest-fire data."
)
