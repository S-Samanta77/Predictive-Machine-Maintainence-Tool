import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai

# PAGE CONFIGURATION

st.set_page_config(
    page_title="AI-Powered Predictive Maintenance & Failure Analysis System",
    page_icon="🤖",
    layout="wide"
)

# LOAD MODEL FILES

model = joblib.load("random_forest.pkl")
encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# GEMINI CONFIGURATION

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("Gemini API key not found. Please configure GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

gemini_model = genai.GenerativeModel("gemini-3.1-flash-lite")

# SESSION STATE

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction_text" not in st.session_state:
    st.session_state.prediction_text = ""

if "prediction_confidence" not in st.session_state:
    st.session_state.prediction_confidence = 0

if "parameter_status" not in st.session_state:
    st.session_state.parameter_status = []

if "input_values" not in st.session_state:
    st.session_state.input_values = {}

# NORMAL OPERATING RANGES

NORMAL_RANGES = {

    "Air temperature [K]": (295,305),

    "Process temperature [K]": (295,308),

    "Rotational speed [rpm]": (1200,1800),

    "Torque [Nm]": (20,60),

    "Tool wear [min]": (0,180)

}

# TITLE

st.title("🤖 AI-Powered Predictive Maintenance & Failure Analysis System")

st.write(
    """
Predict whether a manufacturing machine is likely to fail
using a trained Machine Learning model and generate an
AI-powered maintenance report using Google Gemini.
"""
)

st.divider()

# USER INPUTS

st.subheader("Machine Parameters")

left, right = st.columns(2)

with left:

    st.caption("Choose a type")
    product_type = st.selectbox(

        "Product Type",

        ["L","M","H"]

    )

    st.caption("Valid Range: 295 K - 305 K")
    air_temp = st.number_input(

        "Air Temperature (K)",

        min_value=295.0,

        max_value=305.0,

        value=300.0,

        step=0.1

    )

    st.caption("Valid Range: 305 K - 315 K")
    process_temp = st.number_input(

        "Process Temperature (K)",

        min_value=305.0,

        max_value=315.0,

        value=310.0,

        step=0.1

    )

with right:

    st.caption("Valid Range: 1100 RPM - 3000 RPM")
    rpm = st.number_input(

        "Rotational Speed (RPM)",

        min_value=1100,

        max_value=3000,

        value=1500,

        step=10

    )

    st.caption("Valid Range: 0 Nm - 80 Nm")
    torque = st.number_input(

        "Torque (Nm)",

        min_value=0.0,

        max_value=80.0,

        value=40.0,

        step=0.5

    )

    st.caption("Valid Range: 0 - 300 minutes")
    tool_wear = st.number_input(

        "Tool Wear (minutes)",

        min_value=0,

        max_value=300,

        value=100,

        step=1

    )

st.divider()

# PREDICT BUTTON

predict_button = st.button(

    "🔵 Predict",

    use_container_width=True

)

# PREDICTION

if predict_button:

    # Encode Product Type

    encoded_type = encoder.transform([product_type])[0]

    # Create Input DataFrame

    input_df = pd.DataFrame({

        "Type":[encoded_type],

        "Air temperature [K]":[air_temp],

        "Process temperature [K]":[process_temp],

        "Rotational speed [rpm]":[rpm],

        "Torque [Nm]":[torque],

        "Tool wear [min]":[tool_wear]

    })

    # Maintain same feature order used during training

    input_df = input_df[feature_names]

    # Predict

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)

    confidence = probabilities.max() * 100

    # Store Inputs

    st.session_state.input_values = {

        "Product Type": product_type,

        "Air temperature [K]": air_temp,

        "Process temperature [K]": process_temp,

        "Rotational speed [rpm]": rpm,

        "Torque [Nm]": torque,

        "Tool wear [min]": tool_wear

    }

    # Prediction Text

    if prediction == 1:

        st.session_state.prediction_text = "Machine Failure Predicted"

    else:

        st.session_state.prediction_text = "No Machine Failure Predicted"

    st.session_state.prediction_done = True

    st.session_state.prediction_confidence = confidence

    # PARAMETER STATUS

    parameter_status = []

    for feature, value in st.session_state.input_values.items():

        if feature == "Product Type":

            continue

        minimum, maximum = NORMAL_RANGES[feature]

        if value < minimum:

            parameter_status.append({

                "Feature": feature,

                "Current": value,

                "Expected": f"{minimum} - {maximum}",

                "Status": "LOW"

            })

        elif value > maximum:

            parameter_status.append({

                "Feature": feature,

                "Current": value,

                "Expected": f"{minimum} - {maximum}",

                "Status": "HIGH"

            })

    st.session_state.parameter_status = parameter_status

# DISPLAY PREDICTION

if st.session_state.prediction_done:

    st.divider()

    st.subheader("Prediction")

    if st.session_state.prediction_text == "Machine Failure Predicted":

        st.error("⚠ Machine Failure Predicted")

    else:

        st.success("✅ No Machine Failure Predicted")


    st.subheader("Prediction Confidence")

    st.metric(

    label="Prediction Confidence",

    value=f"{st.session_state.prediction_confidence:.2f}%"

)

    st.progress(

        float(st.session_state.prediction_confidence) / 100

    )

# PARAMETER STATUS

if st.session_state.prediction_done:

    st.divider()

    st.subheader("Parameter Status")

    if len(st.session_state.parameter_status) == 0:

        st.success(

            "🟢 All monitored parameters are within the expected operating range."

        )

    else:

        for item in st.session_state.parameter_status:

            icon = "🔴"

            if item["Status"] == "LOW":

                condition = "Below Expected Range"

            else:

                condition = "Above Expected Range"

            st.warning(

f"""
{icon} **{item['Feature']}**

Current Value : **{item['Current']}**

Expected Range : **{item['Expected']}**

Status : **{condition}**
"""
            )

        st.success(

            "🟢 All remaining monitored parameters are within the expected operating range."

        )

# GENERATE AI REPORT BUTTON

if st.session_state.prediction_done:

    st.divider()

    generate_report = st.button(

        "🟢 Generate AI Report",

        use_container_width=True

    )

else:

    generate_report = False

# GEMINI REPORT

if generate_report:

    with st.spinner("Generating AI Maintenance Report..."):

        # MACHINE READINGS

        machine_readings = ""

        for key, value in st.session_state.input_values.items():

            machine_readings += f"{key} : {value}\n"

        # PARAMETER STATUS

        if len(st.session_state.parameter_status) == 0:

            parameter_status_text = (

                "All monitored parameters are within the expected operating range."

            )

        else:

            parameter_status_text = ""

            for item in st.session_state.parameter_status:

                parameter_status_text += (

                    f"{item['Feature']}\n"

                    f"Current Value : {item['Current']}\n"

                    f"Expected Range : {item['Expected']}\n"

                    f"Status : {item['Status']}\n\n"

                )

            parameter_status_text += (

                "All remaining monitored parameters are within the expected operating range."

            )

        # PROMPT

        prompt = f"""

You are an experienced Industrial Maintenance Engineer.

A trained Random Forest Machine Learning model has already completed the prediction.

DO NOT change the prediction.

DO NOT question the prediction.

DO NOT invent machine readings.

Your task is only to explain the prediction professionally and provide actionable maintenance recommendations.

--------------------------------------------------

Prediction

{st.session_state.prediction_text}

Prediction Confidence

{st.session_state.prediction_confidence:.2f} %

--------------------------------------------------

Machine Readings

{machine_readings}

--------------------------------------------------

Parameter Status

{parameter_status_text}

--------------------------------------------------

Generate a professional maintenance report between 300 and 400 words.

The report MUST contain the following headings:

1. Executive Summary

2. Failure Analysis

3. Key Contributing Factors

4. Severity Level

5. Immediate Maintenance Actions

6. Preventive Maintenance

7. Business Impact

8. Safety Recommendations

Use professional engineering language.

Only explain based on the information provided.

"""

        # GEMINI RESPONSE

        try:

            response = gemini_model.generate_content(prompt)

            st.divider()

            st.subheader("AI Maintenance Report")

            st.markdown(response.text)

        except Exception as e:

            st.error("Unable to generate AI report.")

            st.exception(e)

st.divider()
