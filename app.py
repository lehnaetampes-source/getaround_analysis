import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Getaround — AI Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY    = "#00B0B9"
DARK       = "#1C1C1C"
LIGHT_BG   = "#F5F7F9"
CARD_BG    = "#FFFFFF"
ACCENT_RED = "#FF4B4B"
ACCENT_YLW = "#FFB400"
ACCENT_GRN = "#00C48C"

st.markdown(f"""
<style>
html, body {{ font-family: Arial; }}
.stApp {{ background-color: {LIGHT_BG}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SAFE DATA LOADING
# ─────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        return joblib.load("pricing_model.joblib")
    except Exception as e:
        st.warning("⚠️ Model could not be loaded in this environment.")
        st.warning(str(e))
        return None


@st.cache_data
def load_delay():
    try:
        return pd.read_excel("get_around_delay_analysis.xlsx")
    except:
        return pd.DataFrame()


@st.cache_data
def load_pricing():
    try:
        return pd.read_csv("get_around_pricing_project.csv")
    except:
        return pd.DataFrame()


mdl = load_model()
df_d = load_delay()
df_p = load_pricing()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:

    st.title("🚗 Getaround")

    page = st.radio(
        "Navigation",
        ["Delay Analysis", "Pricing Simulator", "ML Insights"]
    )

    st.markdown("---")

    st.write("Dataset info")
    st.write(f"Delay rows: {len(df_d)}")
    st.write(f"Pricing rows: {len(df_p)}")


# ─────────────────────────────────────────
# PAGE 1
# ─────────────────────────────────────────

if page == "Delay Analysis":

    st.title("📊 Delay Analysis")

    if df_d.empty:
        st.warning("Delay dataset not available in deployed version.")
        st.stop()

    total = len(df_d)
    late = int((df_d["delay_at_checkout_in_minutes"] > 0).sum())

    st.metric("Total rentals", total)
    st.metric("Late returns", late)

    fig, ax = plt.subplots()
    ax.hist(df_d["delay_at_checkout_in_minutes"].dropna(), bins=60)

    st.pyplot(fig)


# ─────────────────────────────────────────
# PAGE 2
# ─────────────────────────────────────────

elif page == "Pricing Simulator":

    st.title("💰 Pricing Simulator")

    mileage = st.number_input("Mileage", 0, 500000, 80000)
    engine_power = st.number_input("Engine power", 50, 500, 120)

    fuel = st.selectbox(
        "Fuel",
        ["diesel", "petrol", "electro", "hybrid_petrol"]
    )

    car_type = st.selectbox(
        "Car type",
        ["sedan", "suv", "hatchback"]
    )

    if st.button("Predict price"):

        X = pd.DataFrame([[
            "Renault",
            mileage,
            engine_power,
            fuel,
            "black",
            car_type,
            True,
            True,
            True,
            False,
            True,
            True,
            False
        ]],
        columns=[
            "model_key",
            "mileage",
            "engine_power",
            "fuel",
            "paint_color",
            "car_type",
            "private_parking_available",
            "has_gps",
            "has_air_conditioning",
            "automatic_car",
            "has_getaround_connect",
            "has_speed_regulator",
            "winter_tires"
        ])

        if mdl is None:
            st.error("Model unavailable in deployed version.")
        else:
            price = mdl.predict(X)[0]
            st.success(f"Recommended price: {price:.0f} € / day")


# ─────────────────────────────────────────
# PAGE 3
# ─────────────────────────────────────────

elif page == "ML Insights":

    st.title("🤖 ML Model Insights")

    st.write("Gradient Boosting model")

    if df_p.empty:
        st.warning("Pricing dataset not available in deployed version.")
        st.stop()

    fig, ax = plt.subplots()

    ax.scatter(
        df_p["engine_power"],
        df_p["rental_price_per_day"],
        alpha=0.3
    )

    ax.set_xlabel("Engine power")
    ax.set_ylabel("Price")

    st.pyplot(fig)