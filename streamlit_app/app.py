import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ── Robust model path (works regardless of where you run streamlit from) ──────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "earthquake_damage_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Label maps for readable UI ────────────────────────────────────────────────
LAND_SURFACE     = {"Flat (n)": "n", "Sloped (o)": "o", "Terraced (t)": "t"}
FOUNDATION_TYPE  = {"Mud (h)": "h", "Bamboo/Timber (i)": "i", "RC (r)": "r", "Other (u)": "u", "Stone/Brick (w)": "w"}
ROOF_TYPE        = {"Bamboo/Timber (n)": "n", "RCC/RB/RBC (q)": "q", "Light Metal (x)": "x"}
GROUND_FLOOR     = {"Mud (f)": "f", "Brick/Stone (m)": "m", "Timber (v)": "v", "Other (x)": "x", "RC (z)": "z"}
OTHER_FLOOR      = {"TImber/Bamboo (j)": "j", "Timber (q)": "q", "RCC/RB/RBC (s)": "s", "Other (x)": "x"}
POSITION         = {"Not attached (j)": "j", "Attached-1 side (o)": "o", "Attached-2 sides (s)": "s", "Attached-3 sides (t)": "t"}
PLAN_CONFIG      = {"Rectangular (a)": "a", "Square (c)": "c", "L-shape (d)": "d", "F-shape (f)": "f",
                    "Multi-edge (m)": "m", "Other (n)": "n", "Oval/Round (o)": "o", "E-shape (q)": "q",
                    "S-shape (s)": "s", "U-shape (u)": "u"}
OWNERSHIP        = {"Ownership (a)": "a", "Rented (r)": "r", "Joint (v)": "v", "Other (w)": "w"}
DAMAGE_LABEL     = {1: "🟢 Grade 1 — Low Damage",
                    2: "🟡 Grade 2 — Medium Damage",
                    3: "🔴 Grade 3 — High Damage"}
DAMAGE_COLOR     = {1: "success", 2: "warning", 3: "error"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Earthquake Damage Predictor", page_icon="🏚️", layout="wide")
st.title("🏚️ Earthquake Damage Prediction")
st.markdown("Predicts the **damage grade** of a building after an earthquake using a trained Random Forest model.")
st.divider()

# ── Input form ──────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location")
    geo1 = st.number_input("Geo Level 1 ID", min_value=0, max_value=30,  value=5)
    geo2 = st.number_input("Geo Level 2 ID", min_value=0, max_value=1427, value=300)
    geo3 = st.number_input("Geo Level 3 ID", min_value=0, max_value=12567, value=1000)
    land = st.selectbox("Land Surface Condition", list(LAND_SURFACE.keys()))

with col2:
    st.subheader("🏗️ Building Structure")
    age    = st.number_input("Building Age (years)", min_value=0,  max_value=995, value=20)
    floors = st.number_input("No. of Floors", min_value=1, max_value=9,   value=2)
    area   = st.number_input("Area Percentage (%)", min_value=1,  max_value=100, value=10)
    height = st.number_input("Height Percentage (%)", min_value=1, max_value=100, value=10)
    families = st.number_input("No. of Families", min_value=1, max_value=9, value=1)

    st.subheader("🏠 Building Type")
    foundation = st.selectbox("Foundation Type",   list(FOUNDATION_TYPE.keys()))
    roof       = st.selectbox("Roof Type",         list(ROOF_TYPE.keys()))
    ground_fl  = st.selectbox("Ground Floor Type", list(GROUND_FLOOR.keys()))
    other_fl   = st.selectbox("Other Floor Type",  list(OTHER_FLOOR.keys()))
    position   = st.selectbox("Position",          list(POSITION.keys()))
    plan_cfg   = st.selectbox("Plan Configuration",list(PLAN_CONFIG.keys()))
    ownership  = st.selectbox("Legal Ownership",   list(OWNERSHIP.keys()))

with col3:
    st.subheader("🧱 Superstructure Materials")
    adobe       = st.checkbox("Adobe/Mud")
    mud_stone   = st.checkbox("Mud Mortar Stone")
    stone_flag  = st.checkbox("Stone Flag")
    cement_stone= st.checkbox("Cement Mortar Stone")
    mud_brick   = st.checkbox("Mud Mortar Brick")
    cement_brick= st.checkbox("Cement Mortar Brick")
    timber      = st.checkbox("Timber")
    bamboo      = st.checkbox("Bamboo")
    rc_non_eng  = st.checkbox("RC Non-Engineered")
    rc_eng      = st.checkbox("RC Engineered")
    other_mat   = st.checkbox("Other Material")

    st.subheader("🏭 Secondary Use")
    secondary       = st.checkbox("Has Secondary Use")
    use_agri        = st.checkbox("Agriculture")
    use_hotel       = st.checkbox("Hotel")
    use_rental      = st.checkbox("Rental")
    use_institution = st.checkbox("Institution")
    use_school      = st.checkbox("School")
    use_industry    = st.checkbox("Industry")
    use_health      = st.checkbox("Health Post")
    use_gov         = st.checkbox("Gov Office")
    use_police      = st.checkbox("Police")
    use_other_sec   = st.checkbox("Other Use")

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Damage Grade", type="primary", use_container_width=True):

    # Derived / engineered features
    height_per_floor = height / floors if floors > 0 else 0
    age_group  = pd.cut([age], bins=[0,10,30,50,1000], labels=["New","Medium","Old","Very_Old"])[0]
    area_val   = area  # will be bucketed same way as training via qcut — approximate with label
    # We pass raw area_percentage; the pipeline's OHE was fitted on the string labels from qcut
    # So we need to re-create the label the same way training did
    # Using approximate quartile boundaries from training data (1-5, 5-8, 8-14, 14-100)
    if area <= 5:   area_group = "Small"
    elif area <= 8: area_group = "Medium"
    elif area <= 14:area_group = "Large"
    else:           area_group = "Huge"

    sample = pd.DataFrame([{
        # Required by pipeline (was in training data)
        "building_id": 0,
        # Numerical
        "geo_level_1_id": geo1,
        "geo_level_2_id": geo2,
        "geo_level_3_id": geo3,
        "count_floors_pre_eq": floors,
        "age": age,
        "area_percentage": area,
        "height_percentage": height,
        "has_superstructure_adobe_mud": int(adobe),
        "has_superstructure_mud_mortar_stone": int(mud_stone),
        "has_superstructure_stone_flag": int(stone_flag),
        "has_superstructure_cement_mortar_stone": int(cement_stone),
        "has_superstructure_mud_mortar_brick": int(mud_brick),
        "has_superstructure_cement_mortar_brick": int(cement_brick),
        "has_superstructure_timber": int(timber),
        "has_superstructure_bamboo": int(bamboo),
        "has_superstructure_rc_non_engineered": int(rc_non_eng),
        "has_superstructure_rc_engineered": int(rc_eng),
        "has_superstructure_other": int(other_mat),
        "count_families": families,
        "has_secondary_use": int(secondary),
        "has_secondary_use_agriculture": int(use_agri),
        "has_secondary_use_hotel": int(use_hotel),
        "has_secondary_use_rental": int(use_rental),
        "has_secondary_use_institution": int(use_institution),
        "has_secondary_use_school": int(use_school),
        "has_secondary_use_industry": int(use_industry),
        "has_secondary_use_health_post": int(use_health),
        "has_secondary_use_gov_office": int(use_gov),
        "has_secondary_use_use_police": int(use_police),
        "has_secondary_use_other": int(use_other_sec),
        "height_per_floor": height_per_floor,
        # Categorical
        "land_surface_condition": LAND_SURFACE[land],
        "foundation_type": FOUNDATION_TYPE[foundation],
        "roof_type": ROOF_TYPE[roof],
        "ground_floor_type": GROUND_FLOOR[ground_fl],
        "other_floor_type": OTHER_FLOOR[other_fl],
        "position": POSITION[position],
        "plan_configuration": PLAN_CONFIG[plan_cfg],
        "legal_ownership_status": OWNERSHIP[ownership],
        "age_group": age_group,
        "area_group": area_group,
    }])

    pred = model.predict(sample)[0] + 1   # model outputs 0,1,2 → convert back to 1,2,3
    proba = model.predict_proba(sample)[0]

    label  = DAMAGE_LABEL[pred]
    method = DAMAGE_COLOR[pred]

    if method == "success":
        st.success(f"### Predicted Damage: {label}")
    elif method == "warning":
        st.warning(f"### Predicted Damage: {label}")
    else:
        st.error(f"### Predicted Damage: {label}")

    st.markdown("#### Confidence Scores")
    prob_df = pd.DataFrame({
        "Damage Grade": ["Grade 1 (Low)", "Grade 2 (Medium)", "Grade 3 (High)"],
        "Probability": [f"{p*100:.1f}%" for p in proba]
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Model: Random Forest Classifier | Trained on 260,601 Nepal earthquake buildings")
