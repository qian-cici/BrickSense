import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="BrickSense - House Price Estimator", layout="wide")

# ----- SESSION STATE -----
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
    st.session_state.latest_input = None

# ----- HEADER -----
st.markdown(
    """
    <div style='background-color: #0072C6; padding: 1rem 2rem;'>
        <h1 style='color: white; font-weight: 700; display: inline;'>🏠 BrickSense</h1>
        <span style='float: right; color: white; font-size: 1.2rem;'>House Price Estimator</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ----- LOAD DATA -----
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("raw_data.csv", quotechar='"', encoding='utf-8')

    def extract_numbers(s):
        if isinstance(s, str):
            nums = re.findall(r'\d+', s)
            if not nums:
                return np.nan
            return ",".join(nums)
        return s

    columns_to_clean = [
        'Price', 'Lot size (m2)', 'Living space size (m2)', 'Build year',
        'Rooms', 'Toilet', 'Floors'
    ]
    for col in columns_to_clean:
        df[col] = df[col].apply(extract_numbers)
        df[col] = df[col].apply(lambda x: float(str(x).split(",")[0]) if pd.notnull(x) else np.nan)

    df['Price'] = df['Price'] * 1000
    df = df.dropna(subset=['Price'])

    # Extract Street from Address
    df['Street'] = df['Address'].astype(str).apply(lambda x: x.split()[0] if isinstance(x, str) else np.nan)

    return df

df = load_data()
st.toast("Hi, we are trying our best to help you 🙂", icon="💪")

# ----- MODEL TRAINING -----
@st.cache_resource
def train_model(df):
    categorical_features = ['Build type', 'House type', 'Roof', 'City', 'Street']
    numeric_features = ['Lot size (m2)', 'Living space size (m2)', 'Build year', 'Rooms', 'Toilet', 'Floors']
    features = numeric_features + categorical_features

    X = df[features]
    y = df['Price']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    return model

model = train_model(df)

# ----- FAVORITES CHECKBOX -----
view_fav = st.checkbox("View Favorites", help="See your saved predictions", key="view_fav", value=False)

# ----- INPUT FORM -----
st.header("Predict House Price")

input_data = {}

# City first
city_options = ['Don\'t know'] + sorted(df['City'].dropna().unique().tolist())
input_data['City'] = st.selectbox("City", city_options)

# Street next
street_options = ['Don\'t know'] + sorted(df['Street'].dropna().unique().tolist())
input_data['Street'] = st.selectbox("Street", street_options)

# Numeric features
numeric_features = ['Lot size (m2)', 'Living space size (m2)', 'Build year', 'Rooms', 'Toilet', 'Floors']
for feature in numeric_features:
    input_data[feature] = st.number_input(
        feature,
        min_value=0,
        max_value=10000,
        value=0,
        step=1
    )

# Other categorical features
for feature in ['Build type', 'House type', 'Roof']:
    options = ['Don\'t know'] + sorted(df[feature].dropna().unique().tolist())
    input_data[feature] = st.selectbox(feature, options)

# ----- PREDICT -----
if st.button("Estimate Price"):
    user_df = pd.DataFrame([input_data])
    for col in user_df.columns:
        if user_df[col][0] == "Don't know":
            user_df[col] = np.nan
    prediction = model.predict(user_df)[0]
    st.session_state.latest_prediction = prediction
    st.session_state.latest_input = input_data.copy()
    st.success(f"💰 Estimated Price: € {prediction:,.2f}")

# ----- ADD TO FAVORITE -----
if st.session_state.latest_prediction is not None:
    if st.button("Add to Favorite"):
        saved = st.session_state.latest_input.copy()
        saved["Predicted Price"] = round(st.session_state.latest_prediction, 2)
        st.session_state.favorites.append(saved)
        st.sidebar.success("✅ Added to favorites!")

# ----- FAVORITES -----
if view_fav and st.session_state.favorites:
    st.sidebar.subheader("🏷️ Your Favorite Properties")
    for i, fav in enumerate(st.session_state.favorites):
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**🏡 Property #{i+1}**")
        for key, value in fav.items():
            st.sidebar.markdown(f"- **{key}**: {value}")
elif view_fav:
    st.sidebar.info("You have no favorites yet.")
