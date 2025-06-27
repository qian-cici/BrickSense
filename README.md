# BrickSense
A Streamlit-based web app that predicts Dutch house prices using real transaction data. Built for a FinTech assignment.
# 🏠 BrickSense: House Price Estimator

BrickSense is a lightweight, interactive MVP that enables users to estimate housing prices in the Netherlands. Users can input property features such as location, build type, and size, and instantly receive an estimated price. The application is built using Python and Streamlit, and uses a Random Forest model trained on cleaned Funda data.

---

## 🚀 Key Features

- Estimate house prices based on custom user inputs
- Friendly web-based interface (via Streamlit)
- Favorites functionality to save predictions
- Real-time ML prediction with trained Random Forest model
- Cached model training and data loading for performance

---

## 🧰 Tech Stack

| Layer      | Tool / Library            |
|------------|---------------------------|
| Frontend   | Streamlit                 |
| Backend    | Python, Pandas, NumPy     |
| ML Model   | Scikit-learn (RandomForestRegressor) |
| Deployment | Local / Streamlit Cloud   |

---
## 🧮 Model Inputs (Features Used)

The model uses the following features, divided into numeric and categorical types:

**Numeric Features:**
- Lot size (m²)
- Living space size (m²)
- Build year
- Number of rooms
- Number of toilets
- Number of floors

**Categorical Features:**
- City (selected from dataset)
- Street (extracted from address)
- Build type (e.g., detached, semi-detached)
- House type (e.g., apartment, villa)
- Roof type (e.g., flat, gable)

All missing values are handled via imputation pipelines (`median` for numeric, `most_frequent` for categorical), and categorical variables are encoded using `OneHotEncoder`.

---
## 🧠 Architecture
User Input (Streamlit UI)
↓
Preprocessing Pipeline (numeric + categorical)
↓
RandomForestRegressor model (sklearn)
↓
Predicted House Price

---

## 📦 Installation & Running the App

1. Clone this repository:
   ```bash
   git clone https://github.com/qian-cici/BrickSense.git
   cd BrickSense
Install dependencies:

pip install -r requirements.txt
Run the application:

streamlit run gui++.py
Make sure raw_data.csv is present in the root folder.

📊 Data Source
The housing data used in this project is based on cleaned and preprocessed data derived from the open-source Funda Webscraper Project by Bryan Lusse. Here is the link: https://github.com/bryanlusse/HousePrices__Webscraper
Only cleaned and anonymized versions are used here.

🔒 License
This project is licensed under the MIT License.


💡 Future Enhancements
Precise address-level price prediction with fuzzy matching

Integration of map-based UI
API or real-time price monitoring
Paid features (investor-oriented alerts and price analysis)


