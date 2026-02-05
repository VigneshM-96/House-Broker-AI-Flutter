import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --------------------------
# Utils
# --------------------------
location_map = {}

def fit_encoder(df):
    global location_map
    locations = df['location'].unique()
    location_map = {loc: i for i, loc in enumerate(locations)}
    return location_map

def encode_location(loc):
    return location_map.get(loc, -1)  # -1 for unknown

# --------------------------
# Paths
# --------------------------
DATA_PATH = r"C:\Plans\Projects\Flutter Projects\house_broker_ai\backend\data\house_prices.csv"
MODELS_FOLDER = "models"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV not found at {DATA_PATH}")

# --------------------------
# Load & encode data
# --------------------------
df = pd.read_csv(DATA_PATH)
location_map = fit_encoder(df)
df['location'] = df['location'].apply(encode_location)

X = df.drop("Price_Lakhs", axis=1)
y = df["Price_Lakhs"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Train model
# --------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------
# Save model & encoder
# --------------------------
os.makedirs(MODELS_FOLDER, exist_ok=True)
joblib.dump(model, os.path.join(MODELS_FOLDER, "house_price_model.pkl"))
joblib.dump(location_map, os.path.join(MODELS_FOLDER, "location_encoder.pkl"))

print("✅ Model & encoder saved successfully")