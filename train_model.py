import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "data/borewell_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "rainfall",
    "soil_type",
    "elevation",
    "groundwater_depth",
    "slope",
    "land_use"
]

TARGET_COLUMN = "borewell_suitability"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


print("\nFeatures:")
print(FEATURE_COLUMNS)

print("\nTarget:")
print(TARGET_COLUMN)


# ============================================================
# 3. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "latitude",
    "longitude",
    "rainfall",
    "elevation",
    "groundwater_depth",
    "slope"
]

categorical_features = [
    "soil_type",
    "land_use"
]


# ============================================================
# 4. ENCODE CATEGORICAL FEATURES
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 5. CREATE RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ============================================================
# 6. CREATE COMPLETE ML PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest)
    ]
)


# ============================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. TRAIN RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 11. SAVE TRAINED MODEL
# ============================================================

MODEL_PATH = "models/borewell_rf_model.joblib"

joblib.dump(model, MODEL_PATH)

print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print(f"Saved model: {MODEL_PATH}")