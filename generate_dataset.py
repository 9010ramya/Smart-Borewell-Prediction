"""
generate_dataset.py
--------------------
Generates a SYNTHETIC / DEMO dataset for the
AI-Based Borewell Recommendation System (college project).

This is NOT real groundwater survey data. It is created using
reasonable domain assumptions (more rainfall + shallow groundwater +
flat slope + favorable soil => higher suitability) combined with
random noise, so the Random Forest model has realistic, non-trivial
patterns to learn -- not a simple one-column formula.
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # reproducibility: same data every time this script runs

N_ROWS = 800

# ----------------------------------------------------------------
# 1. Generate base geographic coordinates (loosely centered around
#    a representative Indian region, just for realism)
# ----------------------------------------------------------------
latitude = np.random.uniform(15.0, 21.0, N_ROWS)      # decimal degrees
longitude = np.random.uniform(73.0, 80.0, N_ROWS)     # decimal degrees

# ----------------------------------------------------------------
# 2. Generate environmental features with realistic ranges
# ----------------------------------------------------------------
rainfall = np.random.normal(900, 300, N_ROWS)          # mm/year
rainfall = np.clip(rainfall, 200, 2000)

elevation = np.random.normal(450, 200, N_ROWS)         # meters
elevation = np.clip(elevation, 10, 1200)

groundwater_depth = np.random.normal(15, 8, N_ROWS)    # meters
groundwater_depth = np.clip(groundwater_depth, 1, 45)

slope = np.random.exponential(scale=5, size=N_ROWS)    # degrees
slope = np.clip(slope, 0, 35)

soil_types = ["Sandy", "Loamy", "Clayey", "Rocky"]
soil_weights = [0.30, 0.35, 0.20, 0.15]
soil_type = np.random.choice(soil_types, size=N_ROWS, p=soil_weights)

land_uses = ["Agricultural", "Forest", "Barren", "Urban"]
land_use_weights = [0.45, 0.20, 0.20, 0.15]
land_use = np.random.choice(land_uses, size=N_ROWS, p=land_use_weights)

# ----------------------------------------------------------------
# 3. Assign a numeric "favorability" score to categorical features
#    (domain-assumption based, used only to blend into the overall
#    score -- not copied directly into the target)
# ----------------------------------------------------------------
soil_score_map = {"Sandy": 0.5, "Loamy": 0.9, "Clayey": 0.3, "Rocky": 0.1}
land_use_score_map = {"Agricultural": 0.8, "Forest": 0.7, "Barren": 0.4, "Urban": 0.2}

soil_score = np.array([soil_score_map[s] for s in soil_type])
land_use_score = np.array([land_use_score_map[l] for l in land_use])

# ----------------------------------------------------------------
# 4. Normalize numeric features to 0-1 range so they can be
#    combined fairly into a single weighted score
# ----------------------------------------------------------------
def normalize(arr, invert=False):
    norm = (arr - arr.min()) / (arr.max() - arr.min())
    return 1 - norm if invert else norm

rainfall_n = normalize(rainfall)                  # more rain = better
elevation_n = normalize(elevation, invert=True)   # lower elevation = better
gw_depth_n = normalize(groundwater_depth, invert=True)  # shallower = better
slope_n = normalize(slope, invert=True)           # flatter = better

# ----------------------------------------------------------------
# 5. Combine into a weighted suitability score.
#    Weights reflect rough real-world importance:
#    groundwater depth and rainfall matter most, slope/soil/land use
#    matter moderately, elevation matters least.
# ----------------------------------------------------------------
weights = {
    "gw_depth": 0.30,
    "rainfall": 0.25,
    "soil": 0.15,
    "slope": 0.12,
    "land_use": 0.10,
    "elevation": 0.08,
}

raw_score = (
    weights["gw_depth"] * gw_depth_n +
    weights["rainfall"] * rainfall_n +
    weights["soil"] * soil_score +
    weights["slope"] * slope_n +
    weights["land_use"] * land_use_score +
    weights["elevation"] * elevation_n
)

# ----------------------------------------------------------------
# 6. Add random noise so the boundary between classes is fuzzy
#    (this is what stops the target from being a clean formula
#    of the inputs, and mimics real-world unpredictability)
# ----------------------------------------------------------------
noise = np.random.normal(0, 0.08, N_ROWS)
final_score = raw_score + noise
final_score = np.clip(final_score, 0, 1)

# ----------------------------------------------------------------
# 7. Convert continuous score into 3 classes using thresholds
#    chosen to give a realistic, not perfectly balanced, class split
# ----------------------------------------------------------------
def classify(score):
    if score < 0.48:
        return "Low"
    elif score < 0.60:
        return "Medium"
    else:
        return "High"

borewell_suitability = [classify(s) for s in final_score]

# ----------------------------------------------------------------
# 8. Assemble final DataFrame
# ----------------------------------------------------------------
df = pd.DataFrame({
    "latitude": np.round(latitude, 4),
    "longitude": np.round(longitude, 4),
    "rainfall": np.round(rainfall, 1),
    "soil_type": soil_type,
    "elevation": np.round(elevation, 1),
    "groundwater_depth": np.round(groundwater_depth, 2),
    "slope": np.round(slope, 2),
    "land_use": land_use,
    "borewell_suitability": borewell_suitability,
})

# Shuffle rows so classes aren't grouped in generation order
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/borewell_dataset.csv", index=False)
print("Dataset generated: data/borewell_dataset.csv")
print(df["borewell_suitability"].value_counts())