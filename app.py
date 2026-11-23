from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import requests


# ============================================================
# 1. CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/borewell_rf_model.joblib"

model = joblib.load(MODEL_PATH)

print("Borewell Random Forest model loaded successfully!")


# ============================================================
# 3. HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# 4. AUTOMATIC LOCATION DATA
# ============================================================

@app.route("/location-data", methods=["GET"])
def location_data():

    try:

        # ----------------------------------------------------
        # Read latitude and longitude from request
        # ----------------------------------------------------

        latitude = float(request.args.get("latitude"))
        longitude = float(request.args.get("longitude"))


        # ----------------------------------------------------
        # Validate coordinates
        # ----------------------------------------------------

        if latitude < -90 or latitude > 90:

            return jsonify({
                "success": False,
                "error": "Latitude must be between -90 and 90."
            }), 400


        if longitude < -180 or longitude > 180:

            return jsonify({
                "success": False,
                "error": "Longitude must be between -180 and 180."
            }), 400


        # ----------------------------------------------------
        # Open-Meteo Elevation API
        # ----------------------------------------------------

        elevation_url = (
            "https://api.open-meteo.com/v1/elevation"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
        )


        # ----------------------------------------------------
        # Call API
        # ----------------------------------------------------

        response = requests.get(
            elevation_url,
            timeout=10
        )


        # Raise error for HTTP failures

        response.raise_for_status()


        # Convert response to JSON

        data = response.json()


        # ----------------------------------------------------
        # Extract elevation
        # ----------------------------------------------------

        elevation_values = data.get("elevation")


        if not elevation_values:

            return jsonify({
                "success": False,
                "error": "Elevation data was not returned."
            }), 500


        elevation = elevation_values[0]


        # ----------------------------------------------------
        # Return data to browser
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": round(float(elevation), 2)
        })


    except requests.RequestException as e:

        print("Elevation API error:", e)

        return jsonify({
            "success": False,
            "error": "Unable to retrieve elevation data."
        }), 503


    except (TypeError, ValueError):

        return jsonify({
            "success": False,
            "error": "Invalid latitude or longitude."
        }), 400


    except Exception as e:

        print("Location data error:", e)

        return jsonify({
            "success": False,
            "error": "Unable to retrieve location data."
        }), 500


# ============================================================
# 5. PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Read values submitted from HTML form
        # ----------------------------------------------------

        latitude = float(request.form["latitude"])
        longitude = float(request.form["longitude"])
        rainfall = float(request.form["rainfall"])
        soil_type = request.form["soil_type"]
        elevation = float(request.form["elevation"])
        groundwater_depth = float(
            request.form["groundwater_depth"]
        )
        slope = float(request.form["slope"])
        land_use = request.form["land_use"]


        # ----------------------------------------------------
        # Create input DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "latitude": latitude,

            "longitude": longitude,

            "rainfall": rainfall,

            "soil_type": soil_type,

            "elevation": elevation,

            "groundwater_depth": groundwater_depth,

            "slope": slope,

            "land_use": land_use

        }])


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # Get prediction probabilities
        # ----------------------------------------------------

        probabilities = model.predict_proba(input_data)[0]

        classes = model.classes_

        probability_dict = dict(
            zip(classes, probabilities)
        )


        # ----------------------------------------------------
        # Get confidence
        # ----------------------------------------------------

        confidence = (
            probability_dict[prediction] * 100
        )

        confidence = round(confidence, 2)


        # ----------------------------------------------------
        # Generate recommendation
        # ----------------------------------------------------

        if prediction == "High":

            recommendation = (
                "The selected location appears suitable for "
                "borewell development. A detailed hydrogeological "
                "survey is recommended before drilling."
            )

        elif prediction == "Medium":

            recommendation = (
                "The selected location may be suitable for "
                "borewell development, but further groundwater "
                "assessment is recommended before drilling."
            )

        else:

            recommendation = (
                "The selected location appears less suitable for "
                "borewell development. Further groundwater and "
                "geological investigation is recommended."
            )


        # ----------------------------------------------------
        # Create input summary
        # ----------------------------------------------------

        input_summary = {

            "latitude": latitude,

            "longitude": longitude,

            "rainfall": rainfall,

            "soil_type": soil_type,

            "elevation": elevation,

            "groundwater_depth": groundwater_depth,

            "slope": slope,

            "land_use": land_use

        }


        # ----------------------------------------------------
        # Determine CSS class
        # ----------------------------------------------------

        if prediction == "High":

            prediction_class = "high"

        elif prediction == "Medium":

            prediction_class = "medium"

        else:

            prediction_class = "low"


        # ----------------------------------------------------
        # Display prediction result
        # ----------------------------------------------------

        return render_template(

            "index.html",

            prediction=prediction,

            confidence=confidence,

            recommendation=recommendation,

            input_summary=input_summary,

            prediction_class=prediction_class,

            selected_latitude=latitude,

            selected_longitude=longitude,

            selected_elevation=elevation

        )


    except Exception as e:

        print("Prediction error:", e)

        return render_template(

            "index.html",

            error=(
                "Unable to process the prediction. "
                "Please check your inputs and try again."
            )

        )


# ============================================================
# 6. RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)