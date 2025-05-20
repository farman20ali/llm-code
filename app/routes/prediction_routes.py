from flask import Blueprint, request, jsonify, current_app
from openai import OpenAI
import os
import json
import pandas as pd
import joblib

# Create blueprint
prediction_bp = Blueprint('prediction', __name__)

# Initialize predictor
class AccidentPredictor:
    def __init__(self):
        pass

    def initialize(self):
        """Lazy initialization of the predictor"""
        if not hasattr(self, 'xgb'):
            # Get the app's root path
            app_root = current_app.root_path
            
            # Define paths for model and data files
            data_dir = os.path.join(app_root, 'data', 'prediction')
            models_dir = os.path.join(app_root, 'models', 'prediction')
            
            # Load the final dataset and weather data
            self.df = pd.read_csv(os.path.join(data_dir, "accident_severity_model_ready.csv"))
            self.weather_cleaned = pd.read_csv(os.path.join(data_dir, "weather.csv"))
            self.weather_cleaned["datetime"] = pd.to_datetime(self.weather_cleaned["datetime"])
            self.weather_cleaned["datetime_hour"] = self.weather_cleaned["datetime"].dt.floor("H")

            # Load the trained model and encoder
            self.xgb = joblib.load(os.path.join(models_dir, "xgb_model.pkl"))
            self.encoder = joblib.load(os.path.join(models_dir, "encoder.pkl"))

    def get_time_of_day(self, hour):
        if 5 <= hour < 12: return "morning"
        elif 12 <= hour < 17: return "afternoon"
        elif 17 <= hour < 21: return "evening"
        else: return "night"

    def predict(self, area, datetime_str):
        # Ensure model is initialized
        self.initialize()
        
        dt = pd.to_datetime(datetime_str)
        hour = dt.hour
        day_of_week = dt.dayofweek
        is_weekend = int(day_of_week in [5, 6])
        time_of_day = self.get_time_of_day(hour)
        datetime_hour = dt.floor("H")

        # Get area stats
        area_stats = self.df[self.df["CombinedArea"] == area][["CombinedArea_freq", "CombinedArea_risk"]].drop_duplicates()
        if area_stats.empty:
            return f"❌ Area '{area}' not found in training data."

        area_freq = area_stats["CombinedArea_freq"].values[0]
        area_risk = area_stats["CombinedArea_risk"].values[0]

        # Get weather for that datetime
        weather_row = self.weather_cleaned[self.weather_cleaned["datetime_hour"] == datetime_hour]
        if weather_row.empty:
            return f"❌ No weather data for datetime '{datetime_hour}'."
        weather = weather_row.iloc[0]

        # Build input row
        input_data = pd.DataFrame([{
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "CombinedArea_freq": area_freq,
            "CombinedArea_risk": area_risk,
            "time_of_day": time_of_day,
            "conditions": weather["conditions"]
        }])

        # Transform and predict
        X_input = self.encoder.transform(input_data)
        probs = self.xgb.predict_proba(X_input)[0]
        pred_class = self.xgb.predict(X_input)[0]

        severity_map = {0: "🟢 No Accident", 1: "🟠 Injured", 2: "🔴 Fatal"}
        result = {
            "🗺️ Area": area,
            "🕓 Time": datetime_str,
            "🎯 Prediction": severity_map[pred_class],
            "📊 Probabilities": {
                "No Accident": round(probs[0], 2),
                "Injured": round(probs[1], 2),
                "Death": round(probs[2], 2)
            }
        }
        return result

# Create a single instance of the predictor
_predictor = None

def get_predictor():
    """Get or create the predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = AccidentPredictor()
    return _predictor

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'area' not in data or 'datetime' not in data:
            return jsonify({
                "error": "Please provide 'area' and 'datetime' in the request body."
            }), 400
        
        predictor = get_predictor()
        result = predictor.predict(data['area'], data['datetime'])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@prediction_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                "error": "Please provide a 'question' in the request body."
            }), 400

        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are an assistant that predicts accident severity using a function."},
                {"role": "user", "content": data['question']}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "predict_severity",
                    "description": "Predicts accident severity for a given area and datetime",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "area": {"type": "string"},
                            "datetime_str": {"type": "string"}
                        },
                        "required": ["area", "datetime_str"]
                    }
                }
            }],
            tool_choice="auto"
        )

        message = response.choices[0].message
        if message.tool_calls:
            args = json.loads(message.tool_calls[0].function.arguments)
            predictor = get_predictor()
            result = predictor.predict(
                area=args["area"],
                datetime_str=args["datetime_str"]
            )
            return jsonify({"prediction_result": result})
        else:
            return jsonify({"gpt_response": message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500 