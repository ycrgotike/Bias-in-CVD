from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        data = CustomData(
            Age=int(request.form.get("Age")),
            Sex=request.form.get("Sex"),
            ChestPainType=request.form.get("ChestPainType"),
            RestingBP=int(request.form.get("RestingBP")),
            Cholesterol=int(request.form.get("Cholesterol")),
            FastingBS=int(request.form.get("FastingBS")),
            RestingECG=request.form.get("RestingECG"),
            MaxHR=int(request.form.get("MaxHR")),
            ExerciseAngina=request.form.get("ExerciseAngina"),
            Oldpeak=float(request.form.get("Oldpeak")),
            ST_Slope=request.form.get("ST_Slope")
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        prediction = int(results[0])  # 0 or 1

        print(f"Prediction: {prediction}")
        return render_template('home.html', results=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)