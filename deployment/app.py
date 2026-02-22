
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Download model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="AkhilRaja/final-report-best-model",
    filename="best_model.joblib"
)

model = joblib.load(model_path)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
