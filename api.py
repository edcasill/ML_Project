from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import jax.numpy as jnp
import json
import os

app = FastAPI(title="Kepler Exoplanet API")

# Configure CORS to avoid pyscript be blocked by opera, google, etc
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"],)

# Load data generated ny trained models
def load_all():
    metrics = {}
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)

    scaler_mean = jnp.load("scaler_mean.npy") if os.path.exists("scaler_mean.npy") else None
    scaler_std = jnp.load("scaler_std.npy") if os.path.exists("scaler_std.npy") else None

    model = None
    if os.path.exists("tree_model.pkl"):
        with open("tree_model.pkl", "rb") as f:
            model = pickle.load(f)

    return metrics, scaler_mean, scaler_std, model

ALL_METRICS, MEAN, STD, TREE_MODEL = load_all()

#dataset structure
class AstroData(BaseModel):
    koi_period: float; koi_time0bk: float; koi_impact: float; koi_duration: float
    koi_depth: float; koi_prad: float; koi_teff: float; koi_insol: float
    koi_model_snr: float; koi_tce_plnt_num: float; koi_steff: float; koi_slogg: float
    koi_srad: float; ra: float; dec: float; koi_kepmag: float

# endpoints
@app.get("/metrics")
async def get_metrics():
    """
    Return the metrics from the trained models
    """
    # if we run the pipeline again, reload the file
    metrics, _, _, _ = load_all()
    if not metrics:
        return []

    # format the list for the front end
    return [{"name": k, "accuracy": v["accuracy"], "f1": v["f1"]} for k, v in metrics.items()]

@app.post("/predict")
async def predict(data: AstroData):
    """
    Receive the 16 data, normilice them and use the desicion tree to generate a prediction
    """
    if TREE_MODEL is None or MEAN is None or STD is None:
        raise HTTPException(status_code=500, detail="There are missing training files, please run the pipeline.")

    # extract values, normalice them and predict
    input_vals = jnp.array([list(data.dict().values())])
    X_scaled = (input_vals - MEAN) / STD
    pred = TREE_MODEL.predict(X_scaled)[0]

    return {"prediction": int(pred),
            "label": "🪐 IS AN EXOPLANET!" if pred == 1 else "⭐ FALSE POSITIVE (Star/Noise)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)