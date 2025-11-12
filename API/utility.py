import joblib
import cloudpickle
import os

def load_pickle_model(model_path: str):
    """
    Loads a trained ML model from the given file path using joblib or cloudpickle.

    Returns:
        model object if file exists and loads successfully
        None if file does not exist or load fails
    """
    if not os.path.exists(model_path):
        return None

    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"⚠️ Joblib load failed ({e}), trying cloudpickle...")
        try:
            with open(model_path, "rb") as f:
                return cloudpickle.load(f)
        except Exception as e2:
            print(f"❌ Cloudpickle load failed: {e2}")
            return None
