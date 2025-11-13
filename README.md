## Files

* `api.py` — main FastAPI application. Must define `app = FastAPI()` at module level and expose the inference endpoint.
* `schema.py` — pydantic models used for request validation and response serialization. Update these models to match the inputs/outputs your model expects.
* `requirements.txt` — list of pip dependencies (FastAPI, uvicorn, pydantic, plus any ML libs like `torch`, `tensorflow`, `scikit-learn`).

---

## Design principles

1. **Keep schema simple and explicit**

   * `schema.py` currently defines `CropPriceInput` and `CropPriceOutput`. If your model expects different fields, rename or create new pydantic models that reflect the exact inputs and outputs. Example:

   ```python
   from pydantic import BaseModel

   class CropPriceInput(BaseModel):
       crop: str
       region: str
       date: str  # ISO format

   class CropPriceOutput(BaseModel):
       crop: str
       region: str
       date: str
       price: float
   ```

   * Pydantic will validate incoming JSON and produce useful 422 errors when the payload is malformed.

2. **Load the model once, at module import time**

   * Load or initialize the model in `api.py` at module level so it is reused across requests instead of being loaded per-request. Use environment variables or configuration to point to model files or endpoints.

   ```python
   import os
   MODEL_PATH = os.environ.get("MODEL_PATH")
   model = load_model(MODEL_PATH)  # implement load_model for your model type
   ```

3. **Keep a small adapter function**

   * Implement a single adapter function (already present as `predict_crop_price`) that accepts the validated fields from the pydantic model, transforms them as needed, runs model inference, and returns a primitive (number or dict) that the route can convert into the response model.

   ```python
   def predict_crop_price(crop: str, region: str, date: str = None):
       # transform inputs, call model, post-process output
       return 123.45  # numeric or convertible to float
   ```

4. **Endpoint matches schema**

   * The endpoint should accept the input model and return the output model using `response_model=...` so responses are validated and documented automatically by FastAPI.

   ```python
   @app.post("/crop_price", response_model=CropPriceOutput)
   async def predict(data: CropPriceInput):
       price = predict_crop_price(data.crop, data.region, data.date)
       return CropPriceOutput(...)
   ```


---

## Quick start (local development)

1. `(optional)` Create and activate a virtual environment:

```bash
# Step 1: Create a new virtual environment named ".venv"
python -m venv .venv

# Step 2: Activate the virtual environment (Windows PowerShell or Git Bash)
./.venv/Scripts/activate

# Step 3: Export all installed libraries to a requirements file
pip freeze > requirements.txt

```

2. Install dependencies (adjust for your model runtime):

```bash
# Install all required dependencies
pip install -r requirements.txt
```

3. Run the app from the directory with `api.py`:

```bash
uvicorn api:app --reload --port 8000
```

4. Open interactive docs to test inputs and outputs:

* `http://127.0.0.1:8000/docs` (Swagger UI)
* `http://127.0.0.1:8000/redoc` (ReDoc)

---

## Example request (JSON)

Use the input shape defined in `schema.py`. Example based on the current schema:

```json
{
  "crop": "wheat",
  "region": "kolhapur",
  "date": "2025-11-13"
}
```

Example response:

```json
{
  "crop": "wheat",
  "region": "kolhapur",
  "date": "2025-11-13",
  "price": 123.45
}
```

---

## Common pitfalls

* **Model reloaded on each request:** Load the model once at module import time.
* **Non-serializable output:** Ensure outputs are primitives (float, int, str) or pydantic models.
* **Missing `app` object:** `uvicorn api:app` requires `app = FastAPI()` in `api.py`.
* **422 Unprocessable Entity:** Request JSON doesn't match the input pydantic model.
* **Port in use:** Run on a different port with `--port`.

---

## For source code of Crop Prediction Model
```
https://github.com/SwapCodesDev/ML-Models---Farmingo
```