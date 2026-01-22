# Full ML Model Training Guide

**Estimated time: 2-3 hours total**

---

## Prerequisites

✅ Already done:
- Conda environment `fire_risk_dashboard` created
- Dependencies installed
- Scripts configured (`MAX_SAMPLES = None`)
- Data directory created

⚠️ **You need:**
- Dataverse API key from https://datospararesiliencia.cl

---

## Step 1: Get Your Dataverse API Key (5 minutes)

1. Visit https://datospararesiliencia.cl
2. Log in or create an account
3. Click on your profile (top right)
4. Go to **Developer Tools**
5. Click **Generate API key**
6. Copy the key (you'll use it in the next step)

---

## Step 2: Download Fire Dataset (2-5 minutes)

**Option A: Using the helper script (easiest)**

```bash
cd "/home/fguarda/Dev/Python/Estacion meteorologica/Fire risk dashboard"

# Activate environment
conda activate fire_risk_dashboard

# Run download script (replace YOUR_API_KEY with actual key)
cd ml_model
./download_fire_data.sh YOUR_API_KEY
```

**Option B: Using download_dataverse.py directly**

```bash
cd "/home/fguarda/Dev/Python/Estacion meteorologica/Fire risk dashboard"
conda activate fire_risk_dashboard

python download_dataverse.py \
    --api-key YOUR_API_KEY \
    --dataset-id "doi:10.71578/XAZAKP" \
    --files "cicatrices_incendios_resumen.geojson" \
    --output-dir "ml_model/data"
```

**Expected output:**
- File saved to: `ml_model/data/cicatrices_incendios_resumen.geojson`
- Size: ~50-100 MB
- Contains: 12,250 fires from Chile (2015-2024)

---

## Step 3: Prepare Training Data (1-2 hours)

This is the **longest step** because it fetches historical weather data for each fire from Open-Meteo API.

```bash
cd "/home/fguarda/Dev/Python/Estacion meteorologica/Fire risk dashboard/ml_model"
conda activate fire_risk_dashboard

python prepare_training_data.py
```

**What happens:**
1. Loads 616 fires from Araucania region (2015-2024, ≥10 hectares)
2. For each fire, fetches weather data from 30 days before to ignition date
3. Computes features: temp_c, rh_pct, wind_kmh, days_no_rain
4. Generates 616 non-fire samples (balanced dataset)
5. Saves to `training_data.csv`

**Progress indicators:**
```
Processing 616 fire samples...
  Progress: 0/616 fires processed...
  Progress: 50/616 fires processed...
  Progress: 100/616 fires processed...
  ...
  ✓ Successfully processed 616 fire samples

Generating 616 negative samples...
  Generated 100/616 negative samples...
  Generated 200/616 negative samples...
  ...
  ✓ Generated 616 negative samples

Processing 616 non-fire samples...
  ...
```

**Expected output:**
- `training_data.csv` (1,232 rows = 616 fires + 616 non-fires)
- Balanced dataset: 50% fire / 50% non-fire

**Time estimate:** 1-2 hours (API rate-limited to ~0.1s per request)

⚠️ **Important:**
- The script includes rate limiting (0.1s per request) to be nice to Open-Meteo API
- You can leave this running in the background
- If interrupted, just run it again (it will restart from beginning)

---

## Step 4: Train Model (5 minutes)

```bash
cd "/home/fguarda/Dev/Python/Estacion meteorologica/Fire risk dashboard/ml_model"
conda activate fire_risk_dashboard

python train_fire_model.py
```

**What happens:**
1. Loads `training_data.csv`
2. Splits into train/test sets (80/20)
3. Trains Random Forest classifier (100 trees)
4. Evaluates performance
5. Generates evaluation plots
6. Retrains on full dataset
7. Saves `fire_model.pkl`

**Expected output:**
```
TRAINING MODEL
====================================
Train set: 985 samples
Test set: 247 samples

Training Random Forest...
  ✓ Model trained

MODEL EVALUATION
====================================
Test Set Metrics:
  Accuracy:  0.800-0.850 (expected)
  Precision: 0.750-0.850
  Recall:    0.750-0.850
  F1 Score:  0.750-0.850
  ROC AUC:   0.850-0.900

Feature Importance:
  1. days_no_rain:    ~0.38 (most important!)
  2. temp_c:          ~0.37
  3. wind_kmh:        ~0.17
  4. rh_pct:          ~0.08

✓ Final model trained on all data
```

**Generated files:**
- `ml_model/fire_model.pkl` (updated, trained on full dataset)
- `ml_model/plots/feature_importance.png`
- `ml_model/plots/confusion_matrix.png`
- `ml_model/plots/roc_curve.png`

---

## Step 5: Test Dashboard Integration (2 minutes)

```bash
cd "/home/fguarda/Dev/Python/Estacion meteorologica/Fire risk dashboard"
conda activate fire_risk_dashboard

streamlit run app.py
```

**What to check:**
1. Dashboard opens in browser (http://localhost:8501)
2. You see **both** risk indicators:
   - Rule-Based Risk Index (0-100)
   - ML Fire Probability (0-100%)
3. Agreement indicator (✓ when methods align)
4. No errors in terminal

**If ML prediction doesn't appear:**
- Check `ml_model/fire_model.pkl` exists
- Check file size > 200 KB
- Restart Streamlit

---

## Summary: Complete Command Sequence

```bash
# Navigate to project
cd "/home/fguarda/Dev/Python/Estacion meteorologica/Fire risk dashboard"

# Activate environment
conda activate fire_risk_dashboard

# Step 1: Download fire data
python download_dataverse.py \
    --api-key YOUR_API_KEY \
    --dataset-id "doi:10.71578/XAZAKP" \
    --files "cicatrices_incendios_resumen.geojson" \
    --output-dir "ml_model/data"

# Step 2: Prepare training data (1-2 hours)
cd ml_model
python prepare_training_data.py

# Step 3: Train model (5 minutes)
python train_fire_model.py

# Step 4: Test dashboard
cd ..
streamlit run app.py
```

---

## Troubleshooting

### "No module named 'pyDataverse'"
```bash
conda activate fire_risk_dashboard
pip install pyDataverse==0.3.3
```

### "API key invalid"
- Check you copied the full key (no spaces)
- Try regenerating the key on datospararesiliencia.cl

### "Training data preparation is slow"
- **This is normal!** Fetching weather for 616 fires takes 1-2 hours
- The script includes 0.1s delays between API calls (rate limiting)
- You can monitor progress in terminal
- Safe to run overnight

### "Model accuracy is lower than expected"
- Check training data CSV has ~1,232 rows
- Check feature values look reasonable (no NaNs)
- Try adjusting Random Forest parameters in `train_fire_model.py`

### "Dashboard shows old predictions"
- Restart Streamlit: Ctrl+C and run `streamlit run app.py` again
- Model is loaded once at startup

---

## Expected Improvements from Full Dataset

**Current model (100 samples):**
- Accuracy: ~80%
- Feature importance: Validated

**Full model (1,232 samples):**
- Accuracy: 80-85% (more stable)
- Better generalization to new data
- More confident predictions
- Reduced variance in cross-validation

---

## For Your Presentation

**Key talking points:**

1. **Data-driven validation**
   - "I trained a Random Forest on 616 real fires from Araucania (2015-2024)"
   - "The ML model learned that days without rain is the most critical factor (38% importance)"
   - "This confirms the Chilean fire danger standards I used"

2. **Model performance**
   - "80-85% accuracy on unseen test data"
   - "ROC AUC of 0.85-0.90 shows good discrimination"
   - "Balanced dataset prevents bias toward fire/non-fire"

3. **Dashboard integration**
   - "Rule-based index uses fixed scoring bins (expert knowledge)"
   - "ML probability learns from actual fire patterns (data-driven)"
   - "When both methods agree, we have high confidence"

**Show these plots:**
- `plots/feature_importance.png` — Validates your approach
- `plots/confusion_matrix.png` — Shows model accuracy
- `plots/roc_curve.png` — Demonstrates discrimination ability
- Dashboard screenshot with both predictions side-by-side

---

## Next Steps After Training

1. **Review evaluation plots** in `ml_model/plots/`
2. **Test dashboard** with different dates
3. **Document model performance** for your report
4. **Take screenshots** for presentation
5. **Consider running validation script**:
   ```bash
   cd ml_model
   python validate_model_agreement.py
   ```

---

**Ready to start!** Just get your API key and run the commands above.

Good luck with your training! 🔥🌲
