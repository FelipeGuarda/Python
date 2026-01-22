# 🚀 START HERE - ML Full Training

**You want to:** Train the ML model on the full dataset (616 fires)

**You have:** 2-3 hours available

**You need:** API key from https://datospararesiliencia.cl

---

## ✅ Current Status (Already Done)

✅ Conda environment created (`fire_risk_dashboard`)  
✅ All dependencies installed  
✅ Scripts configured for full training (`MAX_SAMPLES = None`)  
✅ Data directory created  
✅ Model trained on 100 samples (working but small)  

---

## 📋 What You Need to Do (4 Steps)

### Step 1: Get API Key (5 minutes)

1. Go to: https://datospararesiliencia.cl
2. Log in (or create account)
3. Profile → Developer Tools → Generate API key
4. Copy the key

---

### Step 2: Download Fire Dataset (2 minutes)

Open terminal and run:

```bash
cd "/home/fguarda/Dev/Python/Estacion meteorologica/Fire risk dashboard/ml_model"
conda activate fire_risk_dashboard

# Replace YOUR_API_KEY with actual key
./download_fire_data.sh YOUR_API_KEY
```

**Expected:** File downloaded to `ml_model/data/cicatrices_incendios_resumen.geojson`

---

### Step 3: Prepare Training Data (1-2 hours)

⚠️ **This is the long step!** Can run in background.

```bash
python prepare_training_data.py
```

**What it does:**
- Processes 616 fires from Araucania
- Fetches historical weather for each fire
- Creates balanced dataset (616 fires + 616 non-fires)

**Progress:** You'll see updates every 50 samples

💡 **Tip:** Start this before dinner/bed, continue when done!

---

### Step 4: Train Model (5 minutes)

```bash
python train_fire_model.py
```

**Expected output:**
- Model trained
- Accuracy: 80-85%
- Plots generated in `plots/` folder

---

### Step 5: Test Dashboard (2 minutes)

```bash
cd ..
streamlit run app.py
```

Open browser → Check you see **both**:
- Rule-Based Risk Index
- ML Fire Probability

---

## 🎯 Quick Reference

| File | Purpose |
|------|---------|
| `check_environment.py` | Verify everything is ready |
| `download_fire_data.sh` | Download dataset (use this!) |
| `prepare_training_data.py` | Process fires → training data |
| `train_fire_model.py` | Train model → fire_model.pkl |
| `RUN_FULL_TRAINING.md` | Detailed guide with troubleshooting |
| `QUICK_REFERENCE.md` | Commands cheat sheet |

---

## 📞 Need Help?

**Check environment first:**
```bash
python check_environment.py
```

**See detailed guide:**
- Read `RUN_FULL_TRAINING.md` for troubleshooting

---

## 🎓 Why This Matters

Current model: Trained on **100 fires** (fast testing)  
New model: Trained on **616 fires** (production quality)

**Benefits:**
- More accurate predictions
- Better generalization
- Higher confidence
- Validates your rule-based approach with data

---

**Ready? Get your API key and start with Step 2!** 🔥

---

**Estimated timeline:**
- Now: Get API key (5 min)
- Now: Download dataset (2 min)
- Now: Start training preparation (leave running 1-2 hours)
- Later: Train model (5 min)
- Later: Test dashboard (2 min)
- **Done!** ✨
