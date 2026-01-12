"""
Train Random Forest classifier for fire prediction.

This script:
1. Loads prepared training data
2. Trains a Random Forest model
3. Evaluates performance (accuracy, precision, recall, F1)
4. Generates feature importance visualization
5. Saves trained model for dashboard integration

Usage:
    python train_fire_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Configuration - paths relative to ml_model directory
TRAINING_DATA_PATH = Path(__file__).parent / "training_data.csv"
MODEL_OUTPUT_PATH = Path(__file__).parent / "fire_model.pkl"
PLOTS_DIR = Path(__file__).parent / "plots"
FEATURE_IMPORTANCE_PLOT = PLOTS_DIR / "feature_importance.png"
CONFUSION_MATRIX_PLOT = PLOTS_DIR / "confusion_matrix.png"
ROC_CURVE_PLOT = PLOTS_DIR / "roc_curve.png"

RANDOM_STATE = 42

# Ensure plots directory exists
PLOTS_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load training data and split into features and target."""
    print("Loading training data...")
    df = pd.read_csv(TRAINING_DATA_PATH)
    
    print(f"  Total samples: {len(df)}")
    print(f"  Fire samples: {(df['fire'] == 1).sum()}")
    print(f"  Non-fire samples: {(df['fire'] == 0).sum()}")
    
    # Features: the 4 weather variables
    feature_cols = ['temp_c', 'rh_pct', 'wind_kmh', 'days_no_rain']
    X = df[feature_cols]
    y = df['fire']
    
    print(f"\nFeature columns: {feature_cols}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, feature_cols


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,        # 100 trees
        max_depth=10,            # Prevent overfitting
        min_samples_split=5,     # Require at least 5 samples to split
        min_samples_leaf=2,      # Require at least 2 samples per leaf
        random_state=RANDOM_STATE,
        n_jobs=-1                # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Model trained")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f} (of predicted fires, how many were real?)")
    print(f"  Recall:    {recall:.3f} (of real fires, how many did we catch?)")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  ROC AUC:   {roc_auc:.3f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"                No Fire  Fire")
    print(f"Actual No Fire  {cm[0,0]:7d}  {cm[0,1]:4d}")
    print(f"       Fire     {cm[1,0]:7d}  {cm[1,1]:4d}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Fire', 'Fire']))
    
    # Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-fold)")
    print("="*60)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    feature_names = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nRanked features:")
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names[idx]:15s}: {importances[idx]:.4f}")
    
    # Save visualizations
    save_visualizations(model, X_test, y_test, y_pred, y_pred_proba, feature_names, importances)
    
    # Return trained model (trained on train set for evaluation, but we'll retrain on full data)
    print("\n" + "="*60)
    print("RETRAINING ON FULL DATASET")
    print("="*60)
    
    final_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_model.fit(X, y)
    print("  ✓ Final model trained on all data")
    
    return final_model


def save_visualizations(model, X_test, y_test, y_pred, y_pred_proba, feature_names, importances):
    """Generate and save evaluation plots."""
    print("\nGenerating visualizations...")
    
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance for Fire Prediction')
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {FEATURE_IMPORTANCE_PLOT}")
    plt.close()
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No Fire', 'Fire'])
    plt.yticks(tick_marks, ['No Fire', 'Fire'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]}',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {CONFUSION_MATRIX_PLOT}")
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PLOT, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {ROC_CURVE_PLOT}")
    plt.close()


def save_model(model):
    """Save trained model to disk."""
    print(f"\nSaving model to {MODEL_OUTPUT_PATH}...")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"  ✓ Model saved")


def main():
    """Main execution."""
    print("="*60)
    print("FIRE PREDICTION MODEL - TRAINING")
    print("="*60)
    print()
    
    # Load data
    X, y, feature_cols = load_data()
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    save_model(model)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - {MODEL_OUTPUT_PATH} (trained model)")
    print(f"  - {FEATURE_IMPORTANCE_PLOT}")
    print(f"  - {CONFUSION_MATRIX_PLOT}")
    print(f"  - {ROC_CURVE_PLOT}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review evaluation plots to assess model quality")
    print("2. If satisfied, integrate into dashboard:")
    print("   - Model will load from fire_model.pkl")
    print("   - Dashboard will use same 4 features")
    print("   - Predictions will show as fire probability (0-100%)")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

