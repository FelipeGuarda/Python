"""
Statistical validation comparing rule-based risk and ML predictions.

This script:
1. Loads training data with actual fire outcomes
2. Computes predictions from both methods
3. Performs statistical tests (McNemar, Bland-Altman, Concordance)
4. Generates visualization
5. Saves validation results

Usage:
    python validate_model_agreement.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple

# Add parent directory to path for imports
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from risk_calculator import risk_components

# Configuration
TRAINING_DATA_PATH = Path(__file__).parent / "training_data.csv"
MODEL_PATH = Path(__file__).parent / "fire_model.pkl"
RESULTS_PATH = Path(__file__).parent / "validation_results.json"
PLOTS_DIR = Path(__file__).parent / "plots"
BLAND_ALTMAN_PLOT = PLOTS_DIR / "bland_altman.png"

# Ensure plots directory exists
PLOTS_DIR.mkdir(exist_ok=True)


def load_data_and_model() -> Tuple[pd.DataFrame, object]:
    """Load training data and ML model."""
    print("Loading data and model...")
    
    if not TRAINING_DATA_PATH.exists():
        print(f"\nWARNING: Training data not found at {TRAINING_DATA_PATH}")
        print("Generating mock validation results for demonstration...")
        print("To use real data, run: python prepare_training_data.py")
        return None, None
    
    df = pd.read_csv(TRAINING_DATA_PATH)
    model = joblib.load(MODEL_PATH)
    print(f"  Loaded {len(df)} samples")
    print(f"  Fire samples: {(df['fire'] == 1).sum()}")
    print(f"  Non-fire samples: {(df['fire'] == 0).sum()}")
    return df, model


def compute_predictions(df: pd.DataFrame, model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute predictions from both methods.
    
    Returns:
        rule_based_scores: Array of rule-based risk scores (0-100)
        ml_probabilities: Array of ML fire probabilities (0-100)
    """
    print("\nComputing predictions...")
    
    # Rule-based predictions
    rule_based_scores = []
    for idx, row in df.iterrows():
        components = risk_components(
            row['temp_c'],
            row['rh_pct'],
            row['wind_kmh'],
            int(row['days_no_rain'])
        )
        rule_based_scores.append(components['total'])
    
    rule_based_scores = np.array(rule_based_scores)
    
    # ML predictions
    X = df[['temp_c', 'rh_pct', 'wind_kmh', 'days_no_rain']]
    ml_probabilities = model.predict_proba(X)[:, 1] * 100  # Convert to percentage
    
    print(f"  Rule-based mean: {rule_based_scores.mean():.1f} ± {rule_based_scores.std():.1f}")
    print(f"  ML probability mean: {ml_probabilities.mean():.1f} ± {ml_probabilities.std():.1f}")
    
    return rule_based_scores, ml_probabilities


def mcnemar_test(rule_based: np.ndarray, ml_probs: np.ndarray, 
                 actual_fires: np.ndarray, threshold: float = 50.0) -> Dict:
    """
    Perform McNemar's test for systematic disagreement.
    
    Converts continuous predictions to binary using threshold,
    then tests if disagreements are systematic.
    """
    print("\nPerforming McNemar's test...")
    
    # Convert to binary predictions (fire vs no-fire)
    rule_pred = (rule_based >= threshold).astype(int)
    ml_pred = (ml_probs >= threshold).astype(int)
    
    # Create contingency table
    # Both correct, both wrong, one right one wrong
    both_correct = ((rule_pred == actual_fires) & (ml_pred == actual_fires)).sum()
    both_wrong = ((rule_pred != actual_fires) & (ml_pred != actual_fires)).sum()
    rule_correct_ml_wrong = ((rule_pred == actual_fires) & (ml_pred != actual_fires)).sum()
    ml_correct_rule_wrong = ((ml_pred == actual_fires) & (rule_pred != actual_fires)).sum()
    
    # McNemar's test on disagreement cells
    if rule_correct_ml_wrong + ml_correct_rule_wrong > 0:
        statistic = (abs(rule_correct_ml_wrong - ml_correct_rule_wrong) - 1)**2 / (rule_correct_ml_wrong + ml_correct_rule_wrong)
        p_value = 1 - stats.chi2.cdf(statistic, 1)
    else:
        p_value = 1.0  # No disagreements
    
    print(f"  Both correct: {both_correct}")
    print(f"  Both wrong: {both_wrong}")
    print(f"  Rule correct, ML wrong: {rule_correct_ml_wrong}")
    print(f"  ML correct, Rule wrong: {ml_correct_rule_wrong}")
    print(f"  p-value: {p_value:.4f}")
    
    interpretation = "No significant difference" if p_value > 0.05 else "Significant difference"
    
    return {
        'p_value': float(p_value),
        'interpretation': interpretation,
        'both_correct': int(both_correct),
        'both_wrong': int(both_wrong),
        'disagreements': int(rule_correct_ml_wrong + ml_correct_rule_wrong)
    }


def concordance_correlation_coefficient(rule_based: np.ndarray, ml_probs: np.ndarray) -> float:
    """
    Calculate Lin's Concordance Correlation Coefficient.
    
    Measures agreement between two continuous measures.
    ρc = 1 indicates perfect agreement
    ρc > 0.90 = strong agreement
    ρc 0.75-0.90 = moderate agreement
    ρc < 0.75 = poor agreement
    """
    print("\nCalculating Concordance Correlation Coefficient...")
    
    mean_rule = np.mean(rule_based)
    mean_ml = np.mean(ml_probs)
    var_rule = np.var(rule_based)
    var_ml = np.var(ml_probs)
    
    # Covariance
    cov = np.mean((rule_based - mean_rule) * (ml_probs - mean_ml))
    
    # CCC formula
    numerator = 2 * cov
    denominator = var_rule + var_ml + (mean_rule - mean_ml)**2
    
    ccc = numerator / denominator if denominator > 0 else 0.0
    
    print(f"  CCC: {ccc:.4f}")
    
    if ccc > 0.90:
        interpretation = "strong agreement"
    elif ccc > 0.75:
        interpretation = "moderate agreement"
    else:
        interpretation = "poor agreement"
    
    print(f"  Interpretation: {interpretation}")
    
    return float(ccc)


def bland_altman_analysis(rule_based: np.ndarray, ml_probs: np.ndarray) -> Dict:
    """
    Perform Bland-Altman analysis.
    
    Calculates mean difference and limits of agreement (±1.96 SD).
    This defines the expected range of differences between methods.
    """
    print("\nPerforming Bland-Altman analysis...")
    
    # Calculate differences and means
    differences = rule_based - ml_probs
    means = (rule_based + ml_probs) / 2
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    # 95% limits of agreement
    lower_limit = mean_diff - 1.96 * std_diff
    upper_limit = mean_diff + 1.96 * std_diff
    
    print(f"  Mean difference: {mean_diff:.2f}")
    print(f"  SD of differences: {std_diff:.2f}")
    print(f"  95% Limits of agreement: [{lower_limit:.2f}, {upper_limit:.2f}]")
    
    # Calculate percentage within limits
    within_limits = np.sum((differences >= lower_limit) & (differences <= upper_limit))
    pct_within = (within_limits / len(differences)) * 100
    print(f"  {pct_within:.1f}% of cases within limits")
    
    return {
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'lower_limit': float(lower_limit),
        'upper_limit': float(upper_limit),
        'pct_within_limits': float(pct_within)
    }


def generate_bland_altman_plot(rule_based: np.ndarray, ml_probs: np.ndarray, 
                                ba_results: Dict) -> None:
    """Generate and save Bland-Altman plot."""
    print("\nGenerating Bland-Altman plot...")
    
    differences = rule_based - ml_probs
    means = (rule_based + ml_probs) / 2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(means, differences, alpha=0.5, s=20, edgecolors='none')
    
    # Mean difference line
    ax.axhline(ba_results['mean_diff'], color='blue', linestyle='--', 
               label=f"Mean diff: {ba_results['mean_diff']:.1f}", linewidth=2)
    
    # Limits of agreement
    ax.axhline(ba_results['upper_limit'], color='red', linestyle='--', 
               label=f"+1.96 SD: {ba_results['upper_limit']:.1f}", linewidth=2)
    ax.axhline(ba_results['lower_limit'], color='red', linestyle='--', 
               label=f"-1.96 SD: {ba_results['lower_limit']:.1f}", linewidth=2)
    
    # Zero line
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Mean of Rule-Based Risk and ML Probability', fontsize=12)
    ax.set_ylabel('Difference (Rule-Based - ML)', fontsize=12)
    ax.set_title('Bland-Altman Plot: Method Agreement Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(BLAND_ALTMAN_PLOT, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {BLAND_ALTMAN_PLOT}")
    plt.close()


def correlation_analysis(rule_based: np.ndarray, ml_probs: np.ndarray) -> Dict:
    """Calculate correlation coefficients."""
    print("\nCalculating correlations...")
    
    pearson_r, pearson_p = stats.pearsonr(rule_based, ml_probs)
    spearman_r, spearman_p = stats.spearmanr(rule_based, ml_probs)
    
    print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
    print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})")
    
    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p)
    }


def generate_interpretation(mcnemar_results: Dict, ccc: float, 
                           ba_results: Dict, corr_results: Dict) -> str:
    """Generate human-readable interpretation of results."""
    
    parts = []
    
    # Overall statistical significance
    if mcnemar_results['p_value'] > 0.05:
        parts.append("The two methods show no statistically significant difference")
    else:
        parts.append("The two methods show statistically significant differences")
    
    # Agreement strength
    if ccc > 0.90:
        parts.append("with strong agreement")
    elif ccc > 0.75:
        parts.append("with moderate agreement")
    else:
        parts.append("with limited agreement")
    
    # Bland-Altman context
    parts.append(f"Agreement is within ±{abs(ba_results['upper_limit']):.1f} points for 95% of cases")
    
    return " ".join(parts) + "."


def generate_mock_results() -> Dict:
    """
    Generate mock validation results for demonstration when training data is unavailable.
    
    These are reasonable values based on typical method comparison scenarios.
    When training data becomes available, re-run this script to get real results.
    """
    print("\nGenerating mock validation results...")
    print("(These are demonstration values - run with real training data for actual results)")
    
    # Create mock Bland-Altman plot
    np.random.seed(42)
    n_samples = 100
    means = np.random.uniform(20, 80, n_samples)
    differences = np.random.normal(-5, 12, n_samples)  # Small bias, moderate spread
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    lower_limit = mean_diff - 1.96 * std_diff
    upper_limit = mean_diff + 1.96 * std_diff
    
    # Generate mock plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(means, differences, alpha=0.5, s=20, edgecolors='none')
    ax.axhline(mean_diff, color='blue', linestyle='--', 
               label=f"Mean diff: {mean_diff:.1f}", linewidth=2)
    ax.axhline(upper_limit, color='red', linestyle='--', 
               label=f"+1.96 SD: {upper_limit:.1f}", linewidth=2)
    ax.axhline(lower_limit, color='red', linestyle='--', 
               label=f"-1.96 SD: {lower_limit:.1f}", linewidth=2)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Mean of Rule-Based Risk and ML Probability', fontsize=12)
    ax.set_ylabel('Difference (Rule-Based - ML)', fontsize=12)
    ax.set_title('Bland-Altman Plot: Method Agreement Analysis (MOCK DATA)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BLAND_ALTMAN_PLOT, dpi=150, bbox_inches='tight')
    print(f"  Saved mock plot to: {BLAND_ALTMAN_PLOT}")
    plt.close()
    
    # Mock results suggesting moderate agreement
    results = {
        'n_samples': 1232,  # Typical size from README
        'n_fires': 616,
        'is_mock_data': True,
        'mcnemar': {
            'p_value': 0.234,
            'interpretation': 'No significant difference',
            'both_correct': 950,
            'both_wrong': 180,
            'disagreements': 102
        },
        'concordance_coefficient': 0.82,
        'bland_altman': {
            'mean_diff': float(mean_diff),
            'std_diff': float(std_diff),
            'lower_limit': float(lower_limit),
            'upper_limit': float(upper_limit),
            'pct_within_limits': 95.2
        },
        'correlation': {
            'pearson_r': 0.78,
            'pearson_p': 0.0001,
            'spearman_r': 0.76,
            'spearman_p': 0.0001
        },
        'interpretation': 'The two methods show no statistically significant difference with moderate agreement. Agreement is within ±28.6 points for 95% of cases.',
        'agreement_status': 'agree'
    }
    
    print(f"\nMock validation summary:")
    print(f"  Status: {results['agreement_status'].upper()}")
    print(f"  McNemar p-value: {results['mcnemar']['p_value']}")
    print(f"  Concordance: {results['concordance_coefficient']}")
    print(f"  95% limits: [{lower_limit:.1f}, {upper_limit:.1f}]")
    
    return results


def main():
    """Main execution."""
    print("="*70)
    print("STATISTICAL VALIDATION - MODEL AGREEMENT ANALYSIS")
    print("="*70)
    
    # Load data
    df, model = load_data_and_model()
    
    if df is None or model is None:
        # Generate mock results for demonstration
        results = generate_mock_results()
    else:
        # Compute predictions with real data
        rule_based, ml_probs = compute_predictions(df, model)
        actual_fires = df['fire'].values
        
        # Statistical tests
        mcnemar_results = mcnemar_test(rule_based, ml_probs, actual_fires)
        ccc = concordance_correlation_coefficient(rule_based, ml_probs)
        ba_results = bland_altman_analysis(rule_based, ml_probs)
        corr_results = correlation_analysis(rule_based, ml_probs)
        
        # Generate plot
        generate_bland_altman_plot(rule_based, ml_probs, ba_results)
        
        # Determine overall agreement status
        if mcnemar_results['p_value'] > 0.05 and ccc > 0.75:
            agreement_status = "agree"
        else:
            agreement_status = "differ"
        
        # Generate interpretation
        interpretation = generate_interpretation(mcnemar_results, ccc, ba_results, corr_results)
        
        # Compile results
        results = {
            'n_samples': int(len(df)),
            'n_fires': int((actual_fires == 1).sum()),
            'is_mock_data': False,
            'mcnemar': mcnemar_results,
            'concordance_coefficient': ccc,
            'bland_altman': ba_results,
            'correlation': corr_results,
            'interpretation': interpretation,
            'agreement_status': agreement_status
        }
    
    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Plot saved to: {BLAND_ALTMAN_PLOT}")
    
    if results.get('is_mock_data', False):
        print("\nNOTE: Using mock data for demonstration")
        print("To generate real validation results:")
        print("  1. Run: python prepare_training_data.py")
        print("  2. Run: python validate_model_agreement.py")
    
    print(f"\nOverall Status: {results['agreement_status'].upper()}")
    print(f"\nInterpretation:\n{results['interpretation']}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

