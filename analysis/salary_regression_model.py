"""
CareerCompass — Step 3: Salary Regression Model
====================================================
BHAI, YE FILE KYA KARTI HAI:

Ek regression model banata hai jo salary predict karta hai — LEKIN
sklearn ki jagah STATSMODELS use kiya hai, jo sirf "prediction" nahi
deta, balki:
  - Har factor ka EXACT coefficient (jaise "1 experience-year se
    salary +X LPA badhti hai")
  - p-value har coefficient ke liye (kya ye factor GENUINELY
    matter karta hai, ya statistically-insignificant hai)
  - R² (poora model kitna variance explain karta hai)

Ye "interpretable regression" hai — Data-Analyst interviews me
isi tarah ke models expect kiye jaate hain, sirf accuracy-number nahi.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json


def load_and_prepare(path="job_postings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    HIGH_VALUE_SKILLS = {"Machine Learning", "Deep Learning", "AWS", "Kubernetes", "MLOps", "Spark"}
    df["n_high_value_skills"] = df["skills_required"].apply(
        lambda s: len(set(x.strip() for x in s.split(",")) & HIGH_VALUE_SKILLS)
    )

    tier_map = {"Tier-1": 3, "Tier-2": 2, "Tier-3": 1}
    df["city_tier_ordinal"] = df["city_tier"].map(tier_map)
    df["remote_int"] = df["remote"].astype(int)

    return df


if __name__ == "__main__":
    df = load_and_prepare()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # --- Statsmodels OLS: interpretable coefficients + p-values ---
    formula = "salary_lpa ~ experience_years_mid + n_high_value_skills + city_tier_ordinal + remote_int + C(company_size)"
    model = smf.ols(formula=formula, data=train_df).fit()

    print("=" * 65)
    print("REGRESSION SUMMARY (trained on 80% of data)")
    print("=" * 65)
    print(model.summary())

    print("\n" + "=" * 65)
    print("PLAIN-ENGLISH INTERPRETATION OF COEFFICIENTS")
    print("=" * 65)
    for factor, coef in model.params.items():
        p_val = model.pvalues[factor]
        sig = "significant" if p_val < 0.05 else "NOT significant"
        print(f"  {factor}: coefficient = {coef:.3f}  (p={p_val:.2e}, {sig})")

    # --- Evaluate on held-out 20% test set (real predictive check) ---
    test_predictions = model.predict(test_df)
    r2 = r2_score(test_df["salary_lpa"], test_predictions)
    mae = mean_absolute_error(test_df["salary_lpa"], test_predictions)
    rmse = np.sqrt(mean_squared_error(test_df["salary_lpa"], test_predictions))

    print("\n" + "=" * 65)
    print("TEST-SET PERFORMANCE (unseen 20% data)")
    print("=" * 65)
    print(f"  R² on test set: {r2:.3f}")
    print(f"  MAE: {mae:.2f} LPA  (average prediction error)")
    print(f"  RMSE: {rmse:.2f} LPA")

    # Save results for dashboard
    results = {
        "train_r_squared": model.rsquared,
        "test_r_squared": r2,
        "test_mae": mae,
        "test_rmse": rmse,
        "coefficients": {k: {"value": v, "p_value": model.pvalues[k]} for k, v in model.params.items()},
    }
    with open("regression_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved -> regression_results.json")
