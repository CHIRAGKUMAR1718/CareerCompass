"""
CareerCompass — Step 2: EDA + Statistical Analysis
========================================================
BHAI, YE FILE KYA KARTI HAI:

Ye poore project ka "Data-Analyst core" hai. Yaha hum sirf charts nahi
banate — hum REAL STATISTICAL TESTS chalate hain, taaki hum confidently
bol sakein "X genuinely sach hai (p<0.05)", na ki "chart me aisa DIKH
raha hai."

4 ANALYSES:
1. Correlation — kaunse factors salary se sabse zyada judte hain
2. Hypothesis Test (t-test) — kya remote-jobs ka salary GENUINELY
   alag hai on-site se, ya ye random variation hai?
3. ANOVA — kya Tier-1/2/3 cities ke beech salary-difference
   statistically significant hai?
4. Trend Test (Mann-Kendall) — kya GenAI-skills ki demand GENUINELY
   badh rahi hai time ke saath, ya ye random noise hai?
"""

import pandas as pd
import numpy as np
from scipy import stats
import pymannkendall as mk
import json


def load_data(path="job_postings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["posted_date"] = pd.to_datetime(df["posted_date"])
    df["month"] = df["posted_date"].dt.to_period("M")

    HIGH_VALUE_SKILLS = {"Machine Learning", "Deep Learning", "AWS", "Kubernetes", "MLOps", "Spark"}
    df["n_high_value_skills"] = df["skills_required"].apply(
        lambda s: len(set(x.strip() for x in s.split(",")) & HIGH_VALUE_SKILLS)
    )
    df["has_emerging_skill"] = df["skills_required"].str.contains(
        "LangChain|LangGraph|Prompt Engineering|RAG|Vector Databases|LLM"
    )
    return df


def correlation_analysis(df: pd.DataFrame) -> dict:
    """Salary ka kis-kis factor se kitna correlation hai."""
    tier_map = {"Tier-1": 3, "Tier-2": 2, "Tier-3": 1}
    df["city_tier_ordinal"] = df["city_tier"].map(tier_map)

    factors = {
        "experience_years_mid": df["experience_years_mid"],
        "n_high_value_skills": df["n_high_value_skills"],
        "city_tier_ordinal": df["city_tier_ordinal"],
        "remote": df["remote"].astype(int),
    }

    results = {}
    for name, series in factors.items():
        r, p = stats.pearsonr(series, df["salary_lpa"])
        results[name] = {"correlation_r": round(r, 3), "p_value": p}

    return results


def hypothesis_test_remote(df: pd.DataFrame) -> dict:
    """T-test: Kya remote vs on-site salary genuinely alag hai?"""
    remote_salaries = df[df["remote"] == True]["salary_lpa"]
    onsite_salaries = df[df["remote"] == False]["salary_lpa"]

    t_stat, p_value = stats.ttest_ind(remote_salaries, onsite_salaries)

    return {
        "remote_mean": round(remote_salaries.mean(), 2),
        "onsite_mean": round(onsite_salaries.mean(), 2),
        "mean_difference": round(onsite_salaries.mean() - remote_salaries.mean(), 2),
        "t_statistic": round(t_stat, 3),
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def anova_city_tier(df: pd.DataFrame) -> dict:
    """ANOVA: Kya Tier-1/2/3 cities ke salary-means genuinely alag hain?"""
    tier1 = df[df["city_tier"] == "Tier-1"]["salary_lpa"]
    tier2 = df[df["city_tier"] == "Tier-2"]["salary_lpa"]
    tier3 = df[df["city_tier"] == "Tier-3"]["salary_lpa"]

    f_stat, p_value = stats.f_oneway(tier1, tier2, tier3)

    return {
        "tier1_mean": round(tier1.mean(), 2),
        "tier2_mean": round(tier2.mean(), 2),
        "tier3_mean": round(tier3.mean(), 2),
        "f_statistic": round(f_stat, 3),
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def trend_test_emerging_skills(df: pd.DataFrame) -> dict:
    """Mann-Kendall trend-test: Kya GenAI-skill demand genuinely badh rahi hai?"""
    monthly_pct = df.groupby("month")["has_emerging_skill"].mean() * 100
    monthly_pct = monthly_pct.sort_index()

    result = mk.original_test(monthly_pct.values)

    return {
        "trend": result.trend,  # 'increasing', 'decreasing', or 'no trend'
        "p_value": result.p,
        "significant_at_0.05": result.p < 0.05,
        "first_month_pct": round(monthly_pct.iloc[0], 1),
        "last_month_pct": round(monthly_pct.iloc[-1], 1),
        "slope_per_month": round(result.slope, 3),
        "monthly_series": {str(k): round(v, 2) for k, v in monthly_pct.items()},
    }


if __name__ == "__main__":
    df = load_data()

    print("=" * 60)
    print("1. CORRELATION ANALYSIS (factors vs salary)")
    print("=" * 60)
    corr = correlation_analysis(df)
    for factor, res in corr.items():
        sig = "significant" if res["p_value"] < 0.05 else "NOT significant"
        print(f"  {factor}: r={res['correlation_r']}, p={res['p_value']:.2e} ({sig})")

    print("\n" + "=" * 60)
    print("2. HYPOTHESIS TEST: Remote vs On-site Salary (t-test)")
    print("=" * 60)
    remote_test = hypothesis_test_remote(df)
    print(f"  Remote mean: {remote_test['remote_mean']} LPA")
    print(f"  On-site mean: {remote_test['onsite_mean']} LPA")
    print(f"  t-statistic: {remote_test['t_statistic']}, p-value: {remote_test['p_value']:.2e}")
    print(f"  Statistically significant? {remote_test['significant_at_0.05']}")

    print("\n" + "=" * 60)
    print("3. ANOVA: City-Tier Salary Differences")
    print("=" * 60)
    anova = anova_city_tier(df)
    print(f"  Tier-1: {anova['tier1_mean']} | Tier-2: {anova['tier2_mean']} | Tier-3: {anova['tier3_mean']}")
    print(f"  F-statistic: {anova['f_statistic']}, p-value: {anova['p_value']:.2e}")
    print(f"  Statistically significant? {anova['significant_at_0.05']}")

    print("\n" + "=" * 60)
    print("4. TREND TEST: GenAI Skill Demand Over Time (Mann-Kendall)")
    print("=" * 60)
    trend = trend_test_emerging_skills(df)
    print(f"  Trend: {trend['trend']}")
    print(f"  First month: {trend['first_month_pct']}% -> Last month: {trend['last_month_pct']}%")
    print(f"  Mann-Kendall p-value: {trend['p_value']:.2e}")
    print(f"  Statistically significant? {trend['significant_at_0.05']}")

    # Save all results for the dashboard to use later
    all_results = {
        "correlation": corr,
        "remote_hypothesis_test": remote_test,
        "city_tier_anova": anova,
        "emerging_skill_trend": trend,
    }
    with open("analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved -> analysis_results.json")
