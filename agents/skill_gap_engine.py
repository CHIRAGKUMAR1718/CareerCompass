"""
CareerCompass — Step 6: Skill-Gap Engine
==============================================
BHAI, YE FILE KYA KARTI HAI:

User ke CURRENT skills aur ek TARGET job-role leta hai, aur batata hai
"kaunse skills already hain, kaunse MISSING hain."

ZAROORI DETAIL — Embedding-Similarity, na sirf exact-text-match:
Agar user "ML" likhe aur dataset me "Machine Learning" hai, EXACT-MATCH
fail ho jaata. Isliye maine embeddings use kiye (sentence-transformers,
same model jo SentinelOps ke RAG me use kiya tha) — ye "meaning" ke
basis pe match karta hai, sirf spelling pe nahi.

BONUS: Regression-model ke coefficients (Step 3 se) use karke, batata
hai "agar tum ye missing high-value skill seekh lo, salary kitni badh
sakti hai" — statistics-layer aur AI-layer ko jodta hai.
"""

import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MATCH_THRESHOLD = 0.55  # isse zyada similarity ho to "match" maanenge


def compute_skill_premium(df: pd.DataFrame, job_title: str, skill: str) -> float | None:
    """Ek specific skill ka salary-premium, USI ROLE ke andar, DATA SE nikalta hai —
    koi fixed/hardcoded list nahi. Har role ke liye alag premium ho sakta hai."""
    role_df = df[df["job_title"] == job_title]
    has_skill = role_df[role_df["skills_required"].str.contains(skill, regex=False)]
    no_skill = role_df[~role_df["skills_required"].str.contains(skill, regex=False)]

    # Agar kisi group me bahut kam samples hain, reliable estimate nahi mil sakta
    if len(has_skill) < 15 or len(no_skill) < 15:
        return None

    return round(has_skill["salary_lpa"].mean() - no_skill["salary_lpa"].mean(), 2)


def get_target_skills(df: pd.DataFrame, job_title: str, top_n: int = 8) -> list[dict]:
    """Target-role ke liye, DATA SE, sabse zyada demand-wali skills nikalta hai —
    aur HAR skill ka apna, DATA-DRIVEN salary-premium bhi (fixed-list nahi)."""
    role_df = df[df["job_title"] == job_title]
    all_skills = []
    for skills_str in role_df["skills_required"]:
        all_skills.extend([s.strip() for s in skills_str.split(",")])

    skill_counts = pd.Series(all_skills).value_counts()
    top_skills = skill_counts.head(top_n)

    results = []
    for skill, count in top_skills.items():
        premium = compute_skill_premium(df, job_title, skill)
        results.append({
            "skill": skill,
            "demand_pct": round(count / len(role_df) * 100, 1),
            "premium_lpa": premium,  # None agar reliable estimate nahi mila
        })
    return results


def compute_skill_gap(user_skills: list[str], target_skills: list[dict], model) -> dict:
    """Embedding-similarity se user-skills ko target-skills ke saath match karta hai."""
    user_embeddings = model.encode(user_skills)
    target_skill_names = [t["skill"] for t in target_skills]
    target_embeddings = model.encode(target_skill_names)

    similarity_matrix = cosine_similarity(target_embeddings, user_embeddings)

    matched = []
    missing = []

    for i, target_info in enumerate(target_skills):
        best_match_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_match_idx]

        if best_score >= MATCH_THRESHOLD:
            matched.append({
                **target_info,
                "matched_with": user_skills[best_match_idx],
                "similarity": round(float(best_score), 3),
            })
        else:
            missing.append(target_info)  # already has 'premium_lpa' from get_target_skills

    readiness_pct = round(len(matched) / len(target_skills) * 100, 1) if target_skills else 0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "readiness_pct": readiness_pct,
    }


def estimate_salary_upside(missing_skills: list[dict]) -> float:
    """Missing-skills ke DATA-DRIVEN premiums (get_target_skills se aaye) ko jod ke,
    potential salary-upside estimate karta hai. Fixed-list/coefficient nahi —
    har skill ka apna, us-role-specific, real premium use hota hai."""
    total = sum(
        s["premium_lpa"] for s in missing_skills
        if s.get("premium_lpa") is not None and s["premium_lpa"] > 0
    )
    return round(total, 2)


if __name__ == "__main__":
    df = pd.read_csv("job_postings.csv")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Demo run ---
    user_skills = ["Python", "SQL", "Excel", "Statistics"]
    target_role = "Data Scientist"

    print(f"\nUser skills: {user_skills}")
    print(f"Target role: {target_role}")

    target_skills = get_target_skills(df, target_role)
    print(f"\nTop skills required for '{target_role}' (from real data):")
    for t in target_skills:
        print(f"  {t['skill']}: {t['demand_pct']}% of postings")

    gap = compute_skill_gap(user_skills, target_skills, model)

    print(f"\n{'=' * 50}")
    print(f"SKILL-GAP RESULT (Readiness: {gap['readiness_pct']}%)")
    print(f"{'=' * 50}")
    print("\n✅ Already have:")
    for m in gap["matched_skills"]:
        print(f"  {m['skill']} (matched via '{m['matched_with']}', similarity={m['similarity']})")
    print("\n❌ Missing:")
    for m in gap["missing_skills"]:
        premium_text = f"premium: +₹{m['premium_lpa']}L" if m.get("premium_lpa") is not None and m["premium_lpa"] > 0 else "no reliable premium data"
        print(f"  {m['skill']} (demand: {m['demand_pct']}%, {premium_text})")

    upside = estimate_salary_upside(gap["missing_skills"])
    print(f"\n💰 Estimated salary upside if missing skills are learned: +₹{upside}L")
