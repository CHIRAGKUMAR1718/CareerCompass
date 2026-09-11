"""
CareerCompass — Step 1: Synthetic Job-Postings Dataset Generator
====================================================================
BHAI, YE FILE KYA KARTI HAI:

Real job-board scraping (LinkedIn/Naukri) legal-risk aur infra-heavy hai.
Isliye hum khud ek REALISTIC synthetic dataset generate kar rahe hain —
lekin ismein JAAN-BOOJH KAR genuine statistical patterns "bake" kiye
gaye hain, taaki baad ka EDA/regression/clustering genuinely meaningful
result de, sirf random-noise pe kaam na kare.

EMBEDDED PATTERNS (jo baad me hum "discover" karenge analysis se):
1. High-value skills (Python, AWS, ML, Kubernetes) -> salary premium
2. Tier-1 cities -> higher salary than Tier-2/Tier-3 (testable via ANOVA)
3. Experience -> salary (clear positive correlation, regression ke liye)
4. GenAI/LangChain/LLM skills -> demand TIME KE SAATH badh rahi hai
   (recent months mein zyada frequent — time-series trend ke liye)
5. Remote jobs ka salary-distribution thoda alag hai on-site se
   (t-test / hypothesis-testing ke liye)
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------

JOB_ROLES = {
    "Data Analyst": {"base_salary": 6.0, "skills": ["SQL", "Excel", "Python", "Power BI", "Tableau", "Statistics"]},
    "Data Scientist": {"base_salary": 12.0, "skills": ["Python", "Machine Learning", "SQL", "Statistics", "Deep Learning", "NLP"]},
    "ML Engineer": {"base_salary": 14.0, "skills": ["Python", "Machine Learning", "Deep Learning", "AWS", "Docker", "MLOps"]},
    "Software Engineer": {"base_salary": 9.0, "skills": ["Java", "Python", "SQL", "Git", "System Design"]},
    "Backend Developer": {"base_salary": 8.5, "skills": ["Python", "Node.js", "SQL", "AWS", "Docker"]},
    "Frontend Developer": {"base_salary": 7.5, "skills": ["JavaScript", "React", "HTML/CSS", "TypeScript"]},
    "DevOps Engineer": {"base_salary": 11.0, "skills": ["AWS", "Kubernetes", "Docker", "CI/CD", "Linux"]},
    "Data Engineer": {"base_salary": 11.5, "skills": ["Python", "SQL", "Spark", "AWS", "Airflow"]},
    "Business Analyst": {"base_salary": 7.0, "skills": ["Excel", "SQL", "Power BI", "Statistics", "Communication"]},
    "QA Engineer": {"base_salary": 6.5, "skills": ["Selenium", "Python", "SQL", "Manual Testing"]},
    "Cloud Engineer": {"base_salary": 10.5, "skills": ["AWS", "Azure", "Kubernetes", "Docker", "Linux"]},
    "Product Manager": {"base_salary": 13.0, "skills": ["SQL", "Excel", "Communication", "Agile", "Statistics"]},
}

# In-demand skills — jinki premium DALTE HAIN salary me
HIGH_VALUE_SKILLS = {"Machine Learning", "Deep Learning", "AWS", "Kubernetes", "MLOps", "Spark"}

# GenAI-related skills — jinki FREQUENCY TIME KE SAATH BADHTI HAI (trend ke liye)
EMERGING_SKILLS = ["LangChain", "LangGraph", "Prompt Engineering", "Vector Databases", "RAG", "LLM Fine-tuning"]
EMERGING_ELIGIBLE_ROLES = ["Data Scientist", "ML Engineer", "Data Engineer", "Software Engineer"]

TIER1_CITIES = ["Bangalore", "Mumbai", "Delhi-NCR", "Hyderabad", "Pune"]
TIER2_CITIES = ["Jaipur", "Indore", "Lucknow", "Chandigarh", "Coimbatore"]
TIER3_CITIES = ["Bhopal", "Ranchi", "Nagpur", "Kanpur", "Guwahati"]

EXPERIENCE_LEVELS = [("Fresher (0-1 yrs)", 0.5), ("Junior (1-3 yrs)", 2), ("Mid (3-6 yrs)", 4.5),
                     ("Senior (6-10 yrs)", 8), ("Lead (10+ yrs)", 13)]

COMPANY_SIZES = ["Startup", "Mid-size", "Enterprise"]


def pick_location():
    r = random.random()
    if r < 0.5:
        return random.choice(TIER1_CITIES), "Tier-1"
    elif r < 0.8:
        return random.choice(TIER2_CITIES), "Tier-2"
    else:
        return random.choice(TIER3_CITIES), "Tier-3"


def generate_posting(posting_date: datetime) -> dict:
    role = random.choice(list(JOB_ROLES.keys()))
    role_info = JOB_ROLES[role]

    exp_label, exp_years = random.choice(EXPERIENCE_LEVELS)
    city, tier = pick_location()
    company_size = random.choice(COMPANY_SIZES)
    remote = random.random() < 0.25

    # --- Skills: role ke base-skills se 3-5 randomly, plus chance of high-value skill ---
    n_skills = random.randint(3, 5)
    skills = random.sample(role_info["skills"], min(n_skills, len(role_info["skills"])))

    # Emerging (GenAI) skills ki frequency TIME KE SAATH badhti hai — isliye
    # probability posting_date pe depend karti hai (jitna recent, utna zyada chance)
    if role in EMERGING_ELIGIBLE_ROLES:
        months_ago = (datetime(2026, 9, 1) - posting_date).days / 30
        recency_factor = max(0, 1 - months_ago / 36)  # 0 (36 months pehle) se 1 (abhi)
        emerging_prob = 0.05 + 0.45 * recency_factor  # 5% baseline -> 50% recent
        if random.random() < emerging_prob:
            skills.append(random.choice(EMERGING_SKILLS))

    # --- Salary formula: base + experience-effect + skill-premium + location-multiplier + remote-adjustment + noise ---
    salary = role_info["base_salary"]
    salary += exp_years * 1.2  # experience ka strong positive effect

    n_high_value = len(set(skills) & HIGH_VALUE_SKILLS)
    salary += n_high_value * 1.8  # har high-value skill se premium

    location_multiplier = {"Tier-1": 1.25, "Tier-2": 1.0, "Tier-3": 0.85}[tier]
    salary *= location_multiplier

    if remote:
        salary *= 0.95  # remote roles thoda kam pay karte (testable hypothesis)

    company_multiplier = {"Startup": 0.95, "Mid-size": 1.0, "Enterprise": 1.15}[company_size]
    salary *= company_multiplier

    salary += np.random.normal(0, 1.0)  # random noise
    salary = round(max(3.0, salary), 1)

    return {
        "job_title": role,
        "location": city,
        "city_tier": tier,
        "experience_level": exp_label,
        "experience_years_mid": exp_years,
        "company_size": company_size,
        "remote": remote,
        "skills_required": ", ".join(skills),
        "salary_lpa": salary,
        "posted_date": posting_date.strftime("%Y-%m-%d"),
    }


if __name__ == "__main__":
    N_POSTINGS = 30000
    START_DATE = datetime(2023, 9, 1)
    END_DATE = datetime(2026, 9, 1)
    total_days = (END_DATE - START_DATE).days

    postings = []
    for _ in range(N_POSTINGS):
        random_day_offset = random.randint(0, total_days)
        posting_date = START_DATE + timedelta(days=random_day_offset)
        postings.append(generate_posting(posting_date))

    df = pd.DataFrame(postings)
    df = df.sort_values("posted_date").reset_index(drop=True)
    df.to_csv("job_postings.csv", index=False)

    print(f"Generated {len(df)} synthetic job postings -> job_postings.csv")
    print(f"\nDate range: {df['posted_date'].min()} to {df['posted_date'].max()}")
    print(f"\nJob title distribution:\n{df['job_title'].value_counts()}")
    print(f"\nSalary stats (LPA):\n{df['salary_lpa'].describe()}")
