"""
CareerCompass — Step 4: Clustering (Job-Family Discovery)
================================================================
BHAI, YE FILE KYA KARTI HAI:

Humne pehle regression se "kaun se factors salary predict karte hain"
dekha (SUPERVISED — humne target diya tha). Ab hum UNSUPERVISED karte
hain — model ko sirf SKILLS dete hain, aur woh KHUD dhoondta hai
"kaunse job-postings ek-dusre se milte-julte hain," bina humein
job_title batana.

APPROACH:
1. Har posting ke skills ko "multi-hot vector" banate hain
   (jaise: [Python=1, SQL=1, React=0, AWS=0, ...])
2. K-Means clustering chalate hain is vector-space me
3. Best "k" (kitne clusters) Silhouette Score se choose karte hain
4. Har cluster ke "top skills" dekh ke, use ek naam dete hain
   (jaise "Cluster 2 = Data/ML family")
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import json


def build_skill_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Har posting ke skills ko ek multi-hot (0/1) matrix me convert karta hai."""
    all_skills = set()
    skill_lists = df["skills_required"].apply(lambda s: [x.strip() for x in s.split(",")])
    for skills in skill_lists:
        all_skills.update(skills)
    all_skills = sorted(all_skills)

    matrix = pd.DataFrame(0, index=df.index, columns=all_skills)
    for idx, skills in skill_lists.items():
        matrix.loc[idx, skills] = 1

    return matrix, all_skills


def find_best_k(skill_matrix: np.ndarray, k_range=range(3, 9)) -> tuple[int, dict]:
    """Silhouette score se best number-of-clusters dhoondta hai."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(skill_matrix)
        score = silhouette_score(skill_matrix, labels, sample_size=5000, random_state=42)
        scores[k] = round(score, 4)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def interpret_clusters(df: pd.DataFrame, skill_matrix: pd.DataFrame, labels: np.ndarray, all_skills: list) -> dict:
    """Har cluster ke top-skills aur average-salary nikalta hai, samajhne ke liye."""
    df = df.copy()
    df["cluster"] = labels
    skill_matrix = skill_matrix.copy()
    skill_matrix["cluster"] = labels

    cluster_info = {}
    for c in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == c]
        cluster_skills = skill_matrix[skill_matrix["cluster"] == c][all_skills]

        top_skills = cluster_skills.sum().sort_values(ascending=False).head(5)
        top_job_titles = cluster_df["job_title"].value_counts().head(3)

        cluster_info[int(c)] = {
            "size": len(cluster_df),
            "avg_salary": round(cluster_df["salary_lpa"].mean(), 2),
            "top_skills": top_skills.index.tolist(),
            "dominant_job_titles": top_job_titles.index.tolist(),
        }
    return cluster_info


if __name__ == "__main__":
    df = pd.read_csv("job_postings.csv")

    print("Building skill matrix...")
    skill_matrix, all_skills = build_skill_matrix(df)
    print(f"  {len(all_skills)} unique skills found across {len(df)} postings")

    print("\nFinding optimal number of clusters (Silhouette Score)...")
    best_k, scores = find_best_k(skill_matrix.values)
    print("  Scores per k:", scores)
    print(f"  Best k = {best_k}")

    print(f"\nRunning final K-Means with k={best_k}...")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(skill_matrix.values)

    cluster_info = interpret_clusters(df, skill_matrix, labels, all_skills)

    print("\n" + "=" * 60)
    print("CLUSTER INTERPRETATION")
    print("=" * 60)
    for c, info in cluster_info.items():
        print(f"\nCluster {c} (n={info['size']}, avg salary={info['avg_salary']} LPA)")
        print(f"  Dominant roles: {info['dominant_job_titles']}")
        print(f"  Top skills: {info['top_skills']}")

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(skill_matrix.values)
    print(f"\nPCA explained variance (2 components): {pca.explained_variance_ratio_.sum():.2%}")

    # Save for dashboard
    df["cluster"] = labels
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]
    df.to_csv("job_postings_clustered.csv", index=False)

    with open("cluster_results.json", "w") as f:
        json.dump({"best_k": best_k, "silhouette_scores": scores, "clusters": cluster_info}, f, indent=2)

    print("\nSaved -> job_postings_clustered.csv, cluster_results.json")
