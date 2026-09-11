# CareerCompass — AI-Powered Career Intelligence & Skill-Gap Platform

A data-analyst-first career navigation tool: statistically-grounded job-market analysis (correlation, hypothesis testing, regression, clustering) combined with an embedding-based skill-gap engine, RAG-grounded course recommendations, and an LLM-generated personalized roadmap — with a mandatory human decision point (the user chooses their own target role and reviews the recommendation, nothing is auto-applied).

## Why this project

Most "career guidance" tools are either static content (generic articles) or a thin LLM wrapper with no grounding in real data. CareerCompass is built the other way around: the **statistics layer is the core deliverable** — real hypothesis tests, an interpretable regression model, and unsupervised clustering on job-market data — and the AI layer only sits on top of it, using retrieval (not memorized/hallucinated facts) to ground its recommendations.

**Scope for this implementation:** a synthetic-but-realistic 30,000-row job-postings dataset (real scraping of job boards violates their Terms of Service and is infeasible for a solo project), with genuine statistical patterns deliberately embedded (skill-salary correlation, city-tier effects, a GenAI-skill demand trend) so the analysis produces meaningful, not just plausible-looking, results.

## Architecture

```
LAYER 1: DATA & STATISTICS CORE
======================================
Synthetic job-postings dataset (30,000 rows, 37 months)
        │
        ▼
EDA + Hypothesis Testing        Correlation, t-test (remote vs on-site),
(scipy, pymannkendall)          ANOVA (city-tier), Mann-Kendall trend test
        │
        ▼
Regression Model (statsmodels)  Interpretable OLS — coefficients + p-values,
                                 R²=0.893 on held-out test data
        │
        ▼
Clustering (scikit-learn)       K-Means on skill-vectors → 8 job-families,
                                 discovered without being told job titles
        │
        ▼
Streamlit Dashboard             Interactive charts, filters, all stats inline

LAYER 2: AI/AGENT LAYER
======================================
User profile (skills + target role)
        │
        ▼
Market-Analysis node    Pulls the target role's real top-skills from data
        │
        ▼
Skill-Gap node           Embedding similarity (sentence-transformers) —
                          matches user skills to required skills semantically,
                          not just exact string match
        │
        ▼
Course-Retrieval node    RAG (Qdrant) over a hand-verified catalog of 10 free
                          courses — every course was confirmed via live web
                          search, not generated from memory
        │
        ▼
Salary-Upside node        Reuses the Layer-1 regression coefficients to
                          estimate the salary impact of closing the gap
        │
        ▼
Roadmap-Generation node   LLM (Groq/Llama) writes the final narrative,
                          strictly grounded in the retrieved course data
```

Orchestrated with **LangGraph** (5 nodes, linear — no retry loop here since there's no failure/recovery scenario like in my other two agentic projects). Integrated directly into the **Streamlit** dashboard as a second tab; no separate FastAPI layer, since Streamlit already runs Python and can call the pipeline directly — adding a REST API in between would be unnecessary complexity for this architecture (unlike my SentinelOps/CodeGuardian projects, which use plain HTML/JS dashboards that genuinely need an API bridge).

## Key design decisions

- **RAG grounding is only as honest as its source.** Rather than let the LLM invent course names/links (which would just be a different flavor of hallucination), every course in the knowledge base was verified via live web search before being added — real platforms, real URLs, real instructors.
- **Classical statistics over deep learning for the time-series trend.** The GenAI-skill demand trend has only ~37 monthly data points — nowhere near enough for an LSTM to learn a genuine pattern without overfitting. A Mann-Kendall trend test and linear-trend regression are the statistically appropriate tools here; reaching for a deep-learning model just because a previous project used one would have been the wrong call for this data volume.
- **Embedding similarity, not exact string matching, for skill comparison.** A user typing "ML" should match a job requiring "Machine Learning." Cosine similarity over sentence-transformer embeddings handles this; naive string matching would not.
- **Interpretable regression (statsmodels) over a black-box model.** The goal here isn't just prediction accuracy — it's being able to say "each high-value skill adds ~₹3.9L, p<0.001" with a defensible statistical basis, which a plain scikit-learn model summarized only by an accuracy number wouldn't give.
- **No unnecessary API layer.** Streamlit is Python end-to-end; wrapping the same-process pipeline in a REST API it doesn't need would add latency and complexity without benefit. (Contrast with SentinelOps/CodeGuardian, where the dashboard is plain HTML/JS and genuinely needs a FastAPI bridge — the right architecture depends on what's actually talking to what.)

## Tech Stack

| Layer | Tools |
|---|---|
| Statistics | SciPy (hypothesis testing), pymannkendall (trend test), statsmodels (regression) |
| Clustering | scikit-learn (K-Means, PCA) |
| Dashboard | Streamlit, Plotly |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| RAG | Qdrant (embedded mode) |
| Agent Orchestration | LangGraph |
| LLM | Llama (via Groq API, OpenAI-compatible) |

## Results

**Regression model:** R² = 0.893 (test set), MAE = ₹2.17L — all coefficients statistically significant (p<0.001).

**Clustering:** 8 job-families discovered from skill co-occurrence alone (no job-title label used), aligning closely with real-world role groupings (e.g., a distinct ML/DL cluster with the highest average salary, a QA cluster with the lowest).

**Hypothesis tests:** city-tier salary differences (ANOVA, p<0.001), remote vs on-site pay gap (t-test, p<0.001), and GenAI-skill demand growth (Mann-Kendall, p<0.001) all statistically confirmed, not just visually apparent.

## Project Structure

```
careercompass/
├── data/
│   └── generate_synthetic_jobs.py   # Dataset generator (embedded statistical patterns)
├── analysis/
│   ├── eda_and_hypothesis_tests.py  # Correlation, t-test, ANOVA, Mann-Kendall
│   ├── salary_regression_model.py   # Interpretable OLS regression
│   └── job_clustering.py            # K-Means job-family discovery
├── agents/
│   ├── skill_gap_engine.py          # Embedding-based skill matching
│   ├── course_knowledge_base.py     # RAG knowledge base (verified courses)
│   └── roadmap_agent.py             # LangGraph pipeline tying it all together
├── dashboard/
│   └── dashboard.py                 # Streamlit app (Explore Market + My Roadmap tabs)
├── requirements.txt
└── .env.example
```

## Setup

```bash
git clone <your-repo-url>
cd careercompass
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
```

### 1. Generate data and run the statistics pipeline
```bash
cd data && python generate_synthetic_jobs.py
cd ../analysis
copy ..\data\job_postings.csv .
python eda_and_hypothesis_tests.py
python salary_regression_model.py
python job_clustering.py
```

### 2. Build the RAG knowledge base
```bash
cd ../agents
copy ..\data\job_postings.csv .
copy ..\analysis\regression_results.json .
python course_knowledge_base.py
```

### 3. Run the dashboard
```bash
cd ../dashboard
copy ..\analysis\job_postings_clustered.csv .
copy ..\analysis\analysis_results.json .
copy ..\analysis\regression_results.json .
copy ..\analysis\cluster_results.json .
copy ..\agents\job_postings.csv .
xcopy ..\agents\course_vector_db course_vector_db\ /E /I
streamlit run dashboard.py
```
Open the **Explore Market** tab to browse market-wide statistics, or the **My Roadmap** tab to enter your own skills and get a personalized, RAG-grounded roadmap.

## Future Improvements

- Replace the synthetic dataset with a real, legally-sourced dataset (e.g., a public Kaggle job-postings dataset) once available.
- Expand the course knowledge base beyond the current 10 verified entries to cover more skills.
- Add authentication and persistent user profiles instead of a stateless per-session input.
- Real-time market-data refresh pipeline instead of a static CSV.

## License

MIT (or your preferred license)
