"""
CareerCompass — Step 8: LangGraph Roadmap Agent (FINAL PIECE)
====================================================================
BHAI, YE FILE KYA KARTI HAI:

Poore project ka "climax" hai — Skill-Gap Engine (Step 6) + RAG
Knowledge-Base (Step 7) + LLM ko ek LangGraph pipeline me jodta hai.

    User Profile (skills + target-role)
            |
            v
    [Market-Analysis Node]   -- data se target-role ki asli skills nikalta hai
            |
            v
    [Skill-Gap Node]         -- embedding-similarity se match/missing nikalta hai
            |
            v
    [Course-Retrieval Node]  -- RAG se, har missing-skill ke liye verified course
            |
            v
    [Salary-Upside Node]     -- regression-coefficients se, potential salary-badhat
            |
            v
    [Roadmap-Generation Node] -- LLM (Groq/Llama) sab kuch ek friendly roadmap me likhta hai
            |
            v
         END

NOTE: Is pipeline me koi RETRY-LOOP nahi hai (jaisa CodeGuardian me tha) —
kyunki yaha koi "test fail hone" jaisi cheez nahi hai. Har real-world
LangGraph pipeline me loops zaroori nahi hote; yaha linear flow hi sahi hai.
"""

import sys
import os
import pandas as pd
import json
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from skill_gap_engine import get_target_skills, compute_skill_gap, estimate_salary_upside
from course_knowledge_base import find_course_for_skill

load_dotenv()


class RoadmapState(TypedDict):
    user_skills: list[str]
    target_role: str
    target_skills: Optional[list[dict]]
    gap_result: Optional[dict]
    course_recommendations: Optional[dict]
    salary_upside: Optional[float]
    roadmap_text: Optional[str]


# --- Shared resources (loaded once) ---
_df = None
_embed_model = None
_qdrant_client = None
_regression_coefs = None
_llm_client = None


def load_resources():
    global _df, _embed_model, _qdrant_client, _regression_coefs, _llm_client
    print(">> Loading resources (dataset, embedding-model, Qdrant, LLM)...")
    _df = pd.read_csv("job_postings.csv")
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    _qdrant_client = QdrantClient(path="./course_vector_db")
    with open("regression_results.json") as f:
        _regression_coefs = json.load(f)["coefficients"]
    _llm_client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")


def market_analysis_node(state: RoadmapState) -> RoadmapState:
    print(">> [Market-Analysis] Finding required skills for target role...")
    target_skills = get_target_skills(_df, state["target_role"])
    return {**state, "target_skills": target_skills}


def skill_gap_node(state: RoadmapState) -> RoadmapState:
    print(">> [Skill-Gap] Comparing user skills against target role...")
    gap = compute_skill_gap(state["user_skills"], state["target_skills"], _embed_model)
    print(f"   Readiness: {gap['readiness_pct']}%")
    return {**state, "gap_result": gap}


def course_retrieval_node(state: RoadmapState) -> RoadmapState:
    print(">> [Course-Retrieval] Finding verified courses (RAG) for missing skills...")
    recommendations = {}
    for missing in state["gap_result"]["missing_skills"]:
        matches = find_course_for_skill(missing["skill"], _qdrant_client, _embed_model, top_k=1)
        if matches:
            recommendations[missing["skill"]] = matches[0]
    return {**state, "course_recommendations": recommendations}


def salary_upside_node(state: RoadmapState) -> RoadmapState:
    print(">> [Salary-Upside] Estimating potential salary increase...")
    upside = estimate_salary_upside(state["gap_result"]["missing_skills"], _regression_coefs)
    return {**state, "salary_upside": upside}


def roadmap_generation_node(state: RoadmapState) -> RoadmapState:
    print(">> [Roadmap-Generation] Writing personalized roadmap (LLM)...")

    course_lines = []
    for skill, course in state["course_recommendations"].items():
        course_lines.append(f"- {skill}: \"{course['course_title']}\" on {course['platform']} ({course['url']})")

    prompt = f"""You are a career advisor. Write a friendly, motivating, and CONCISE career roadmap
in Markdown for a student, using ONLY the information below (do not invent courses or numbers).

TARGET ROLE: {state['target_role']}
CURRENT READINESS: {state['gap_result']['readiness_pct']}%

ALREADY HAS THESE SKILLS: {[m['skill'] for m in state['gap_result']['matched_skills']]}

MISSING SKILLS AND THEIR VERIFIED COURSES:
{chr(10).join(course_lines)}

ESTIMATED SALARY UPSIDE if high-value missing skills are learned: +Rs.{state['salary_upside']}L per annum

Structure with these sections: # Your Path to {state['target_role']}, ## Where You Stand,
## What To Learn Next (list each missing skill with its EXACT course link given above),
## Why It's Worth It (mention the salary upside)."""

    response = _llm_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return {**state, "roadmap_text": response.choices[0].message.content}


# --- Build the graph ---
graph = StateGraph(RoadmapState)
graph.add_node("market_analysis", market_analysis_node)
graph.add_node("skill_gap", skill_gap_node)
graph.add_node("course_retrieval", course_retrieval_node)
graph.add_node("salary_upside", salary_upside_node)
graph.add_node("roadmap_generation", roadmap_generation_node)

graph.set_entry_point("market_analysis")
graph.add_edge("market_analysis", "skill_gap")
graph.add_edge("skill_gap", "course_retrieval")
graph.add_edge("course_retrieval", "salary_upside")
graph.add_edge("salary_upside", "roadmap_generation")
graph.add_edge("roadmap_generation", END)

app = graph.compile()


if __name__ == "__main__":
    load_resources()

    initial_state = {
        "user_skills": ["Python", "SQL", "Excel", "Statistics"],
        "target_role": "Data Scientist",
        "target_skills": None, "gap_result": None,
        "course_recommendations": None, "salary_upside": None, "roadmap_text": None,
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL ROADMAP")
    print("=" * 60)
    print(result["roadmap_text"])
