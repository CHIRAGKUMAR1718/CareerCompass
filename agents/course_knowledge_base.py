"""
CareerCompass — Step 7: RAG Knowledge-Base (Verified Free Courses)
========================================================================
BHAI, YE FILE KYA KARTI HAI:

Ek "course catalog" banata hai — jaise SentinelOps ke runbooks the,
bas yaha topic hai "free courses jo har skill ke liye seekhne layak hain."

ZAROORI BAAT: Har course yaha WEB-SEARCH se VERIFY kiya gaya hai — koi
bhi course-name/link maine khud se "bana" nahi diya. Agar main khud se
fake-links banata, ye bhi ek tarah ka hallucination hota — RAG ka poora
point hi ye hai ki hum GENUINELY-REAL information par based rahein.

SCOPE: Ye ek CORE catalog hai (10 sabse-common missing-skills cover
karta hai humare dataset me) — poore-100-skills ka catalog banana ek
alag, bada kaam hoga (jaisa humne CodeGuardian me scope-discipline
rakha tha, yaha bhi wahi approach hai).
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- VERIFIED (web-search se confirm kiya) free courses ---
COURSE_CATALOG = [
    {
        "skill": "Machine Learning",
        "course_title": "Machine Learning Specialization",
        "platform": "Coursera (Stanford + DeepLearning.AI, Andrew Ng)",
        "url": "https://www.coursera.org/specializations/machine-learning-introduction",
        "description": "Foundational ML course covering supervised learning, neural networks, decision trees, and unsupervised learning. Taught by Andrew Ng, rated 4.9/5 by 4.8M+ learners.",
    },
    {
        "skill": "Deep Learning",
        "course_title": "Deep Learning Specialization",
        "platform": "Coursera (DeepLearning.AI, Andrew Ng)",
        "url": "https://www.coursera.org/specializations/deep-learning",
        "description": "Covers neural networks, CNNs, sequence models, and how to diagnose/improve deep learning systems.",
    },
    {
        "skill": "NLP",
        "course_title": "NLP Course",
        "platform": "Hugging Face",
        "url": "https://huggingface.co/learn/llm-course/chapter1/1",
        "description": "Free course on natural language processing and LLMs using the Hugging Face Transformers, Datasets, and Tokenizers libraries.",
    },
    {
        "skill": "SQL",
        "course_title": "Intro to SQL: Querying and Managing Data",
        "platform": "Khan Academy",
        "url": "https://www.khanacademy.org/computing/computer-programming/sql",
        "description": "Video lessons plus interactive coding challenges covering SELECT, WHERE, JOIN, GROUP BY and database fundamentals.",
    },
    {
        "skill": "AWS",
        "course_title": "AWS Cloud Practitioner Essentials",
        "platform": "AWS Skill Builder (official)",
        "url": "https://skillbuilder.aws",
        "description": "Official AWS training covering cloud computing fundamentals, core services (EC2, S3, RDS), billing, and pricing. 600+ free courses available.",
    },
    {
        "skill": "Statistics",
        "course_title": "Statistics and Probability",
        "platform": "Khan Academy",
        "url": "https://www.khanacademy.org/math/statistics-probability",
        "description": "Covers descriptive statistics, probability, distributions, and inferential statistics fundamentals.",
    },
    {
        "skill": "LangChain",
        "course_title": "LangChain for LLM Application Development",
        "platform": "DeepLearning.AI",
        "url": "https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/",
        "description": "One-hour course taught by LangChain creator Harrison Chase and Andrew Ng, covering prompts, memory, chains, and agents.",
    },
    {
        "skill": "Prompt Engineering",
        "course_title": "ChatGPT Prompt Engineering for Developers",
        "platform": "DeepLearning.AI",
        "url": "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers",
        "description": "Taught by Isa Fulford (OpenAI) and Andrew Ng — best practices for prompting, summarizing, inferring, and transforming text with LLM APIs.",
    },
    {
        "skill": "Python",
        "course_title": "CS50's Introduction to Programming with Python",
        "platform": "Harvard (via edX/CS50.harvard.edu)",
        "url": "https://cs50.harvard.edu/python/",
        "description": "Harvard's beginner-friendly Python course covering functions, variables, conditionals, loops, exceptions, and file I/O.",
    },
    {
        "skill": "RAG",
        "course_title": "Vector Databases: from Embeddings to Applications",
        "platform": "DeepLearning.AI",
        "url": "https://www.deeplearning.ai/courses/vector-databases-embeddings-applications",
        "description": "Covers embeddings, similarity search, and building RAG applications using vector databases.",
    },
    {
        "skill": "Vector Databases",
        "course_title": "Retrieval Augmented Generation (RAG)",
        "platform": "DeepLearning.AI",
        "url": "https://www.deeplearning.ai/courses/retrieval-augmented-generation",
        "description": "Five-module course on RAG architecture, vector databases, chunking, reranking, and production deployment.",
    },
]


def build_course_vector_store():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = QdrantClient(path="./course_vector_db")

    # Har course ka "skill + title + description" embed karte hain, taaki
    # search karte waqt zyada context match ho
    texts = [
        f"{c['skill']}: {c['course_title']} - {c['description']}"
        for c in COURSE_CATALOG
    ]
    embeddings = model.encode(texts)

    if client.collection_exists("courses"):
        client.delete_collection("courses")

    client.create_collection(
        collection_name="courses",
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload=COURSE_CATALOG[i])
        for i in range(len(COURSE_CATALOG))
    ]
    client.upsert(collection_name="courses", points=points)

    print(f"Indexed {len(COURSE_CATALOG)} verified courses into Qdrant -> course_vector_db/")
    return client, model


def find_course_for_skill(skill_name: str, client, model, top_k: int = 1) -> list[dict]:
    """Ek missing-skill ke liye, sabse relevant course dhoondhta hai."""
    query_embedding = model.encode([skill_name])[0]
    results = client.query_points(
        collection_name="courses",
        query=query_embedding.tolist(),
        limit=top_k,
    ).points
    return [r.payload for r in results]


if __name__ == "__main__":
    client, model = build_course_vector_store()

    # --- Demo: missing-skills se test karte hain ---
    test_skills = ["Deep Learning", "Machine Learning", "NLP", "LangGraph"]

    print("\n" + "=" * 60)
    print("TEST: Finding courses for missing skills")
    print("=" * 60)
    for skill in test_skills:
        matches = find_course_for_skill(skill, client, model)
        for m in matches:
            print(f"\n  Missing skill: {skill}")
            print(f"  -> Recommended: {m['course_title']} ({m['platform']})")
            print(f"     {m['url']}")
