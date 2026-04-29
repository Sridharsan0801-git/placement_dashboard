"""
llm_analysis.py — Recruiter-grade resume analysis and role matching.

Uses Anthropic Claude API (claude-haiku) with a full rule-based fallback
if the API key is absent or the call fails.

Set ANTHROPIC_API_KEY in your Streamlit secrets (.streamlit/secrets.toml):
  ANTHROPIC_API_KEY = "sk-ant-..."
"""

from __future__ import annotations
import json
import os
import re
from typing import Any

# ── Anthropic SDK (optional — graceful fallback) ──────────────────
try:
    import anthropic as _anthropic_lib
    _ANTHROPIC = True
except ImportError:
    _ANTHROPIC = False

MODEL_NAME = "claude-haiku-4-5-20251001"

LLM_SYSTEM = (
    "You are an expert AI recruitment assistant who evaluates student resumes. "
    "Always respond with valid JSON and nothing else — no markdown, no code fences."
)

LLM_PROMPT_TEMPLATE = """Target Job Role: {job_role}

Resume Content:
{resume_text}

Analyze this resume and return ONLY a JSON object with exactly these keys:
{{
  "target_role": "{job_role}",
  "skills_detected": [],
  "projects_detected": [],
  "internships_detected": [],
  "certificates_detected": [],
  "tools_detected": [],
  "role_match_score": "0",
  "placement_readiness_score": "0",
  "missing_skills": [],
  "recommendations": [],
  "recommended_companies": []
}}

Rules:
- skills_detected: list of technical skills found in the resume
- projects_detected: brief one-line summaries of projects mentioned
- internships_detected: company + role + duration for each internship
- certificates_detected: certificate names and issuers found
- tools_detected: tools/technologies actually used
- role_match_score: integer 0-100 how well the resume matches {job_role}
- placement_readiness_score: integer 0-100 overall employability score
- missing_skills: skills required for {job_role} that are absent from resume
- recommendations: 3-5 actionable improvement tips
- recommended_companies: 3-5 real companies that hire for {job_role}
"""

ROLE_SKILL_MAP = {
    "Data Scientist": [
        "Python", "Machine Learning", "SQL", "Statistics", "Deep Learning",
        "Data Visualization", "Pandas", "NumPy", "Model Validation", "Feature Engineering",
    ],
    "Software Developer": [
        "Java", "DSA", "System Design", "APIs", "Databases",
        "Git", "Unit Testing", "Object-Oriented Programming", "Debugging", "REST",
    ],
    "AI Engineer": [
        "Python", "Deep Learning", "TensorFlow", "PyTorch", "NLP",
        "Computer Vision", "Model Deployment", "Docker", "MLOps", "Cloud",
    ],
    "ML Engineer": [
        "Python", "Machine Learning", "TensorFlow", "PyTorch", "Scikit-Learn",
        "Feature Engineering", "Model Deployment", "Docker", "MLOps", "SQL",
    ],
    "Data Engineer": [
        "SQL", "Python", "ETL", "Data Warehousing", "Apache Spark",
        "Airflow", "Kafka", "Databases", "Cloud", "Data Modeling",
    ],
    "Product Manager": [
        "Roadmapping", "Stakeholder Management", "User Research", "Analytics", "Prioritization",
        "Communication", "Market Research", "Agile", "Storytelling", "Product Strategy",
    ],
    "Business Analyst": [
        "SQL", "Excel", "Data Visualization", "Business Intelligence", "Communication",
        "Requirements Gathering", "Process Optimization", "Stakeholder Management", "Reporting", "Power BI",
    ],
    "Cloud Engineer": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
        "Linux", "CI/CD", "Networking", "Python",
    ],
    "Backend Developer": [
        "Java", "Python", "Node.js", "REST APIs", "SQL", "System Design",
        "Docker", "Git", "Microservices", "Spring Boot",
    ],
    "Full Stack Developer": [
        "React", "Node.js", "HTML", "CSS", "JavaScript", "SQL",
        "REST APIs", "Git", "Docker", "System Design",
    ],
    "DevOps Engineer": [
        "Docker", "Kubernetes", "CI/CD", "Linux", "Terraform", "Jenkins",
        "AWS", "Python", "Git", "Monitoring",
    ],
    "Cybersecurity Analyst": [
        "Network Security", "Ethical Hacking", "Cryptography", "Linux", "Python",
        "SIEM Tools", "Penetration Testing", "Incident Response", "Digital Forensics", "Cloud Security",
    ],
}

CORE_TOOLS = {
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Git", "Linux",
    "TensorFlow", "PyTorch", "Tableau", "Power BI", "SQL", "MySQL",
    "PostgreSQL", "MongoDB", "Firebase", "Spark", "Hadoop", "React",
    "Node.js", "Django", "Flask", "FastAPI", "OpenCV", "NLP",
}

ROLE_COMPANY_SUGGESTIONS = {
    "Data Scientist":       ["Google", "Amazon", "IBM", "Microsoft", "Flipkart"],
    "Software Developer":   ["Microsoft", "Infosys", "TCS", "Zoho", "Wipro"],
    "AI Engineer":          ["Microsoft", "Amazon", "Adobe", "Google", "Meta"],
    "ML Engineer":          ["Amazon", "Microsoft", "Adobe", "Flipkart", "Swiggy"],
    "Data Engineer":        ["Amazon", "Walmart Global Tech", "Accenture", "IBM", "TCS"],
    "Product Manager":      ["Google", "Amazon", "Microsoft", "Zoho", "Flipkart"],
    "Business Analyst":     ["Deloitte", "Accenture", "Cognizant", "KPMG", "Infosys"],
    "Cloud Engineer":       ["Amazon", "Microsoft", "Google", "Capgemini", "Wipro"],
    "Backend Developer":    ["Amazon", "Flipkart", "Zoho", "Razorpay", "Swiggy"],
    "Full Stack Developer": ["Zoho", "Flipkart", "ThoughtWorks", "Cognizant", "Infosys"],
    "DevOps Engineer":      ["Amazon", "Microsoft", "ThoughtWorks", "Capgemini", "HCL Technologies"],
    "Cybersecurity Analyst":["Wipro", "HCL Technologies", "Tech Mahindra", "Accenture", "TCS"],
}


# ══════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════

def _extract_json_block(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object in LLM output.")
    return match.group(0)


def _parse_llm_json(text: str) -> dict[str, Any]:
    text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        block = _extract_json_block(text)
        return json.loads(block)


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [i.strip() for i in re.split(r"[\n,;]+", value) if i.strip()]
    return []


def _role_match_score(candidate_skills: list[str], required_skills: list[str]) -> int:
    if not required_skills:
        return 0
    cs = {s.lower() for s in candidate_skills}
    matched = sum(1 for s in required_skills if s.lower() in cs)
    return round(100 * matched / len(required_skills))


def _recommend_companies(skills: list[str], role: str) -> list[str]:
    norm = role.strip().title()
    for key, val in ROLE_COMPANY_SUGGESTIONS.items():
        if key.lower() in norm.lower() or norm.lower() in key.lower():
            return val
    if "data" in norm.lower():
        return ["Google", "Amazon", "IBM", "Microsoft", "TCS"]
    if "software" in norm.lower() or "developer" in norm.lower():
        return ["Microsoft", "Infosys", "TCS", "Zoho", "Wipro"]
    return ["Google", "Amazon", "Microsoft", "TCS", "Infosys"]


def _recommend_roles(candidate_skills: list[str]) -> list[str]:
    cs = {s.lower() for s in candidate_skills}
    scores = []
    for role, required in ROLE_SKILL_MAP.items():
        matched = sum(1 for s in required if s.lower() in cs)
        scores.append((round(100 * matched / max(len(required), 1)), role))
    scores.sort(reverse=True)
    return [f"{role} ({pct}%)" for pct, role in scores[:3]]


_SKILL_PATTERNS = {
    r"\bpython\b": "Python", r"\bjava\b": "Java", r"c\+\+": "C++",
    r"\bjavascript\b": "JavaScript", r"\btypescript\b": "TypeScript",
    r"\bkotlin\b": "Kotlin", r"\bswift\b": "Swift", r"\bgo(?:lang)?\b": "Go",
    r"\breact\.?js\b|\breact\b": "React", r"\bnode\.?js\b": "Node.js",
    r"\bvue\.?js\b": "Vue.js", r"\bangular\b": "Angular",
    r"\bdjango\b": "Django", r"\bflask\b": "Flask", r"\bfastapi\b": "FastAPI",
    r"machine learning": "Machine Learning", r"deep learning": "Deep Learning",
    r"data science": "Data Science", r"data analysis": "Data Analysis",
    r"\btensorflow\b": "TensorFlow", r"\bpytorch\b": "PyTorch",
    r"\bkeras\b": "Keras", r"\bnlp\b|natural language processing": "NLP",
    r"computer vision": "Computer Vision", r"\bpandas\b": "Pandas",
    r"\bnumpy\b": "NumPy", r"\bscikit.?learn\b": "Scikit-Learn",
    r"\bopencv\b": "OpenCV", r"\bpower\s?bi\b": "Power BI",
    r"\btableau\b": "Tableau", r"\bsql\b": "SQL", r"\bmysql\b": "MySQL",
    r"\bpostgresql\b|\bpostgres\b": "PostgreSQL", r"\bmongodb\b": "MongoDB",
    r"\bfirebase\b": "Firebase", r"\baws\b|amazon web services": "AWS",
    r"\bgcp\b|google cloud": "Google Cloud", r"\bazure\b": "Azure",
    r"\bdocker\b": "Docker", r"\bkubernetes\b|\bk8s\b": "Kubernetes",
    r"\bgit\b": "Git", r"\blinux\b|\bubuntu\b": "Linux",
    r"\bci/cd\b|\bcicd\b": "CI/CD", r"\bembedded\b": "Embedded Systems",
    r"\bvlsi\b": "VLSI", r"\bmatlab\b": "MATLAB", r"\biot\b": "IoT",
    r"\bcybersecurity\b|ethical hack|\bpenetration test": "Cybersecurity",
    r"\bteam\s?work\b|\bcollaboration\b": "Teamwork",
    r"\bleadership\b": "Leadership", r"\bcommunication\b": "Communication",
    r"\bproblem.?solving\b": "Problem Solving", r"\bagile\b|\bscrum\b": "Agile/Scrum",
}


def _detect_skills(text: str) -> list[str]:
    found = []
    for pattern, label in _SKILL_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE) and label not in found:
            found.append(label)
    return found


def _detect_tools(skills: list[str]) -> list[str]:
    return [s for s in skills if s in CORE_TOOLS]


def _find_sections(text: str, keywords: list[str], limit: int = 4) -> list[str]:
    sentences = re.split(r"(?<=[\.\?!])\s+", text)
    return [s.strip() for s in sentences if any(k.lower() in s.lower() for k in keywords)][:limit]


def _fallback_analysis(job_role: str, resume_text: str) -> dict[str, Any]:
    candidate_skills = _detect_skills(resume_text)
    required_skills  = get_required_skills_for_role(job_role)
    role_score       = _role_match_score(candidate_skills, required_skills)
    missing_skills   = [s for s in required_skills
                        if s.lower() not in {x.lower() for x in candidate_skills}]
    tools       = _detect_tools(candidate_skills)
    internships = _find_sections(resume_text, ["internship", "trainee", "work experience", "summer project"])
    projects    = _find_sections(resume_text, ["project", "built", "developed", "implemented", "designed"])
    certificates= _find_sections(resume_text, ["certificate", "certified", "Coursera", "Udemy", "NPTEL"])

    readiness = max(20, min(95,
        role_score
        + (5 if projects else 0)
        + (5 if internships else 0)
        + (5 if certificates else 0)
    ))

    recommendations = []
    if role_score < 60:
        recommendations.append(
            f"Strengthen your profile for {job_role} by focusing on: {', '.join(missing_skills[:5])}."
        )
    if not projects:
        recommendations.append("Add one or two project summaries demonstrating end-to-end problem solving.")
    if not internships:
        recommendations.append("Complete an internship or applied project to show practical experience.")
    if not certificates:
        recommendations.append("Earn targeted certifications aligned with your target job role.")
    if not tools:
        recommendations.append("Highlight tools and technologies used in your projects or internships.")

    return {
        "target_role": job_role.strip(),
        "skills_detected": candidate_skills,
        "projects_detected": projects,
        "internships_detected": internships,
        "certificates_detected": certificates,
        "tools_detected": tools,
        "role_match_score": str(role_score),
        "placement_readiness_score": str(readiness),
        "missing_skills": missing_skills,
        "recommendations": recommendations or [
            "Your resume is a strong base. Continue refining project and internship context."
        ],
        "recommended_companies": _recommend_companies(candidate_skills, job_role),
    }


# ══════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════

def get_required_skills_for_role(role: str) -> list[str]:
    normalized = role.strip().title()
    if normalized in ROLE_SKILL_MAP:
        return ROLE_SKILL_MAP[normalized]
    for key, val in ROLE_SKILL_MAP.items():
        if key.lower() in normalized.lower() or normalized.lower() in key.lower():
            return val
    return []


def build_llm_prompt(job_role: str, resume_text: str) -> str:
    return LLM_PROMPT_TEMPLATE.format(
        job_role=job_role.strip(),
        resume_text=resume_text.strip()[:6000]
    )


def analyze_resume_with_llm(job_role: str, resume_text: str) -> dict[str, Any]:
    """Analyze resume using Claude API; falls back to rule-based analysis if unavailable."""
    if not job_role or not resume_text.strip():
        return _fallback_analysis(job_role or "Unknown Role", resume_text)

    # Resolve API key — env var first, then Streamlit secrets
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass

    if _ANTHROPIC and api_key:
        try:
            client  = _anthropic_lib.Anthropic(api_key=api_key)
            prompt  = build_llm_prompt(job_role, resume_text)
            message = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1000,
                system=LLM_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = message.content[0].text
            payload  = _parse_llm_json(raw_text)

            list_fields = [
                "skills_detected", "projects_detected", "internships_detected",
                "certificates_detected", "tools_detected", "missing_skills",
                "recommendations", "recommended_companies",
            ]
            for field in list_fields:
                payload[field] = _normalize_list(payload.get(field, []))

            payload["target_role"] = job_role.strip()
            payload.setdefault("role_match_score", str(_role_match_score(
                payload["skills_detected"], get_required_skills_for_role(job_role)
            )))
            payload.setdefault("placement_readiness_score", "0")

            if not payload["recommended_companies"]:
                payload["recommended_companies"] = _recommend_companies(
                    payload["skills_detected"], job_role
                )
            return payload

        except Exception:
            pass  # Fall through to rule-based fallback

    return _fallback_analysis(job_role, resume_text)


def get_recommended_roles(candidate_skills: list[str]) -> list[str]:
    return _recommend_roles(candidate_skills)
