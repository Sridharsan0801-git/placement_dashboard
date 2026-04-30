"""
llm_analysis.py — Resume analysis with proper weighted scoring.

Scoring weights:
  CGPA            25%
  Skills match    25%
  Internships     20%
  Projects        15%
  Certifications  10%
  Resume quality   5%
"""

from __future__ import annotations
import json, os, re
from typing import Any

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

Student Profile:
- CGPA: {cgpa}
- Internships: {num_internships}
- Projects: {num_projects}
- Certifications: {num_certs}

Internship Details:
{intern_ctx}

Project Details:
{proj_ctx}

Certificate Details:
{cert_ctx}

Analyze and return ONLY this JSON:
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

For placement_readiness_score, use this logic:
- CGPA {cgpa}/10 contributes 25 points max
- Skills match contributes 25 points max
- {num_internships} internship(s) contribute 20 points max (0=0, 1=12, 2=17, 3+=20)
- {num_projects} project(s) contribute 15 points max (0=0, 1=8, 2=12, 3+=15)
- {num_certs} certificate(s) contribute 10 points max (0=0, 1=6, 2=9, 3+=10)
- Resume quality contributes 5 points max
Be precise and realistic. Never return exactly 20.
"""

ROLE_SKILL_MAP = {
    "Data Scientist": ["Python","Machine Learning","SQL","Statistics","Deep Learning","Data Visualization","Pandas","NumPy","Model Validation","Feature Engineering"],
    "Software Developer": ["Java","DSA","System Design","APIs","Databases","Git","Unit Testing","Object-Oriented Programming","Debugging","REST"],
    "AI Engineer": ["Python","Deep Learning","TensorFlow","PyTorch","NLP","Computer Vision","Model Deployment","Docker","MLOps","Cloud"],
    "ML Engineer": ["Python","Machine Learning","TensorFlow","PyTorch","Scikit-Learn","Feature Engineering","Model Deployment","Docker","MLOps","SQL"],
    "Data Engineer": ["SQL","Python","ETL","Data Warehousing","Apache Spark","Airflow","Kafka","Databases","Cloud","Data Modeling"],
    "Product Manager": ["Roadmapping","Stakeholder Management","User Research","Analytics","Prioritization","Communication","Market Research","Agile","Storytelling","Product Strategy"],
    "Business Analyst": ["SQL","Excel","Data Visualization","Business Intelligence","Communication","Requirements Gathering","Process Optimization","Stakeholder Management","Reporting","Power BI"],
    "Cloud Engineer": ["AWS","Azure","GCP","Docker","Kubernetes","Terraform","Linux","CI/CD","Networking","Python"],
    "Backend Developer": ["Java","Python","Node.js","REST APIs","SQL","System Design","Docker","Git","Microservices","Spring Boot"],
    "Full Stack Developer": ["React","Node.js","HTML","CSS","JavaScript","SQL","REST APIs","Git","Docker","System Design"],
    "DevOps Engineer": ["Docker","Kubernetes","CI/CD","Linux","Terraform","Jenkins","AWS","Python","Git","Monitoring"],
    "Cybersecurity Analyst": ["Network Security","Ethical Hacking","Cryptography","Linux","Python","SIEM Tools","Penetration Testing","Incident Response","Digital Forensics","Cloud Security"],
    "Embedded Systems Engineer": ["Embedded C","C++","RTOS","Microcontrollers","UART/SPI/I2C","MATLAB","PCB Design","ARM","IoT","Linux"],
    "Hardware / VLSI Engineer": ["VLSI","Verilog","VHDL","MATLAB","FPGA","Circuit Design","Cadence","Synopsis","Embedded C","Signal Processing"],
    "Data Analyst": ["SQL","Python","Excel","Power BI","Tableau","Statistics","Data Visualization","Pandas","NumPy","Reporting"],
}

CORE_TOOLS = {
    "AWS","GCP","Azure","Docker","Kubernetes","Git","Linux","TensorFlow","PyTorch",
    "Tableau","Power BI","SQL","MySQL","PostgreSQL","MongoDB","Firebase","Spark",
    "Hadoop","React","Node.js","Django","Flask","FastAPI","OpenCV","NLP",
}

ROLE_COMPANY_SUGGESTIONS = {
    "Data Scientist":       ["Google","Amazon","IBM","Microsoft","Flipkart"],
    "Software Developer":   ["Microsoft","Infosys","TCS","Zoho","Wipro"],
    "AI Engineer":          ["Microsoft","Amazon","Adobe","Google","Meta"],
    "ML Engineer":          ["Amazon","Microsoft","Adobe","Flipkart","Swiggy"],
    "Data Engineer":        ["Amazon","Walmart Global Tech","Accenture","IBM","TCS"],
    "Product Manager":      ["Google","Amazon","Microsoft","Zoho","Flipkart"],
    "Business Analyst":     ["Deloitte","Accenture","Cognizant","KPMG","Infosys"],
    "Cloud Engineer":       ["Amazon","Microsoft","Google","Capgemini","Wipro"],
    "Backend Developer":    ["Amazon","Flipkart","Zoho","Razorpay","Swiggy"],
    "Full Stack Developer": ["Zoho","Flipkart","ThoughtWorks","Cognizant","Infosys"],
    "DevOps Engineer":      ["Amazon","Microsoft","ThoughtWorks","Capgemini","HCL Technologies"],
    "Cybersecurity Analyst":["Wipro","HCL Technologies","Tech Mahindra","Accenture","TCS"],
    "Data Analyst":         ["TCS","Infosys","Accenture","Deloitte","Cognizant"],
    "Embedded Systems Engineer": ["L&T Technology Services","HCL Technologies","Bosch","Tata Elxsi","Wipro"],
    "Hardware / VLSI Engineer":  ["Intel","Qualcomm","Texas Instruments","HCL Technologies","L&T Technology Services"],
}

_SKILL_PATTERNS = {
    r"\bpython\b":"Python", r"\bjava\b":"Java", r"c\+\+":"C++",
    r"\bjavascript\b":"JavaScript", r"\btypescript\b":"TypeScript",
    r"\bkotlin\b":"Kotlin", r"\bswift\b":"Swift", r"\bgo(?:lang)?\b":"Go",
    r"\breact\.?js\b|\breact\b":"React", r"\bnode\.?js\b":"Node.js",
    r"\bvue\.?js\b":"Vue.js", r"\bangular\b":"Angular",
    r"\bdjango\b":"Django", r"\bflask\b":"Flask", r"\bfastapi\b":"FastAPI",
    r"machine learning":"Machine Learning", r"deep learning":"Deep Learning",
    r"data science":"Data Science", r"data analysis":"Data Analysis",
    r"\btensorflow\b":"TensorFlow", r"\bpytorch\b":"PyTorch",
    r"\bkeras\b":"Keras", r"\bnlp\b|natural language processing":"NLP",
    r"computer vision":"Computer Vision", r"\bpandas\b":"Pandas",
    r"\bnumpy\b":"NumPy", r"\bscikit.?learn\b":"Scikit-Learn",
    r"\bopencv\b":"OpenCV", r"\bpower\s?bi\b":"Power BI",
    r"\btableau\b":"Tableau", r"\bsql\b":"SQL", r"\bmysql\b":"MySQL",
    r"\bpostgresql\b|\bpostgres\b":"PostgreSQL", r"\bmongodb\b":"MongoDB",
    r"\bfirebase\b":"Firebase", r"\baws\b|amazon web services":"AWS",
    r"\bgcp\b|google cloud":"Google Cloud", r"\bazure\b":"Azure",
    r"\bdocker\b":"Docker", r"\bkubernetes\b|\bk8s\b":"Kubernetes",
    r"\bgit\b":"Git", r"\blinux\b|\bubuntu\b":"Linux",
    r"\bci/cd\b|\bcicd\b":"CI/CD", r"\bembedded\b":"Embedded Systems",
    r"\bvlsi\b":"VLSI", r"\bmatlab\b":"MATLAB", r"\biot\b":"IoT",
    r"\bcybersecurity\b|ethical hack|\bpenetration test":"Cybersecurity",
    r"\bteam\s?work\b|\bcollaboration\b":"Teamwork",
    r"\bleadership\b":"Leadership", r"\bcommunication\b":"Communication",
    r"\bproblem.?solving\b":"Problem Solving", r"\bagile\b|\bscrum\b":"Agile/Scrum",
    r"\bverilog\b":"Verilog", r"\bvhdl\b":"VHDL", r"\bfpga\b":"FPGA",
    r"\brtos\b":"RTOS", r"\barm\b":"ARM", r"\bspring boot\b":"Spring Boot",
    r"\brest\b|\brestful\b":"REST APIs", r"\bdsa\b|data structure":"DSA",
    r"\bexcel\b":"Excel", r"\bpower bi\b":"Power BI",
}


# ══════════════════════════════════════════════════════════════════
# SCORING ENGINE — proper weighted formula
# ══════════════════════════════════════════════════════════════════

def _compute_scores(
    candidate_skills: list[str],
    required_skills: list[str],
    cgpa: float,
    num_internships: int,
    num_projects: int,
    num_certs: int,
    word_count: int,
) -> tuple[int, int]:
    """
    Returns (role_match_score, placement_readiness_score) both 0-100.

    Weights:
      CGPA           25 pts
      Skill match    25 pts
      Internships    20 pts  (0→0, 1→12, 2→17, 3+→20)
      Projects       15 pts  (0→0, 1→8,  2→12, 3+→15)
      Certifications 10 pts  (0→0, 1→6,  2→9,  3+→10)
      Resume quality  5 pts
    """
    # Role match — pure skill overlap
    if required_skills:
        cs = {s.lower() for s in candidate_skills}
        matched = sum(1 for s in required_skills if s.lower() in cs)
        role_score = round(100 * matched / len(required_skills))
    else:
        role_score = min(100, len(candidate_skills) * 7)

    # CGPA component (25 pts max)
    cgpa_pts = round((min(cgpa, 10.0) / 10.0) * 25)

    # Skill component (25 pts max)
    skill_pts = round((role_score / 100) * 25)

    # Internship component (20 pts max)
    intern_pts = {0: 0, 1: 12, 2: 17}.get(num_internships, 20)

    # Project component (15 pts max)
    proj_pts = {0: 0, 1: 8, 2: 12}.get(num_projects, 15)

    # Certificate component (10 pts max)
    cert_pts = {0: 0, 1: 6, 2: 9}.get(num_certs, 10)

    # Resume quality (5 pts max) — based on word count
    if word_count >= 400:
        qual_pts = 5
    elif word_count >= 200:
        qual_pts = 3
    elif word_count >= 80:
        qual_pts = 1
    else:
        qual_pts = 0

    readiness = cgpa_pts + skill_pts + intern_pts + proj_pts + cert_pts + qual_pts
    readiness = max(5, min(99, readiness))

    return role_score, readiness


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

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
    return [s.strip() for s in sentences
            if any(k.lower() in s.lower() for k in keywords)][:limit]


def _recommend_companies(role: str) -> list[str]:
    norm = role.strip().title()
    for key, val in ROLE_COMPANY_SUGGESTIONS.items():
        if key.lower() in norm.lower() or norm.lower() in key.lower():
            return val
    return ["Google", "Amazon", "Microsoft", "TCS", "Infosys"]


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(i).strip() for i in value if str(i).strip()]
    if isinstance(value, str):
        return [i.strip() for i in re.split(r"[\n,;]+", value) if i.strip()]
    return []


def _extract_json_block(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON in LLM output.")
    return match.group(0)


def _parse_llm_json(text: str) -> dict[str, Any]:
    text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_extract_json_block(text))


# ══════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════

def get_required_skills_for_role(role: str) -> list[str]:
    norm = role.strip().title()
    if norm in ROLE_SKILL_MAP:
        return ROLE_SKILL_MAP[norm]
    for key, val in ROLE_SKILL_MAP.items():
        if key.lower() in norm.lower() or norm.lower() in key.lower():
            return val
    return []


def get_recommended_roles(candidate_skills: list[str]) -> list[str]:
    cs = {s.lower() for s in candidate_skills}
    scores = []
    for role, required in ROLE_SKILL_MAP.items():
        matched = sum(1 for s in required if s.lower() in cs)
        scores.append((round(100 * matched / max(len(required), 1)), role))
    scores.sort(reverse=True)
    return [f"{role} ({pct}%)" for pct, role in scores[:3]]


def analyze_resume_with_llm(
    job_role: str,
    resume_text: str,
    cgpa: float = 0.0,
    num_internships: int = 0,
    num_projects: int = 0,
    num_certs: int = 0,
    intern_ctx: str = "",
    proj_ctx: str = "",
    cert_ctx: str = "",
) -> dict[str, Any]:
    """
    Analyze resume. Uses Claude API if key available, else rule-based.
    Always computes scores from the weighted formula — never static.
    """
    if not job_role:
        job_role = "Software Developer"

    # Detect skills from resume + context
    combined_text = resume_text + " " + intern_ctx + " " + proj_ctx + " " + cert_ctx
    candidate_skills = _detect_skills(combined_text)
    required_skills  = get_required_skills_for_role(job_role)
    word_count       = len(resume_text.split())

    # Compute proper scores
    role_score, readiness = _compute_scores(
        candidate_skills, required_skills,
        cgpa, num_internships, num_projects, num_certs, word_count,
    )

    missing_skills = [s for s in required_skills
                      if s.lower() not in {x.lower() for x in candidate_skills}]
    tools = _detect_tools(candidate_skills)

    # Find sections from resume text
    internships_found  = _find_sections(resume_text, ["internship","trainee","work experience"])
    projects_found     = _find_sections(resume_text, ["project","built","developed","implemented"])
    certs_found        = _find_sections(resume_text, ["certificate","certified","Coursera","Udemy","NPTEL"])

    # Build recommendations based on actual gaps
    recommendations = []
    if role_score < 50:
        recommendations.append(f"Your skills need significant work for {job_role}. Focus on: {', '.join(missing_skills[:4])}.")
    elif role_score < 75:
        recommendations.append(f"Good base for {job_role}. Fill these gaps: {', '.join(missing_skills[:3])}.")
    if num_internships == 0:
        recommendations.append("No internship detected. Apply for internships on Internshala, LinkedIn, or Unstop.")
    if num_projects < 2:
        recommendations.append(f"Add more projects relevant to {job_role}. Aim for at least 2-3 strong projects.")
    if num_certs == 0:
        recommendations.append(f"Earn a certification in your target domain — try Google, Coursera, or NPTEL.")
    if cgpa < 7.0:
        recommendations.append("CGPA below 7.0 may filter you out of some companies. Aim to improve it.")
    if not recommendations:
        recommendations.append("Strong profile! Keep building projects and applying for internships.")

    base_result = {
        "target_role": job_role,
        "skills_detected": candidate_skills,
        "projects_detected": projects_found,
        "internships_detected": internships_found,
        "certificates_detected": certs_found,
        "tools_detected": tools,
        "role_match_score": str(role_score),
        "placement_readiness_score": str(readiness),
        "missing_skills": missing_skills,
        "recommendations": recommendations,
        "recommended_companies": _recommend_companies(job_role),
    }

    # Try Claude API — if available, override skills/projects/internships/certs/recommendations
    # but KEEP our computed scores (they're more accurate than hallucinated LLM scores)
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass

    if _ANTHROPIC and api_key and resume_text.strip():
        try:
            client  = _anthropic_lib.Anthropic(api_key=api_key)
            prompt  = LLM_PROMPT_TEMPLATE.format(
                job_role=job_role, resume_text=resume_text[:5000],
                cgpa=cgpa, num_internships=num_internships,
                num_projects=num_projects, num_certs=num_certs,
                intern_ctx=intern_ctx or "None",
                proj_ctx=proj_ctx or "None",
                cert_ctx=cert_ctx or "None",
            )
            msg = client.messages.create(
                model=MODEL_NAME, max_tokens=1000,
                system=LLM_SYSTEM,
                messages=[{"role":"user","content":prompt}],
            )
            payload = _parse_llm_json(msg.content[0].text)

            # Use LLM for qualitative fields, our formula for scores
            base_result["skills_detected"]      = _normalize_list(payload.get("skills_detected", candidate_skills))
            base_result["projects_detected"]     = _normalize_list(payload.get("projects_detected", projects_found))
            base_result["internships_detected"]  = _normalize_list(payload.get("internships_detected", internships_found))
            base_result["certificates_detected"] = _normalize_list(payload.get("certificates_detected", certs_found))
            base_result["tools_detected"]        = _normalize_list(payload.get("tools_detected", tools))
            base_result["missing_skills"]        = _normalize_list(payload.get("missing_skills", missing_skills))
            base_result["recommendations"]       = _normalize_list(payload.get("recommendations", recommendations))
            if payload.get("recommended_companies"):
                base_result["recommended_companies"] = _normalize_list(payload["recommended_companies"])
            # Keep our formula-based scores — don't trust LLM numbers
            base_result["role_match_score"]           = str(role_score)
            base_result["placement_readiness_score"]  = str(readiness)
        except Exception:
            pass  # fallback result already built

    return base_result
