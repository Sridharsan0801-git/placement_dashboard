"""
╔══════════════════════════════════════════════════════════════════╗
║     UNIVERSITY — AI PLACEMENT INTELLIGENCE SYSTEM               ║
║                        Version 5.0                              ║
╚══════════════════════════════════════════════════════════════════╝

What's new in v5.0:
  ✦ AI-powered resume analysis with GPT-OSS-20B
  ✦ Intelligent role-resume matching
  ✦ Skill detection and gap analysis
  ✦ Placement readiness scoring
  ✦ Company eligibility matching
  ✦ Real-time AI insights
"""

import re, io, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from utils.auth import validate_email, parse_register_number, extract_name, is_admin, ADMIN_EMAIL
from utils.companies import COMPANY_CRITERIA, get_eligible_companies, ALL_ROLES
from utils.departments import DEPARTMENTS, DEPARTMENT_SCHOOLS, DEPARTMENT_SKILLS, get_skill_category
from utils.resume_parser import extract_text, parse_resume, get_support_status, STREAMLIT_TYPES
from utils.llm_analysis import analyze_resume_with_llm
from utils.student_store import (save_student_submission, get_all_submissions,
                                  get_submission_count, clear_store, submissions_to_df)

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Placement Analysis",
                   page_icon="🚀", layout="wide",
                   initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],.main{background:#080e1d!important;color:#dde3f0!important;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#0b1120!important;border-right:1px solid rgba(96,165,250,.1)!important}
[data-testid="stSidebar"] *{color:#c4cede!important}
h1,h2,h3,h4,h5,h6{font-family:'Syne',sans-serif!important;letter-spacing:-.02em}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden;display:none}
[data-testid="stSidebarCollapseButton"]{visibility:visible!important;display:flex!important}
[data-testid="collapsedControl"]{visibility:visible!important;display:flex!important}
section[data-testid="stSidebar"]{display:flex!important;visibility:visible!important;width:280px!important;min-width:280px!important;transform:translateX(0px)!important;left:0!important;position:relative!important;}
[data-testid="stSidebarContent"]{display:flex!important;flex-direction:column!important;visibility:visible!important;}
.stMainBlockContainer,[data-testid="stAppViewBlockContainer"]{margin-left:0!important;}

[data-testid="stSidebar"] [role="radiogroup"] label{display:flex!important;align-items:center!important;padding:9px 14px!important;border-radius:9px!important;margin:2px 0!important;cursor:pointer!important;transition:background .18s!important;font-size:.88rem!important;font-weight:500!important}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(96,165,250,.08)!important}

[data-testid="metric-container"]{background:rgba(255,255,255,.035)!important;border:1px solid rgba(255,255,255,.07)!important;border-radius:14px!important;padding:1rem 1.3rem!important;transition:border-color .2s}
[data-testid="metric-container"]:hover{border-color:rgba(96,165,250,.3)!important}
[data-testid="metric-container"] label{color:#64748b!important;font-size:.75rem!important;text-transform:uppercase;letter-spacing:.06em}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#f0f9ff!important;font-family:'Syne',sans-serif!important;font-size:1.65rem!important}
[data-testid="metric-container"] [data-testid="stMetricDelta"] svg{display:none}

[data-testid="stTextInput"] input{background:rgba(255,255,255,.05)!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:10px!important;color:#f1f5f9!important;padding:.6rem .9rem!important;transition:border-color .2s,box-shadow .2s!important}
[data-testid="stTextInput"] input:focus{border-color:#3b82f6!important;box-shadow:0 0 0 3px rgba(59,130,246,.15)!important;outline:none!important}
[data-testid="stTextInput"] input::placeholder{color:#475569!important}
[data-testid="stSelectbox"]>div>div,[data-baseweb="select"]>div{background:rgba(255,255,255,.05)!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:10px!important;color:#f1f5f9!important}

.stButton>button{background:linear-gradient(135deg,#3b82f6 0%,#1d4ed8 100%)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:.92rem!important;padding:.65rem 1.6rem!important;transition:transform .2s,box-shadow .2s!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(59,130,246,.4)!important}
[data-testid="stDownloadButton"] button{background:rgba(255,255,255,.06)!important;border:1px solid rgba(255,255,255,.14)!important;color:#94a3b8!important;border-radius:10px!important}
[data-testid="stDownloadButton"] button:hover{background:rgba(255,255,255,.1)!important;color:#f1f5f9!important;transform:translateY(-1px)!important;box-shadow:none!important}

[data-baseweb="tab-list"]{background:rgba(255,255,255,.04)!important;border-radius:12px!important;padding:4px!important;gap:4px!important;border:1px solid rgba(255,255,255,.06)!important}
[data-baseweb="tab"]{background:transparent!important;border-radius:8px!important;color:#64748b!important;font-weight:500!important}
[aria-selected="true"][data-baseweb="tab"]{background:rgba(59,130,246,.22)!important;color:#93c5fd!important;font-weight:600!important}

[data-testid="stExpander"]{background:rgba(255,255,255,.025)!important;border:1px solid rgba(255,255,255,.07)!important;border-radius:12px!important;margin-bottom:6px!important;transition:border-color .2s!important}
[data-testid="stExpander"]:hover{border-color:rgba(96,165,250,.2)!important}
[data-testid="stExpander"] summary{color:#c4cede!important;font-weight:500!important}

[data-testid="stProgressBar"]>div{background:rgba(255,255,255,.08)!important;border-radius:999px!important;height:6px!important}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#3b82f6,#06b6d4)!important;border-radius:999px!important}
[data-testid="stDataFrameResizable"]{border-radius:12px!important;overflow:hidden!important;border:1px solid rgba(255,255,255,.08)!important}
[data-testid="stAlert"]{border-radius:12px!important;border-left-width:3px!important}
[data-testid="stMultiSelect"] [data-baseweb="tag"]{background:rgba(59,130,246,.22)!important;color:#93c5fd!important;border-radius:6px!important}
[data-testid="stFileUploader"]{background:rgba(255,255,255,.03)!important;border:2px dashed rgba(99,179,237,.3)!important;border-radius:14px!important;transition:border-color .2s!important}
[data-testid="stFileUploader"]:hover{border-color:rgba(59,130,246,.6)!important}
hr{border-color:rgba(255,255,255,.07)!important}

/* Custom classes */
.ku-header{background:linear-gradient(135deg,#0c1628 0%,#13233f 60%,#0f1e38 100%);border:1px solid rgba(96,165,250,.18);border-radius:20px;padding:2.4rem 2rem;text-align:center;margin-bottom:1.8rem;position:relative;overflow:hidden}
.ku-header::before{content:'';position:absolute;inset:0;pointer-events:none;background:radial-gradient(ellipse at 20% 50%,rgba(59,130,246,.13) 0%,transparent 55%),radial-gradient(ellipse at 80% 50%,rgba(6,182,212,.09) 0%,transparent 55%)}
.ku-header h1{font-size:1.9rem;color:#f0f9ff;margin:0;position:relative}
.ku-header p{color:#64748b;margin:.5rem 0 0;font-size:.9rem;position:relative}
.ku-card{background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.075);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem;transition:border-color .2s}
.ku-card:hover{border-color:rgba(96,165,250,.2)}
.ku-card-accent{background:linear-gradient(135deg,rgba(59,130,246,.08),rgba(6,182,212,.05));border:1px solid rgba(59,130,246,.22);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem}
.ku-card-success{background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.22);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem}
.ku-card-warn{background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.22);border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1rem}

.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.8rem;margin-bottom:1rem}
.stat-box{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:1rem;text-align:center}
.stat-box .sv{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:#f0f9ff;line-height:1}
.stat-box .sl{font-size:.72rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-top:.3rem}

.co-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:.7rem;display:flex;gap:1rem;align-items:flex-start;transition:all .2s}
.co-card:hover{border-color:rgba(96,165,250,.25);background:rgba(255,255,255,.05);transform:translateY(-2px)}
.co-dot{width:40px;height:40px;border-radius:10px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.85rem;color:#fff;font-family:'Syne',sans-serif}

.badge{display:inline-block;background:rgba(59,130,246,.18);border:1px solid rgba(59,130,246,.32);color:#93c5fd;padding:3px 10px;border-radius:999px;font-size:.74rem;margin:2px;font-weight:500}
.badge-green{background:rgba(16,185,129,.12);border-color:rgba(16,185,129,.28);color:#6ee7b7}
.badge-yellow{background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.28);color:#fcd34d}
.badge-red{background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.28);color:#fca5a5}
.badge-purple{background:rgba(139,92,246,.12);border-color:rgba(139,92,246,.28);color:#c4b5fd}
.badge-gold{background:rgba(234,179,8,.15);border-color:rgba(234,179,8,.35);color:#fde047}
.badge-admin{background:rgba(239,68,68,.18);border-color:rgba(239,68,68,.4);color:#fca5a5;font-weight:700}
.badge-orange{background:rgba(245,158,11,.15);border-color:rgba(245,158,11,.35);color:#fdba74}
.badge-teal{background:rgba(20,184,166,.12);border-color:rgba(20,184,166,.28);color:#5eead4}

.section-label{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#475569;margin-bottom:.4rem}
.info-box{background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.2);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem}
.success-box{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#a7f3d0}
.warning-box{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#fde68a}
.danger-box{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.22);border-radius:12px;padding:1rem 1.2rem;font-size:.85rem;color:#fca5a5}
.skill-chip{display:inline-flex;align-items:center;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.25);color:#6ee7b7;padding:4px 10px;border-radius:8px;font-size:.78rem;font-weight:500;margin:3px}
.access-denied{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);border-radius:16px;padding:3rem 2rem;text-align:center;margin:2rem 0}
.resume-required{background:rgba(245,158,11,.07);border:2px dashed rgba(245,158,11,.4);border-radius:14px;padding:1.8rem;text-align:center;margin-bottom:1rem}
.resume-ok{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.3);border-radius:14px;padding:1.2rem;margin-bottom:1rem}
.upload-zone{background:rgba(59,130,246,.05);border:2px dashed rgba(59,130,246,.3);border-radius:16px;padding:2rem;text-align:center}
.year-badge{display:inline-block;background:rgba(6,182,212,.12);border:1px solid rgba(6,182,212,.28);color:#5eead4;padding:4px 12px;border-radius:8px;font-size:.8rem;font-weight:600;margin:2px}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "logged_in": False, "role": None,
    "user_data": {}, "placement_data": None,
    "resume_result": None, "resume_text": "",
    "resume_ready": False, "target_role": None,
    "analysis_output": None, "analysis_context": None,
    "analysis_method": None,
    "last_prediction": None,
    "dataset_source": "synthetic",
    "dataset_meta": {},
    "placement_dataset": None,
    "company_db": {},
    "nav_target": None,          # programmatic nav — never bound to a widget
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _logout():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

def _is_admin():   return st.session_state.role == "admin"
def _is_student(): return st.session_state.role == "student"

def _access_denied():
    st.markdown("""<div class="access-denied">
        <div style="font-size:2.5rem;margin-bottom:.8rem;">🔒</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;color:#fca5a5;font-weight:700;">Access Restricted</div>
        <p style="color:#64748b;margin:.4rem 0 0;font-size:.88rem;">This section is only visible to administrators.</p>
    </div>""", unsafe_allow_html=True)

def _tier_badge(tier):
    return {1:'<span class="badge badge-gold">🌟 Tier 1</span>',
            2:'<span class="badge badge-yellow">⭐ Tier 2</span>',
            3:'<span class="badge">💫 Tier 3</span>'}.get(tier, "")

def _hex_initials(name):
    p = name.split(); return (p[0][0]+(p[1][0] if len(p)>1 else "")).upper()

def _support_badges():
    s = get_support_status()
    pdf  = '<span class="badge badge-red">📄 PDF</span>'   if s["PDF"]  else '<span class="badge">📄 PDF ✗</span>'
    docx = '<span class="badge">📝 DOCX</span>'            if s["DOCX"] else '<span class="badge">📝 DOCX ✗</span>'
    ocr  = '<span class="badge badge-purple">🖼️ OCR</span>' if s["OCR (images / scanned PDFs)"] else '<span class="badge">🖼️ OCR ✗</span>'
    return f"{pdf}{docx}{ocr}"

def _dataset_source_badge():
    if st.session_state.dataset_source == "uploaded":
        meta = st.session_state.dataset_meta
        return f'<span class="badge badge-teal">📂 {meta.get("filename","uploaded")} · {meta.get("rows",0)} rows</span>'
    return '<span class="badge">📊 Synthetic data</span>'

def dataset_summary(df) -> dict:
    if df is None or df.empty:
        return {"total_students":0,"placed_students":0,"placement_rate":0,
                "unique_companies":0,"avg_cgpa":0,"avg_package":0,"years":[],"top_company":"—"}
    pc  = next((c for c in df.columns if c.lower() in ("placed","is_placed")), None)
    coc = next((c for c in df.columns if c.lower() == "company"), None)
    cc  = next((c for c in df.columns if c.lower() == "cgpa"), None)
    pkc = next((c for c in df.columns if "package" in c.lower()), None)
    yc  = next((c for c in df.columns if c.lower() == "year"), None)
    tot = len(df)
    pl  = int(df[pc].sum()) if pc else 0
    return {"total_students":tot,"placed_students":pl,
            "placement_rate":round(pl/tot*100,1) if tot else 0,
            "unique_companies":df[coc].nunique() if coc else 0,
            "avg_cgpa":round(df[cc].mean(),2) if cc else 0,
            "avg_package":round(df[pkc].mean(),1) if pkc else 0,
            "years":sorted(df[yc].dropna().unique().tolist()) if yc else [],
            "top_company":df[coc].value_counts().idxmax() if (coc and pl>0) else "—"}

def compute_yearly_stats(df) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    yc  = next((c for c in df.columns if c.lower() == "year"), None)
    pc  = next((c for c in df.columns if c.lower() in ("placed","is_placed")), None)
    cc  = next((c for c in df.columns if c.lower() == "cgpa"), None)
    pkc = next((c for c in df.columns if "package" in c.lower()), None)
    if not yc or not pc: return pd.DataFrame()
    rows = []
    for yr, g in df.groupby(yc):
        rows.append({"Year":int(yr),"Total":len(g),"Placed":int(g[pc].sum()),
                     "Rate":f"{g[pc].mean()*100:.1f}%",
                     "Avg CGPA":f"{g[cc].mean():.2f}" if cc else "—",
                     "Avg Pkg LPA":f"{g[pkc].mean():.1f}" if pkc else "—"})
    return pd.DataFrame(rows).sort_values("Year",ascending=False)

# ══════════════════════════════════════════════════════════════════
# PAGE: LOGIN
# ══════════════════════════════════════════════════════════════════
def login_page():
    st.markdown("""<div class="ku-header" style="padding:3rem 2rem;">
        <h1 style="font-size:2.2rem;">🎓 Placement Intelligence</h1>
        <p style="font-size:1rem;color:#475569;">AI-Powered Placement Intelligence System v5.0</p>
    </div>""", unsafe_allow_html=True)

    _, mid, _ = st.columns([1,1.5,1])
    with mid:
        mode = st.radio("", ["🎓 Student","🔑 Admin"], horizontal=True, label_visibility="collapsed")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        if mode == "🔑 Admin":
            st.markdown('<div class="section-label">ADMIN LOGIN</div>', unsafe_allow_html=True)
            ae = st.text_input("Admin Email", placeholder=ADMIN_EMAIL)
            ap = st.text_input("Admin Password", type="password")
            if st.button("Login as Admin  →", use_container_width=True):
                if is_admin(ae, ap):
                    st.session_state.logged_in = True
                    st.session_state.role      = "admin"
                    st.session_state.user_data = {"email": ADMIN_EMAIL, "name": "Admin"}
                    st.rerun()
                else:
                    st.error("Invalid admin credentials.")
        else:
            st.markdown('<div class="section-label">STUDENT LOGIN</div>', unsafe_allow_html=True)
            email  = st.text_input("Email ID", placeholder="yourname@karunya.edu.in")
            reg_no = st.text_input("Register Number", placeholder="URK25AI1074")
            dept   = st.selectbox("Department / Programme", ["— Select —"]+DEPARTMENTS)
            yr     = st.selectbox("Current Year", ["— Select —","1st Year","2nd Year","3rd Year","4th Year"])
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if st.button("Login  →", use_container_width=True):
                errs = []
                if not email: errs.append("Email is required.")
                elif not validate_email(email): errs.append("Only **@karunya.edu.in** emails accepted.")
                parsed = None
                if not reg_no: errs.append("Register number is required.")
                else:
                    parsed = parse_register_number(reg_no)
                    if not parsed["valid"]: errs.append("Invalid register number — e.g. **URK25AI1074**")
                if dept == "— Select —": errs.append("Please select your department.")
                if yr   == "— Select —": errs.append("Please select your current year.")
                if errs:
                    for e in errs: st.error(e)
                else:
                    st.session_state.logged_in = True
                    st.session_state.role      = "student"
                    st.session_state.user_data = {
                        "email": email.strip().lower(), "name": extract_name(email),
                        "register_number": reg_no.upper().strip(), "department": dept,
                        "school": DEPARTMENT_SCHOOLS.get(dept,"University"),
                        "year_of_joining": parsed["year_of_joining"],
                        "program_type": parsed["program_type"],
                        "roll_number": parsed["roll_number"], "current_year": yr,
                    }
                    st.rerun()

        st.markdown("""<div class="info-box" style="margin-top:1rem;">
            <b>📋 Register Number:</b> <code>URK25AI1074</code><br>
            <span style="color:#475569;font-size:.82rem;">URK=Undergrad · 25=Year · AI=Dept · 1074=Roll</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════
def dashboard_page():
    user = st.session_state.user_data
    df   = st.session_state.placement_data or pd.DataFrame()
    name = user.get("name","User")
    sub_count = get_submission_count()
    company_count = len(COMPANY_CRITERIA)

    if _is_admin():
        db_badge  = _dataset_source_badge()
        st.markdown(f"""<div class="ku-header">
            <h1>Admin Dashboard 🛡️</h1>
            <p>Full analytics · {sub_count} student submission{"s" if sub_count!=1 else ""} · {db_badge}</p>
        </div>""", unsafe_allow_html=True)

        if df.empty:
            st.info("No placement dataset is loaded yet. Upload a dataset first or use a sample dataset for analytics.")
            st.markdown("### Dataset Status")
            st.markdown(f"- Dataset Records: 0")
            st.markdown(f"- Companies Tracked: {company_count}")
            st.markdown(f"- Live Submissions: {sub_count}")
            return

        rate  = df["Placed"].mean()*100 if "Placed" in df.columns else 0
        avg_c = df["CGPA"].mean()       if "CGPA"   in df.columns else 0
        placed = int(df["Placed"].sum()) if "Placed" in df.columns else 0
        total  = len(df)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Placement Rate",    f"{rate:.1f}%")
        c2.metric("Average CGPA",      f"{avg_c:.2f}")
        c3.metric("Dataset Records",   f"{total:,}")
        c4.metric("Companies Tracked", f"{company_count}")
        c5.metric("Live Submissions",  f"{sub_count}")

        # Year-over-year comparison (if uploaded dataset has year column)
        pds = st.session_state.placement_dataset
        if pds is not None and "year" in pds.columns and pds["year"].nunique() > 1:
            yoy = compute_yearly_stats(pds)
            if not yoy.empty:
                st.markdown("### 📅 Year-over-Year Placement Comparison")
                st.dataframe(yoy, use_container_width=True, hide_index=True)

        st.markdown("### Placement Summary")
        st.markdown(f"- Placed: {placed:,}")
        st.markdown(f"- Not Placed: {total-placed:,}")
        st.markdown(f"- Placement Rate: {rate:.1f}%")

        col = "CGPA" if "CGPA" in df.columns else df.columns[0]
        col_placed = "Placed" if "Placed" in df.columns else None
        if col_placed:
            placement_breakdown = df.groupby(col)[col_placed].agg(Students='count', Placed='sum').reset_index()
            st.markdown("#### CGPA and Placement Breakdown")
            st.dataframe(placement_breakdown, use_container_width=True, hide_index=True)

        # Company distribution if uploaded
        if pds is not None and "company" in pds.columns:
            placed_col = next((c for c in pds.columns if c.lower() in ("placed","is_placed")), None)
            if placed_col:
                st.markdown("### 🏢 Company Distribution (Uploaded Dataset)")
                top_co = pds[pds[placed_col]==1]["company"].value_counts().head(15).reset_index()
                top_co.columns = ["Company","Hired"]
                st.dataframe(top_co, use_container_width=True, hide_index=True)

    else:
        st.markdown(f"""<div class="ku-header">
            <h1>Welcome, {name}! 👋</h1>
            <p>{user['department']}&nbsp;·&nbsp;{user['school']}</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="ku-card" style="text-align:center;padding:2rem;">
            <div style="font-size:2.5rem;margin-bottom:.8rem;">📊</div>
            <p style="color:#64748b;font-size:.9rem;">Use the sidebar to navigate to
            <b style="color:#93c5fd;">AI Resume Insights</b>,
            <b style="color:#93c5fd;">Company Eligibility</b>, or
            <b style="color:#93c5fd;">My Account</b>.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to AI Resume Insights", use_container_width=True, key="go_to_prediction"):
            st.session_state.nav_target = "\U0001f3af  AI Resume Insights"
            st.rerun()
        c1,c2,c3 = st.columns(3)
        c1.metric("Dataset Records",   f"{len(df):,}")
        c2.metric("Companies Tracked", f"{company_count}")
        c3.metric("Live Submissions",  f"{sub_count}")

# ══════════════════════════════════════════════════════════════════
# PAGE: ADMIN DATASET UPLOAD  (the star of v5.0)
# ══════════════════════════════════════════════════════════════════
def admin_dataset_page():
    if _is_student():
        _access_denied()
        return

    st.markdown("""<div class="ku-header">
        <h1>📂 Dataset Upload System</h1>
        <p>Upload your annual placement CSV or Excel — companies update automatically</p>
    </div>""", unsafe_allow_html=True)

    # ── Current dataset status ────────────────────────────────────
    if st.session_state.dataset_source == "uploaded":
        meta = st.session_state.dataset_meta
        pds  = st.session_state.placement_dataset
        s    = dataset_summary(pds)

        st.markdown(f"""<div class="ku-card-success">
            <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#6ee7b7;margin-bottom:.8rem;">
                ✅ Live Dataset Active — {meta.get('filename','')}
            </div>
            <div class="stat-grid">
                <div class="stat-box"><div class="sv">{s['total_students']:,}</div><div class="sl">Students</div></div>
                <div class="stat-box"><div class="sv">{s['placed_students']:,}</div><div class="sl">Placed</div></div>
                <div class="stat-box"><div class="sv">{s['placement_rate']}%</div><div class="sl">Rate</div></div>
                <div class="stat-box"><div class="sv">{s['unique_companies']}</div><div class="sl">Companies</div></div>
                <div class="stat-box"><div class="sv">{s['avg_cgpa']}</div><div class="sl">Avg CGPA</div></div>
                <div class="stat-box"><div class="sv">₹{s['avg_package']}</div><div class="sl">Avg Pkg LPA</div></div>
            </div>
            <div style="font-size:.8rem;color:#475569;margin-top:.4rem;">
                Uploaded: {meta.get('uploaded_at','')} &nbsp;·&nbsp;
                Years: {', '.join(str(y) for y in s['years']) or 'N/A'} &nbsp;·&nbsp;
                Top company: <b style="color:#6ee7b7;">{s['top_company']}</b>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="ku-card-warn">
            <div style="font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:#fcd34d;margin-bottom:.4rem;">
                ⚠️ Using synthetic sample data
            </div>
            <p style="color:#94a3b8;font-size:.85rem;margin:0;">
                Upload a real placement dataset below to activate dynamic companies and year-comparison analytics.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "📤  Upload New Dataset",
        "📋  Preview Data",
        "📊  Company Analytics",
        "📅  Year Comparison",
    ])

    # ── TAB 1: Upload ─────────────────────────────────────────────
    with t1:
        lc, rc = st.columns([1.4, 1])

        with lc:
            st.markdown("#### Upload Placement Dataset")
            st.markdown(f"""<div class="info-box" style="margin-bottom:1rem;">
                <b>Accepted formats:</b>
                <span class="badge badge-green">CSV</span>
                <span class="badge badge-green">Excel (.xlsx)</span>
                {"<span class='badge badge-yellow'>⚠️ openpyxl required for Excel</span>" if True else ""}<br><br>
                <b>Required columns:</b>
                <code>student_name</code> · <code>cgpa</code> · <code>company</code><br><br>
                <b>Optional columns:</b>
                <code>internships</code> · <code>projects</code> · <code>certifications</code> ·
                <code>communication_skill</code> · <code>coding_skill</code> ·
                <code>placed</code> · <code>role</code> · <code>package_lpa</code> ·
                <code>company_type</code> · <code>department</code> · <code>year</code><br><br>
                <span style="color:#475569;font-size:.8rem;">
                Column names are flexible — the system auto-detects common variations.
                </span>
            </div>""", unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Drop your placement dataset here",
                type=['csv', 'xlsx', 'xls'],
                key="placement_uploader",
            )

            if uploaded:
                st.info("Dataset upload functionality has been optimized for the AI-based system. Please use the AI Resume Insights feature for resume analysis.")

        with rc:
            st.markdown("#### About AI Analysis")
            st.markdown("""<div class="ku-card" style="padding:1.5rem;">
                <p style="color:#c4cede;font-size:.9rem;line-height:1.6;">
                The AI Resume Insights feature uses GPT-OSS-20B to analyze student resumes and match them with job roles. 
                Upload your resume to get started with AI-powered placement analysis.
                </p>
            </div>""", unsafe_allow_html=True)

    # ── TAB 2: Preview Data ───────────────────────────────────────
    with t2:
        pds = st.session_state.placement_dataset
        if pds is not None and not pds.empty:
            st.markdown(f"**{len(pds):,} records loaded**")
            st.dataframe(pds.head(50), use_container_width=True, hide_index=True)
        else:
            st.info("No placement dataset uploaded yet. Upload a dataset in the first tab.")

    # ── TAB 3: Company Analytics ──────────────────────────────────
    with t3:
        st.markdown(f"**{len(COMPANY_CRITERIA)} companies tracked**")
        rows = []
        for cname, info in COMPANY_CRITERIA.items():
            rows.append({
                "Company":   cname,
                "Min CGPA":  info["min_cgpa"],
                "Package":   info["package"],
                "Tier":      info["tier"],
                "Type":      info["type"],
            })
        co_df = pd.DataFrame(rows).sort_values("Company")
        st.dataframe(co_df, use_container_width=True, hide_index=True)
        st.markdown("#### Top Companies by Tier")
        st.dataframe(co_df.sort_values("Tier"), use_container_width=True, hide_index=True)

    # ── TAB 4: Year Comparison ────────────────────────────────────
    with t4:
        pds = st.session_state.placement_dataset
        if pds is not None and not pds.empty:
            yoy = compute_yearly_stats(pds)
            if not yoy.empty:
                st.markdown("### Year-over-Year Placement Comparison")
                st.dataframe(yoy, use_container_width=True, hide_index=True)
            else:
                st.info("No 'year' column found in the uploaded dataset.")
        else:
            st.info("Upload a placement dataset to view year-over-year comparison.")

# ══════════════════════════════════════════════════════════════════
# PAGE: COMPANY ELIGIBILITY  (now dynamic)
# ══════════════════════════════════════════════════════════════════
def company_page():
    st.markdown("""<div class="ku-header">
        <h1>🏢 Company Eligibility</h1>
        <p>Discover companies that match your CGPA and target role</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="info-box" style="margin-bottom:1rem;">
        📊 Showing <b>{len(COMPANY_CRITERIA)} companies</b> from our placement database.
        Filter by role, CGPA, and company type to identify the best opportunities.
    </div>""", unsafe_allow_html=True)

    lc, rc = st.columns([1,3])
    with lc:
        cgpa  = st.slider("Your CGPA", 5.0, 10.0, 7.5, 0.1)
        role_opts = ["All Roles"] + ALL_ROLES
        cur   = st.session_state.target_role
        idx   = role_opts.index(cur) if cur and cur in role_opts else 0
        sel   = st.selectbox("🎯 Target Job Role", role_opts, index=idx)
        st.session_state.target_role = None if sel == "All Roles" else sel

        all_types = sorted(set(info["type"] for info in COMPANY_CRITERIA.values()))
        types = st.multiselect("Company Type", all_types, default=all_types[:min(4,len(all_types))])

    with rc:
        role_arg = st.session_state.target_role
        eligible = get_eligible_companies(cgpa, role=role_arg, types=types if types else None)

        t1l = [c for c in eligible if c["tier"]==1]
        t2l = [c for c in eligible if c["tier"]==2]
        t3l = [c for c in eligible if c["tier"]==3]

        role_pill = f'<span class="badge badge-purple">🎯 {role_arg}</span>' if role_arg else ""
        st.markdown(f"""<div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1rem;">
            <span class="badge badge-green">✓ Eligible: {len(eligible)}</span>
            <span class="badge badge-gold">🌟 Tier 1: {len(t1l)}</span>
            <span class="badge badge-yellow">⭐ Tier 2: {len(t2l)}</span>
            <span class="badge">💫 Tier 3: {len(t3l)}</span>
            {role_pill}
        </div>""", unsafe_allow_html=True)

        tab1,tab2,tab3 = st.tabs([f"🌟 Dream ({len(t1l)})",f"⭐ Target ({len(t2l)})",f"💫 Safe ({len(t3l)})"])

        def show_co(lst):
            if not lst:
                st.info("Adjust CGPA, role, or type filters.")
                return
            for c in lst:
                init    = _hex_initials(c["name"])
                roles_h = "".join(f'<span class="badge badge-purple" style="font-size:.7rem;">{r}</span>' for r in c.get("roles",[]))
                skills_t= " · ".join(c.get("skills",[]))
                extra = ""
                st.markdown(f"""<div class="co-card">
                    <div class="co-dot" style="background:{c['logo_color']};">{init}</div>
                    <div style="flex:1;min-width:0;">
                        <div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;">
                            <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f0f9ff;">{c['name']}</span>
                            {_tier_badge(c['tier'])}
                            <span class="badge badge-green">{c['package']}</span>
                            {extra}
                        </div>
                        <div style="font-size:.78rem;color:#475569;margin:.3rem 0;">
                            <b style="color:#64748b;">Min CGPA:</b> {c['min_cgpa']} &nbsp;·&nbsp;
                            <b style="color:#64748b;">Type:</b> {c['type']}
                        </div>
                        <div style="font-size:.76rem;color:#64748b;margin-bottom:.4rem;"><b>Skills:</b> {skills_t}</div>
                        <div>{roles_h}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with tab1: show_co(t1l)
        with tab2: show_co(t2l)
        with tab3: show_co(t3l)

    st.markdown("### 📊 CGPA Requirements")
    df_c = pd.DataFrame([{"Company":n,"Min CGPA":i["min_cgpa"],"Type":i["type"]} for n,i in COMPANY_CRITERIA.items()])
    st.markdown("Review the company eligibility requirements below based on CGPA and company type.")
    st.dataframe(df_c.sort_values("Min CGPA"), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: AI RESUME INSIGHTS
# ══════════════════════════════════════════════════════════════════
def prediction_page():
    from utils.resume_parser import SKILL_MAP as _SKILL_MAP

    st.markdown("""<div class="ku-header">
        <h1>🎯 AI Resume Insights</h1>
        <p>Role-matched resume analysis · Skill gap detection · Placement readiness scoring</p>
    </div>""", unsafe_allow_html=True)

    # ── Step 1 — Academic Profile (CGPA + counts) ─────────────────
    st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
        color:#f0f9ff;margin:.5rem 0 .8rem;">Step 1 — Academic Profile</div>""",
        unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        cgpa = st.number_input("📊 CGPA *", min_value=0.0, max_value=10.0,
            value=7.5, step=0.1, format="%.1f")
    with p2:
        num_internships = st.selectbox("💼 Internships", [0,1,2,3,4,5], index=0)
    with p3:
        num_certs = st.selectbox("📜 Certifications", list(range(11)), index=0)
    with p4:
        num_projects = st.selectbox("🔧 Projects", list(range(11)), index=0)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    # ── Step 2 — Target Job Role ───────────────────────────────────
    st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
        color:#f0f9ff;margin:.5rem 0 .6rem;">Step 2 — Target Job Role *</div>""",
        unsafe_allow_html=True)

    r1, r2 = st.columns([2, 2])
    with r1:
        job_role_select = st.selectbox("Select from list", [
            "Software Developer","Data Scientist","Data Analyst","ML Engineer",
            "Cloud Engineer","Backend Developer","Full Stack Developer","DevOps Engineer",
            "Cybersecurity Analyst","Business Analyst","Consulting Analyst","Product Manager",
            "iOS / Mobile Developer","Embedded Systems Engineer","Hardware / VLSI Engineer",
            "Data Engineer","AI Engineer",
        ], label_visibility="collapsed")
    with r2:
        job_role_custom = st.text_input("Or type a custom role",
            placeholder="e.g. Robotics Engineer, Biomedical Analyst",
            label_visibility="collapsed")

    job_role = job_role_custom.strip() if job_role_custom.strip() else job_role_select
    st.session_state.target_role = job_role

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    # ── Step 3 — Upload Resume ─────────────────────────────────────
    st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
        color:#f0f9ff;margin:.5rem 0 .4rem;">Step 3 — Upload Your Résumé *
        <span style="font-size:.75rem;color:#64748b;font-weight:400;margin-left:.5rem;">
        PDF · DOCX · PNG · JPG · TIFF</span></div>""", unsafe_allow_html=True)

    resume_file = st.file_uploader("", type=["pdf","docx","png","jpg","jpeg","tiff"],
        key="resume_uploader", label_visibility="collapsed")

    if resume_file:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.25);
                    border-radius:10px;padding:.6rem 1rem;font-size:.82rem;color:#6ee7b7;margin-bottom:.5rem;">
            ✅ <b>{resume_file.name}</b> uploaded and ready.
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    # ── Step 4 — Internship Details (only if count > 0) ───────────
    internship_entries = []
    if num_internships > 0:
        st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
            color:#f0f9ff;margin:.8rem 0 .4rem;">Step 4 — Internship Details
            <span style="font-size:.75rem;color:#ef4444;margin-left:.4rem;">* All fields required</span>
            </div>""", unsafe_allow_html=True)

        for i in range(num_internships):
            st.markdown(f"""
            <div style="background:rgba(59,130,246,.06);border:1px solid rgba(59,130,246,.14);
                        border-radius:12px;padding:.9rem 1rem .3rem;margin:.5rem 0;">
                <div style="font-size:.73rem;font-weight:700;color:#93c5fd;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:.5rem;">Internship #{i+1}</div>
            """, unsafe_allow_html=True)
            ic1, ic2 = st.columns(2)
            with ic1:
                co = st.text_input("Company Name *", key=f"i_co_{i}",
                    placeholder="e.g. Zoho, TCS, Amazon")
            with ic2:
                jt = st.text_input("Job Title / Role *", key=f"i_jt_{i}",
                    placeholder="e.g. ML Intern, Data Science Intern")
            dur = st.selectbox("Duration *", key=f"i_dur_{i}",
                options=["— Select —","< 1 month","1 month","2 months","3 months",
                         "4 months","5 months","6 months","6+ months"])
            wd = st.text_area("Work Description * — what you built / did", key=f"i_wd_{i}",
                height=80, placeholder="e.g. Built a customer churn prediction model using Python and scikit-learn. Deployed via Flask API on AWS EC2.")
            tu = st.text_input("Technologies / Tools Used *", key=f"i_tu_{i}",
                placeholder="e.g. Python, Pandas, TensorFlow, MySQL, Git")
            st.markdown("</div>", unsafe_allow_html=True)
            internship_entries.append({"company":co,"title":jt,"duration":dur,"description":wd,"tech":tu})

    # ── Step 5 — Certificate Details (only if count > 0) ──────────
    cert_entries = []
    step_n = 4 + (1 if num_internships > 0 else 0)
    if num_certs > 0:
        st.markdown(f"""<div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
            color:#f0f9ff;margin:.8rem 0 .4rem;">Step {step_n+1} — Certificate Details
            <span style="font-size:.75rem;color:#ef4444;margin-left:.4rem;">* All fields required</span>
            </div>""", unsafe_allow_html=True)
        step_n += 1

        for i in range(num_certs):
            st.markdown(f"""
            <div style="background:rgba(139,92,246,.06);border:1px solid rgba(139,92,246,.14);
                        border-radius:12px;padding:.9rem 1rem .3rem;margin:.5rem 0;">
                <div style="font-size:.73rem;font-weight:700;color:#c4b5fd;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:.5rem;">Certificate #{i+1}</div>
            """, unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                cn = st.text_input("Certificate Name *", key=f"c_nm_{i}",
                    placeholder="e.g. AWS Cloud Practitioner")
            with cc2:
                ci = st.text_input("Issuing Authority *", key=f"c_is_{i}",
                    placeholder="e.g. Amazon, Google, NPTEL, Coursera")
            cs = st.text_area("Key Skills Covered *", key=f"c_sk_{i}", height=70,
                placeholder="e.g. Cloud computing, S3, EC2, Lambda, IAM — what you learned and practised")
            st.markdown("</div>", unsafe_allow_html=True)
            cert_entries.append({"name":cn,"issuer":ci,"skills":cs})

    # ── Step 6 — Project Details (only if count > 0) ───────────────
    project_entries = []
    if num_projects > 0:
        st.markdown(f"""<div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
            color:#f0f9ff;margin:.8rem 0 .4rem;">Step {step_n+1} — Project Details
            <span style="font-size:.75rem;color:#ef4444;margin-left:.4rem;">* All fields required</span>
            </div>""", unsafe_allow_html=True)

        for i in range(num_projects):
            st.markdown(f"""
            <div style="background:rgba(6,182,212,.06);border:1px solid rgba(6,182,212,.14);
                        border-radius:12px;padding:.9rem 1rem .3rem;margin:.5rem 0;">
                <div style="font-size:.73rem;font-weight:700;color:#67e8f9;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:.5rem;">Project #{i+1}</div>
            """, unsafe_allow_html=True)
            pp1, pp2 = st.columns(2)
            with pp1:
                pn = st.text_input("Project Title *", key=f"p_nm_{i}",
                    placeholder="e.g. Customer Churn Predictor")
            with pp2:
                pt = st.selectbox("Project Type *", key=f"p_ty_{i}",
                    options=["— Select —","Academic","Personal","Open Source",
                             "Hackathon","Internship Project","Research"])
            pd_ = st.text_area("Project Description *", key=f"p_ds_{i}", height=80,
                placeholder="e.g. Built ML model to predict customer churn using XGBoost. Achieved 92% accuracy. Deployed as Flask API.")
            ptch = st.text_input("Technologies / Tools Used *", key=f"p_tc_{i}",
                placeholder="e.g. Python, XGBoost, Flask, MySQL, Streamlit, GitHub")
            st.markdown("</div>", unsafe_allow_html=True)
            project_entries.append({"title":pn,"type":pt,"description":pd_,"tech":ptch})

    # ── Validation ─────────────────────────────────────────────────
    def _validate():
        if not resume_file: return False, "Please upload your résumé."
        if cgpa <= 0:        return False, "Please enter your CGPA."
        for idx,e in enumerate(internship_entries):
            n=idx+1
            if not e["company"].strip():     return False, f"Internship #{n}: Company Name required."
            if not e["title"].strip():       return False, f"Internship #{n}: Job Title required."
            if e["duration"]=="— Select —":  return False, f"Internship #{n}: Select a duration."
            if not e["description"].strip(): return False, f"Internship #{n}: Work Description required."
            if not e["tech"].strip():        return False, f"Internship #{n}: Technologies Used required."
        for idx,c in enumerate(cert_entries):
            n=idx+1
            if not c["name"].strip():   return False, f"Certificate #{n}: Name required."
            if not c["issuer"].strip(): return False, f"Certificate #{n}: Issuing Authority required."
            if not c["skills"].strip(): return False, f"Certificate #{n}: Key Skills Covered required."
        for idx,p in enumerate(project_entries):
            n=idx+1
            if not p["title"].strip():       return False, f"Project #{n}: Title required."
            if p["type"]=="— Select —":      return False, f"Project #{n}: Select project type."
            if not p["description"].strip(): return False, f"Project #{n}: Description required."
            if not p["tech"].strip():        return False, f"Project #{n}: Technologies required."
        return True, ""

    # ── Context strings for AI ──────────────────────────────────────
    intern_ctx = "\n".join(
        f"Internship {i+1}: {e['title']} at {e['company']} ({e['duration']}). "
        f"Work: {e['description']} | Tech: {e['tech']}"
        for i,e in enumerate(internship_entries)) or "None provided."

    cert_ctx = "\n".join(
        f"Certificate {i+1}: {c['name']} by {c['issuer']}. Skills: {c['skills']}"
        for i,c in enumerate(cert_entries)) or "None provided."

    proj_ctx = "\n".join(
        f"Project {i+1}: {p['title']} ({p['type']}). {p['description']} | Tech: {p['tech']}"
        for i,p in enumerate(project_entries)) or "None provided."

    # ── Analyse button ─────────────────────────────────────────────
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    ctx_key = (job_role, cgpa, num_internships, num_certs, num_projects,
               resume_file.name if resume_file else None)
    if st.session_state.analysis_context != ctx_key:
        st.session_state.analysis_context = ctx_key
        st.session_state.analysis_output  = None

    analyse_btn = st.button("🚀  Analyse with AI", use_container_width=True)

    if analyse_btn:
        valid, err = _validate()
        if not valid:
            st.error(f"⚠️ {err}")
        else:
            with st.spinner("🤖 Analysing résumé for your target role…"):
                raw_text, method = extract_text(resume_file)
                resume_data      = parse_resume(raw_text)

                # Merge context skills
                ctx_all = intern_ctx + " " + cert_ctx + " " + proj_ctx
                ctx_skills = [lbl for pat,lbl in _SKILL_MAP.items()
                              if re.search(pat, ctx_all, re.I)]
                all_sk = list(dict.fromkeys(resume_data.get("skills",[]) + ctx_skills))
                resume_data["skills"] = all_sk

                output = analyze_resume_with_llm(job_role, raw_text)
                st.session_state.resume_result   = resume_data
                st.session_state.resume_text     = raw_text
                st.session_state.analysis_output = output
                st.session_state.analysis_method = method
                st.session_state.resume_ready    = len(raw_text.strip()) > 30

            # Save to admin
            prob  = min(int(float(output.get("placement_readiness_score",0)))/100, 1.0)
            label = "High 🎉" if prob>=.65 else ("Moderate ⚡" if prob>=.40 else "Needs Work ⚠️")
            save_student_submission(
                {**st.session_state.user_data, "target_role": job_role},
                {"cgpa":cgpa,"internships":num_internships,"projects":num_projects,
                 "certifications":num_certs,"comm":3,"coding":3,
                 "probability":prob,"label":label,
                 "ai_role_match":output.get("role_match_score",""),
                 "ai_readiness":output.get("placement_readiness_score",""),
                 "internship_details":[f"{e['title']} @ {e['company']} ({e['duration']})" for e in internship_entries],
                 "certificate_details":[f"{c['name']} by {c['issuer']}" for c in cert_entries],
                 "project_details":[f"{p['title']} ({p['type']})" for p in project_entries]},
                resume_data,
            )
            st.session_state.last_prediction = {"cgpa":cgpa,"probability":prob,"label":label}
            st.rerun()

    # ── Results ────────────────────────────────────────────────────
    output = st.session_state.analysis_output
    if not output:
        return

    role_score = int(float(output.get("role_match_score",0)))
    readiness  = int(float(output.get("placement_readiness_score",0)))
    rms_c = "#6ee7b7" if role_score>=70 else ("#fcd34d" if role_score>=45 else "#fca5a5")
    prs_c = "#6ee7b7" if readiness>=70  else ("#fcd34d" if readiness>=45  else "#fca5a5")

    st.markdown("---")
    st.markdown("## 📊 AI Analysis Results")

    sc1, sc2 = st.columns(2)
    sc1.markdown(f"""
    <div class="ku-card" style="text-align:center;padding:1.5rem;">
        <div class="section-label">Role Match Score</div>
        <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;color:{rms_c};">{role_score}%</div>
        <div style="font-size:.78rem;color:#475569;">for {job_role}</div>
    </div>""", unsafe_allow_html=True)
    sc2.markdown(f"""
    <div class="ku-card" style="text-align:center;padding:1.5rem;">
        <div class="section-label">Placement Readiness</div>
        <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;color:{prs_c};">{readiness}%</div>
        <div style="font-size:.78rem;color:#475569;">overall profile</div>
    </div>""", unsafe_allow_html=True)

    skills   = output.get("skills_detected",[])
    tools    = output.get("tools_detected",[])
    missing  = output.get("missing_skills",[])
    recs     = output.get("recommendations",[])
    cos      = output.get("recommended_companies",[])

    ta, tb, tc, td = st.tabs([
        "✅  Skills & Tools",
        "⚠️  Missing Skills",
        "💡  Recommendations",
        "🏢  Companies to Target",
    ])

    with ta:
        if skills:
            chips = "".join(f'<span class="skill-chip">✓ {s}</span>' for s in skills)
            st.markdown(f"<div style='margin-bottom:.5rem;'>{chips}</div>", unsafe_allow_html=True)
        if tools:
            st.markdown("**Tools & Technologies:**")
            st.markdown("".join(f'<span class="badge badge-purple">{t}</span>' for t in tools), unsafe_allow_html=True)
        if not skills and not tools:
            st.info("Upload a more detailed résumé or add context in the project/internship fields above.")

    with tb:
        if missing:
            for m in missing:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:.6rem;padding:.35rem 0;
                            border-bottom:1px solid rgba(255,255,255,.04);">
                    <span class="badge badge-red">{m}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No critical skill gaps detected for this role! 🎉")

    with tc:
        if recs:
            for i,r in enumerate(recs,1):
                st.markdown(f"""
                <div style="display:flex;gap:.7rem;align-items:flex-start;
                            padding:.6rem 0;border-bottom:1px solid rgba(255,255,255,.05);">
                    <span style="background:rgba(59,130,246,.2);border-radius:6px;
                                 padding:2px 8px;font-size:.72rem;color:#93c5fd;
                                 font-weight:700;flex-shrink:0;">{i}</span>
                    <span style="color:#c4cede;font-size:.87rem;line-height:1.5;">{r}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("Strong profile! Keep building on your current experience.")

    with td:
        if cos:
            from utils.companies import COMPANY_CRITERIA
            for co in cos:
                info = COMPANY_CRITERIA.get(co,{})
                pkg  = info.get("package","—")
                tier_c = info.get("tier",0)
                badge = {1:'<span class="badge badge-gold">🌟 Tier 1</span>',
                         2:'<span class="badge badge-yellow">⭐ Tier 2</span>',
                         3:'<span class="badge">💫 Tier 3</span>'}.get(tier_c,"")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:.6rem;padding:.5rem 0;
                            border-bottom:1px solid rgba(255,255,255,.05);">
                    <span style="font-weight:600;color:#f0f9ff;min-width:160px;">{co}</span>
                    {badge}<span style="font-size:.78rem;color:#475569;">{pkg}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Fill in your profile details for targeted company recommendations.")

    st.markdown("""
    <div style="background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.2);
                border-radius:10px;padding:.7rem 1rem;margin-top:1rem;font-size:.78rem;color:#6ee7b7;">
        ✅ Analysis saved to Admin panel.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: SKILL RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════
def skills_page():
    user = st.session_state.user_data
    st.markdown("""<div class="ku-header">
        <h1>🛠️ Skill Recommendations</h1>
        <p>Personalised learning roadmap for your programme</p>
    </div>""", unsafe_allow_html=True)

    cat = get_skill_category(user["department"])
    ds  = DEPARTMENT_SKILLS.get(cat, DEPARTMENT_SKILLS["cse"])
    all_skills = sorted(set(ds["core"]+ds["recommended"]+ds["additional"]+[
        "Python","Java","C++","JavaScript","SQL","Git","Linux","Docker",
        "Machine Learning","Deep Learning","Cloud Computing","React","Node.js"]))

    rr      = st.session_state.resume_result or {}
    prefill = [s for s in all_skills if s in rr.get("skills",[])]

    lc, rc = st.columns([1,2.1])
    with lc:
        st.markdown("#### Your Current Skills")
        if prefill:
            st.markdown(f'<div class="success-box" style="margin-bottom:.8rem;">✅ Auto-filled {len(prefill)} skills from résumé.</div>',unsafe_allow_html=True)
        current = st.multiselect("Select all skills you have:", all_skills, default=prefill or all_skills[:2])
        badges  = "".join(f'<span class="badge">{s}</span>' for s in current)
        st.markdown(f"<div style='margin-top:.6rem;'>{badges}</div>", unsafe_allow_html=True)

    with rc:
        cd = sum(1 for s in ds["core"] if s in current)
        rd = sum(1 for s in ds["recommended"] if s in current)
        t1,t2,t3,t4 = st.tabs(["🔴 Core","🟡 Recommended","🟢 Additional","⚠️ Gaps"])
        with t1:
            for s in ds["core"]: st.markdown(f"{'✅' if s in current else '❌'} &nbsp; {s}")
            st.progress(cd/max(len(ds["core"]),1)); st.caption(f"Core: {cd}/{len(ds['core'])}")
        with t2:
            for s in ds["recommended"]: st.markdown(f"{'✅' if s in current else '⬜'} &nbsp; {s}")
            st.progress(rd/max(len(ds["recommended"]),1)); st.caption(f"Recommended: {rd}/{len(ds['recommended'])}")
        with t3:
            for s in ds["additional"]: st.markdown(f"{'✅' if s in current else '⬜'} &nbsp; {s}")
        with t4:
            pri  = ds["core"]+ds["recommended"]+ds["additional"]
            gaps = [s for s in pri if s not in current][:10]
            if gaps:
                for i,s in enumerate(gaps,1):
                    tier="🔴" if s in ds["core"] else("🟡" if s in ds["recommended"] else "🟢")
                    st.markdown(f"{tier} **{i}.** {s}")
                st.markdown("---")
                st.markdown("**📚 Platforms:** Coursera · Udemy · NPTEL · LeetCode · HackerRank · GeeksforGeeks")
            else:
                st.success("🎉 You've covered almost all recommended skills!")

    st.markdown("### 🗺️ 6-Month Roadmap")
    st.markdown("""
1. **Core Skills** — Build programming fundamentals, data structures, and system design.
2. **DSA & Problem Solving** — Practice algorithms and interview-style problems consistently.
3. **Project Building** — Create resume-ready projects with clear outcomes and technologies.
4. **Internship** — Pursue internships or applied projects to gain real-world experience.
5. **Advanced Concepts** — Learn domain-relevant tools, cloud platforms, and deployment skills.
6. **Interview Prep** — Polish communication, mock interviews, and role-specific problem solving.
""")

# ══════════════════════════════════════════════════════════════════
# PAGE: STUDENT SUBMISSIONS  (admin only)
# ══════════════════════════════════════════════════════════════════
def submissions_page():
    if _is_student():
        _access_denied()
        return

    st.markdown("""<div class="ku-header">
        <h1>👥 Student Submissions</h1>
        <p>Live prediction results from all students</p>
    </div>""", unsafe_allow_html=True)

    sub_df = submissions_to_df()
    if sub_df.empty:
        st.info("No student submissions yet.")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Submissions", len(sub_df))
    c2.metric("Unique Students",   sub_df["email"].nunique())
    c3.metric("Avg Probability",   f"{sub_df['Placement_Probability_%'].mean():.1f}%")
    c4.metric("Avg CGPA",          f"{sub_df['CGPA'].mean():.2f}")

    st.dataframe(sub_df, use_container_width=True, height=380)
    st.download_button("📥 Download Submissions", sub_df.to_csv(index=False),
        "student_submissions.csv","text/csv", use_container_width=True)

    st.markdown("### Submission Summary")
    st.markdown(f"- Highest placement probability: {sub_df['Placement_Probability_%'].max():.1f}%")
    st.markdown(f"- Lowest placement probability: {sub_df['Placement_Probability_%'].min():.1f}%")
    st.markdown(f"- Average CGPA: {sub_df['CGPA'].mean():.2f}")

    st.markdown("---")
    if st.button("🗑️ Clear All Submissions", use_container_width=True):
        clear_store(); st.success("Cleared."); st.rerun()

# ══════════════════════════════════════════════════════════════════
# PAGE: MY ACCOUNT
# ══════════════════════════════════════════════════════════════════
def account_page():
    user = st.session_state.user_data
    name = user.get("name","User")
    st.markdown("""<div class="ku-header">
        <h1>👤 My Account</h1>
        <p>Your personal profile and placement summary</p>
    </div>""", unsafe_allow_html=True)

    if _is_admin():
        st.markdown("""<div class="ku-card-accent">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#fca5a5;margin-bottom:.5rem;">🛡️ Administrator Account</div>
            <p style="color:#94a3b8;font-size:.85rem;margin:0;">Full access to all data and analytics.</p>
        </div>""", unsafe_allow_html=True)
        return

    lc, rc = st.columns([1,2])
    with lc:
        initials = "".join(w[0].upper() for w in name.split()[:2])
        st.markdown(f"""<div class="ku-card" style="text-align:center;padding:2rem 1rem;">
            <div style="width:80px;height:80px;border-radius:50%;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                        display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;
                        font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#fff;">{initials}</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#f0f9ff;">{name}</div>
            <div style="font-size:.8rem;color:#64748b;margin-top:.3rem;">{user['email']}</div>
            <div style="margin-top:.8rem;"><span class="badge badge-green">🎓 Student</span></div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="ku-card">
            <div class="section-label">Account Details</div>
            <div style="display:grid;gap:.7rem;font-size:.83rem;margin-top:.5rem;">
                <div><span style="color:#475569;">Register No.</span><br><b style="color:#cbd5e1;">{user['register_number']}</b></div>
                <div><span style="color:#475569;">Year of Joining</span><br><b style="color:#cbd5e1;">{user['year_of_joining']}</b></div>
                <div><span style="color:#475569;">Current Year</span><br><b style="color:#cbd5e1;">{user['current_year']}</b></div>
                <div><span style="color:#475569;">Programme</span><br><b style="color:#cbd5e1;">{user['program_type']}</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with rc:
        st.markdown(f"""<div class="ku-card-accent">
            <div class="section-label">Department</div>
            <div style="font-size:.9rem;color:#e2e8f0;font-weight:500;margin-top:.3rem;">{user['department']}</div>
            <div style="font-size:.78rem;color:#475569;">{user['school']}</div>
        </div>""", unsafe_allow_html=True)

        tr = st.session_state.target_role or "Not set"
        st.markdown(f"""<div class="ku-card">
            <div class="section-label">Target Job Role</div>
            <span class="badge badge-purple">🎯 {tr}</span>
            <p style="color:#475569;font-size:.78rem;margin-top:.5rem;">Set in Company Eligibility.</p>
        </div>""", unsafe_allow_html=True)

        rr     = st.session_state.resume_result or {}
        skills = rr.get("skills",[])
        if skills:
            chips = "".join(f'<span class="skill-chip">✓ {s}</span>' for s in skills)
            auto_cgpa = rr.get("cgpa")
            cgpa_line = f'<div style="font-size:.78rem;color:#6ee7b7;margin-top:.3rem;">📊 CGPA from résumé: <b>{auto_cgpa}</b></div>' if auto_cgpa else ""
            st.markdown(f"""<div class="ku-card">
                <div class="section-label">Skills from Résumé ({len(skills)})</div>
                {cgpa_line}<div style="margin-top:.5rem;">{chips}</div>
            </div>""", unsafe_allow_html=True)

        lp = st.session_state.last_prediction
        if lp:
            pct = lp["probability"]*100
            clr = "#6ee7b7" if pct>=65 else ("#fcd34d" if pct>=40 else "#fca5a5")
            st.markdown(f"""<div class="ku-card">
                <div class="section-label">Last Prediction</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:{clr};">{pct:.1f}%</div>
                <div style="font-size:.8rem;color:#64748b;margin-top:.2rem;">{lp['label']}</div>
                <div style="font-size:.76rem;color:#334155;margin-top:.3rem;">CGPA:{lp['cgpa']}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    if _is_admin():
        PAGES = {
            "\U0001f3e0  Dashboard":             "dashboard",
            "\U0001f4c2  Dataset Upload":        "dataset_upload",
            "\U0001f465  Student Submissions":   "submissions",
            "\U0001f3af  AI Resume Insights":    "prediction",
            "\U0001f6e0\ufe0f  Skill Recommendations": "skills",
            "\U0001f3e2  Company Eligibility":   "companies",
            "\U0001f464  My Account":            "account",
        }
    else:
        PAGES = {
            "\U0001f3e0  Dashboard":             "dashboard",
            "\U0001f3af  AI Resume Insights":    "prediction",
            "\U0001f6e0\ufe0f  Skill Recommendations": "skills",
            "\U0001f3e2  Company Eligibility":   "companies",
            "\U0001f464  My Account":            "account",
        }

    menu_items = list(PAGES.keys())

    # Consume nav_target BEFORE the radio widget renders (avoids widget key conflict)
    nav_tgt = st.session_state.get("nav_target")
    if nav_tgt and nav_tgt in menu_items:
        default_idx = menu_items.index(nav_tgt)
        st.session_state.nav_target = None
    else:
        last = st.session_state.get("_last_page")
        default_idx = menu_items.index(last) if last in menu_items else 0

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 .6rem;">
            <div style="font-family:'Syne',sans-serif;font-size:1.25rem;
                        font-weight:800;color:#f0f9ff;letter-spacing:-.02em;">
                \U0001f4ca Placement Analysis
            </div>
            <div style="font-size:.72rem;color:#334155;text-transform:uppercase;
                        letter-spacing:.1em;margin-top:2px;">
                AI Intelligence System
            </div>
        </div>""", unsafe_allow_html=True)
        st.divider()

        user   = st.session_state.user_data
        name   = user.get("name", "User")
        dept   = user.get("department", "")
        dept_s = dept[:34] + "\u2026" if len(dept) > 36 else dept

        role_lbl = (
            '<span class="badge badge-admin">\U0001f6e1\ufe0f Admin</span>'
            if _is_admin()
            else '<span class="badge badge-green">\U0001f393 Student</span>'
        )
        r_status = ""
        if _is_student():
            r_status = (
                '<span class="badge badge-green" style="font-size:.68rem;">\U0001f4c4 \u2713</span>'
                if st.session_state.resume_ready
                else '<span class="badge badge-orange" style="font-size:.68rem;">\U0001f4c4 needed</span>'
            )

        st.markdown(f"""
        <div style="padding:.7rem .9rem;background:rgba(255,255,255,.04);
                    border:1px solid rgba(255,255,255,.07);border-radius:10px;margin-bottom:1rem;">
            <div style="font-size:.9rem;font-weight:600;color:#e2e8f0;">{name}</div>
            <div style="font-size:.75rem;color:#64748b;margin-top:1px;">{user.get('email','')}</div>
            {"" if _is_admin() else f'<div style="font-size:.72rem;color:#475569;margin-top:1px;">{dept_s}</div>'}
            {"" if _is_admin() else f'<div style="font-size:.72rem;color:#334155;margin-top:2px;">{user.get("current_year","")}</div>'}
            <div style="margin-top:.5rem;display:flex;gap:4px;flex-wrap:wrap;">
                {role_lbl}{r_status}
            </div>
        </div>""", unsafe_allow_html=True)

        if _is_admin():
            sc       = get_submission_count()
            ds_badge = _dataset_source_badge()
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                        border-radius:8px;padding:.6rem .8rem;font-size:.78rem;margin-bottom:.8rem;">
                {ds_badge}<br>
                <span style="color:#6ee7b7;margin-top:.3rem;display:block;">
                    \U0001f4e5 <b>{sc}</b> submission{"s" if sc != 1 else ""}
                </span>
            </div>""", unsafe_allow_html=True)

        selected = st.radio(
            "",
            menu_items,
            index=default_idx,
            label_visibility="collapsed",
            key="sidebar_nav",
        )
        st.session_state["_last_page"] = selected

        st.divider()
        if st.button("\U0001f6aa  Logout", use_container_width=True):
            _logout()

        st.markdown("""
        <div style="text-align:center;font-size:.68rem;color:#1e293b;margin-top:.8rem;">
            \u00a9 2025 Placement Analysis<br>AI Intelligence System v6.0
        </div>""", unsafe_allow_html=True)

    page = PAGES.get(selected, "dashboard")
    if   page == "dashboard":      dashboard_page()
    elif page == "dataset_upload": admin_dataset_page()
    elif page == "submissions":    submissions_page()
    elif page == "prediction":     prediction_page()
    elif page == "skills":         skills_page()
    elif page == "companies":      company_page()
    elif page == "account":        account_page()
    else:                          dashboard_page()


if __name__ == "__main__":
    main()
