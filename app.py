# Resume Screening System
# Author: Lujain Mahesar (2312120)
# BSCS 6A

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from utils import (
    extract_text_from_pdf,
    clean_text,
    compute_tfidf_similarity,
    extract_skills,
    score_resume,
    get_match_category,
    COMMON_SKILLS
)

st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #111827 60%, #0d1117 100%);
        color: #e2e8f0;
    }
    .hero-banner {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-banner h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        color: #fff;
        margin: 0;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .section-header {
        font-family: 'Space Mono', monospace;
        color: #a5b4fc;
        font-size: 1.05rem;
        border-left: 4px solid #6366f1;
        padding-left: 0.8rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 0.5rem;
    }
    .rank-badge {
        display: inline-block;
        background: #6366f1;
        color: #fff;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        padding: 2px 10px;
        border-radius: 20px;
        margin-bottom: 0.5rem;
    }
    .match-strong { color: #34d399; font-weight: 700; }
    .match-medium { color: #fbbf24; font-weight: 700; }
    .match-weak   { color: #f87171; font-weight: 700; }
    .skill-pill {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.5);
        color: #a5b4fc;
        font-size: 0.78rem;
        padding: 2px 10px;
        border-radius: 20px;
        margin: 2px 3px;
    }
    .metric-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-box .value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        color: #a5b4fc;
    }
    .metric-box .label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .timestamp-box {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
        text-align: right;
        margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(99,102,241,0.2);
    }
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 0.95rem;
        padding: 0.6rem 1.4rem;
    }
    .stProgress > div > div > div { background: linear-gradient(90deg,#6366f1,#06b6d4); }
    .info-box {
        background: rgba(6,182,212,0.1);
        border: 1px solid rgba(6,182,212,0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #67e8f9;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Resume Screener")
    st.markdown("---")
    st.markdown("**Steps:**")
    st.markdown("""
    1. Upload PDF resumes
    2. Paste the job description
    3. Click Analyze
    4. View results and charts
    """)
    st.markdown("---")
    st.markdown("**Score Categories:**")
    st.markdown("🟢 Strong — 60 and above")
    st.markdown("🟡 Medium — 35 to 59")
    st.markdown("🔴 Weak — below 35")


st.markdown("""
<div class="hero-banner">
    <h1>Resume Screening System</h1>
    <p>Uses NLP and Machine Learning to rank resumes against a job description</p>
</div>
""", unsafe_allow_html=True)


col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<p class="section-header">Upload Resumes</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more PDF resumes",
        type=["pdf"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")

with col_right:
    st.markdown('<p class="section-header">Job Description</p>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste the job description here",
        height=200,
        placeholder="e.g. Looking for a Python developer with machine learning experience, SQL, data analysis skills..."
    )
    if job_description.strip():
        st.caption(f"{len(job_description.split())} words")

st.markdown("---")
run_analysis = st.button("Analyze Resumes", use_container_width=True)


# run analysis and save results to session state
if run_analysis:

    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()
    if not job_description.strip():
        st.error("Please paste a job description.")
        st.stop()

    st.markdown("### Processing...")
    progress = st.progress(0)
    status = st.empty()
    resumes = []

    for i, file in enumerate(uploaded_files):
        status.text(f"Reading {file.name}...")
        raw_text = extract_text_from_pdf(file)

        if not raw_text.strip():
            st.warning(f"Could not read {file.name}, skipping.")
            continue

        resumes.append({
            "filename": file.name,
            "raw_text": raw_text,
            "cleaned_text": clean_text(raw_text),
            "skills": extract_skills(raw_text)
        })
        progress.progress(int((i + 1) / len(uploaded_files) * 60))

    if not resumes:
        st.error("No readable resumes found.")
        st.stop()

    status.text("Calculating similarity scores...")
    similarities = compute_tfidf_similarity(job_description, [r["cleaned_text"] for r in resumes])

    jd_lower = job_description.lower()
    jd_skills = [s for s in COMMON_SKILLS if s in jd_lower]

    for i, r in enumerate(resumes):
        sim = round(float(similarities[i]) * 100, 2)
        r["similarity"] = sim
        r["score"] = score_resume(sim, r["skills"], job_description)
        r["category"] = get_match_category(r["score"])
        r["missing_skills"] = [s for s in jd_skills if s not in r["skills"]]
        boosted_sim = min(sim * 3.5, 100)
        r["base_score"] = round(boosted_sim * 0.55, 2)
        total_jd_skills = max(len(jd_skills), 1)
        matched = sum(1 for s in r["skills"] if s in jd_lower)
        r["skill_bonus"] = round((matched / total_jd_skills) * 100 * 0.35, 2)
        r["diversity_bonus"] = round(min(len(r["skills"]) * 0.8, 10.0), 2)

    progress.progress(90)
    resumes.sort(key=lambda x: x["score"], reverse=True)
    progress.progress(100)
    status.empty()

    # save everything to session state so slider doesn't reset results
    st.session_state.resumes = resumes
    st.session_state.jd_lower = jd_lower
    st.session_state.analysis_time = datetime.now().strftime("%d %B %Y  •  %I:%M %p")


# show results if we have them in session state
if "resumes" in st.session_state and st.session_state.resumes:

    resumes  = st.session_state.resumes
    jd_lower = st.session_state.jd_lower

    st.markdown(f'<div class="timestamp-box">Analyzed on: {st.session_state.analysis_time}</div>', unsafe_allow_html=True)
    st.success(f"Done. {len(resumes)} resume(s) ranked.")
    st.markdown("---")

    st.markdown('<p class="section-header">Summary</p>', unsafe_allow_html=True)

    scores = [r["score"] for r in resumes]
    cats   = [r["category"] for r in resumes]
    strong = cats.count("Strong Match")
    medium = cats.count("Medium Match")
    weak   = cats.count("Weak Match")

    def metric_html(value, label):
        return f"""<div class="metric-box">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>"""

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(metric_html(len(resumes),            "Total"),     unsafe_allow_html=True)
    m2.markdown(metric_html(f"{max(scores):.0f}",    "Top Score"), unsafe_allow_html=True)
    m3.markdown(metric_html(f"{np.mean(scores):.0f}","Average"),   unsafe_allow_html=True)
    m4.markdown(metric_html(strong,                  "Strong"),    unsafe_allow_html=True)
    m5.markdown(metric_html(weak,                    "Weak"),      unsafe_allow_html=True)

    st.markdown("---")

    # filter slider — works now because results are in session state
    st.markdown('<p class="section-header">Filter</p>', unsafe_allow_html=True)
    min_score = st.slider("Minimum score to display:", 0, 100, 0, step=5)
    filtered  = [r for r in resumes if r["score"] >= min_score]
    st.caption(f"Showing {len(filtered)} of {len(resumes)} resumes")
    st.markdown("---")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<p class="section-header">Score Comparison</p>', unsafe_allow_html=True)
        bar_colors = []
        for r in filtered:
            if r["category"] == "Strong Match": bar_colors.append("#34d399")
            elif r["category"] == "Medium Match": bar_colors.append("#fbbf24")
            else: bar_colors.append("#f87171")

        fig_bar = go.Figure(go.Bar(
            x=[r["filename"].replace(".pdf", "") for r in filtered],
            y=[r["score"] for r in filtered],
            marker_color=bar_colors,
            text=[f"{r['score']:.0f}" for r in filtered],
            textposition="outside"
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="DM Sans"),
            yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.07)"),
            xaxis=dict(tickangle=-30), margin=dict(t=20, b=10), showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown('<p class="section-header">Match Distribution</p>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Strong Match", "Medium Match", "Weak Match"],
            values=[strong, medium, weak],
            marker_colors=["#34d399", "#fbbf24", "#f87171"],
            hole=0.5, textinfo="percent", textposition="inside"
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="DM Sans"),
            margin=dict(t=40, b=40, l=40, r=40), height=420,
            showlegend=True,
            legend=dict(orientation="v", x=1.05, y=0.5, font=dict(color="#e2e8f0", size=12))
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    st.markdown('<p class="section-header">Score Breakdown</p>', unsafe_allow_html=True)
    names_bd = [r["filename"].replace(".pdf", "") for r in filtered]
    fig_bd = go.Figure()
    fig_bd.add_trace(go.Bar(name="Similarity Score (55%)", x=names_bd, y=[r["base_score"] for r in filtered], marker_color="#6366f1"))
    fig_bd.add_trace(go.Bar(name="Skill Match (35%)",      x=names_bd, y=[r["skill_bonus"] for r in filtered], marker_color="#06b6d4"))
    fig_bd.add_trace(go.Bar(name="Skill Diversity (10%)",  x=names_bd, y=[r["diversity_bonus"] for r in filtered], marker_color="#8b5cf6"))
    fig_bd.update_layout(
        barmode="stack", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="DM Sans"),
        yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.07)"),
        xaxis=dict(tickangle=-30),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
        margin=dict(t=20, b=10)
    )
    st.plotly_chart(fig_bd, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Ranked Results</p>', unsafe_allow_html=True)

    if not filtered:
        st.warning("No resumes match this score. Lower the filter to see results.")

    for rank, r in enumerate(filtered, 1):
        cat = r["category"]
        cat_class = "match-strong" if cat == "Strong Match" else ("match-medium" if cat == "Medium Match" else "match-weak")
        cat_icon  = "🟢" if cat == "Strong Match" else ("🟡" if cat == "Medium Match" else "🔴")
        pills_html = "".join(f'<span class="skill-pill">{s}</span>' for s in r["skills"]) or "<em style='color:#64748b'>No skills found</em>"

        st.markdown(f"""
        <div class="result-card">
            <span class="rank-badge">#{rank}</span>
            <h4 style="margin:0.3rem 0 0.6rem;color:#e2e8f0">{r['filename']}</h4>
            <div style="display:flex;gap:2rem;margin-bottom:0.8rem;flex-wrap:wrap;">
                <div>
                    <span style="color:#94a3b8;font-size:0.85rem">Score</span><br>
                    <span style="font-family:'Space Mono',monospace;font-size:1.5rem;color:#a5b4fc">{r['score']:.0f}<span style="font-size:0.9rem">/100</span></span>
                </div>
                <div>
                    <span style="color:#94a3b8;font-size:0.85rem">Similarity</span><br>
                    <span style="font-family:'Space Mono',monospace;font-size:1.5rem;color:#a5b4fc">{r['similarity']:.1f}<span style="font-size:0.9rem">%</span></span>
                </div>
                <div>
                    <span style="color:#94a3b8;font-size:0.85rem">Category</span><br>
                    <span class="{cat_class}">{cat_icon} {cat}</span>
                </div>
                <div>
                    <span style="color:#94a3b8;font-size:0.85rem">Skills Found</span><br>
                    <span style="font-family:'Space Mono',monospace;font-size:1.5rem;color:#a5b4fc">{len(r['skills'])}</span>
                </div>
            </div>
            <div style="margin-bottom:0.4rem;font-size:0.82rem;color:#94a3b8">Detected Skills:</div>
            <div>{pills_html}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(r["score"]))

        if r["missing_skills"] and cat in ["Weak Match", "Medium Match"]:
            missing = "  ".join([f"`{s}`" for s in r["missing_skills"][:8]])
            st.markdown(f"Missing skills: {missing}")

        st.markdown("")

    st.markdown("---")

    if len(resumes) >= 2:
        st.markdown('<p class="section-header">Skill Radar — Top 3</p>', unsafe_allow_html=True)
        top3      = resumes[:3]
        skill_set = [s for s in COMMON_SKILLS if s in jd_lower][:8]

        if skill_set:
            fig_radar = go.Figure()
            for r in top3:
                r_lower = r["raw_text"].lower()
                vals = [1 if s in r_lower else 0 for s in skill_set]
                vals += vals[:1]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=skill_set + skill_set[:1],
                    fill="toself",
                    name=r["filename"].replace(".pdf", ""),
                    opacity=0.75
                ))
            fig_radar.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False),
                           angularaxis=dict(color="#94a3b8")),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="DM Sans"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=30, b=10)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("---")

    st.markdown('<p class="section-header">Download Results</p>', unsafe_allow_html=True)

    df_export = pd.DataFrame([{
        "Rank":          rank,
        "Resume":        r["filename"],
        "Score":         round(r["score"], 2),
        "Similarity %":  round(r["similarity"], 2),
        "Category":      r["category"],
        "Skills Found":  ", ".join(r["skills"]),
        "Missing Skills": ", ".join(r["missing_skills"]),
        "Skill Count":   len(r["skills"])
    } for rank, r in enumerate(resumes, 1)])

    st.dataframe(df_export, use_container_width=True)
    st.download_button(
        label="Download as CSV",
        data=df_export.to_csv(index=False).encode("utf-8"),
        file_name="resume_results.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.markdown("""
    <div class="info-box">
        Upload PDF resumes on the left and paste the job description on the right, then click <b>Analyze Resumes</b>.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#475569">
        <div style="font-size:4rem">📂</div>
        <div style="font-size:1rem;margin-top:1rem">Upload resumes and a job description to begin</div>
    </div>
    """, unsafe_allow_html=True)
