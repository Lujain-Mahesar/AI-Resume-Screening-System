# Resume Screening System

An NLP and Machine Learning-based resume screening tool that ranks resumes against a job description using TF-IDF vectorization and cosine similarity. Built with Streamlit for an interactive web interface and Plotly for visual analytics.

**Author:** Lujain Mahesar (2312120)
**Course:** BSCS 6A | AI Theory Project

---

## Features

- Upload multiple PDF resumes at once
- Paste any job description for comparison
- Automatic text cleaning, tokenization, and TF-IDF vectorization
- Cosine similarity scoring between resumes and job description
- Keyword-based skill extraction and gap analysis (missing skills)
- Composite scoring system (similarity + skill match + skill diversity)
- Match categorization: Strong / Medium / Weak
- Interactive visualizations: score comparison, match distribution, score breakdown, skill radar chart
- Export ranked results as a downloadable CSV

---

## Project Structure

ai_resume_screener/
├── app.py              -> main Streamlit app
├── utils.py            -> NLP and ML helper functions
├── requirements.txt    -> required packages
└── README.md           -> this file
---

## How to Run

**Step 1 — Install packages:**
```bash
pip install -r requirements.txt
```

**Step 2 — Run the app:**
```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`

---

## How to Use

1. Upload one or more PDF resumes
2. Paste the job description in the text box
3. Click **Analyze Resumes**
4. View ranked results, charts, and download CSV

---

## Concepts Used

- Text preprocessing — lowercase, stopword removal, tokenization
- TF-IDF vectorization — converts text into numeric vectors
- Cosine similarity — measures how closely a resume matches the job description
- Keyword-based skill extraction
- Composite scoring out of 100 based on similarity and skill matches
- Match categories — Strong, Medium, Weak

---

## Packages Used

| Package        | Purpose                          |
|----------------|-----------------------------------|
| streamlit      | Web interface                     |
| scikit-learn   | TF-IDF and cosine similarity      |
| pandas         | Data handling and CSV export      |
| numpy          | Numerical calculations            |
| plotly         | Charts and visualizations         |
| PyMuPDF        | PDF text extraction               |
| pdfminer.six   | Backup PDF reader                 |

---

## Future Improvements

- Support for DOCX resume uploads
- Named Entity Recognition for better skill/experience extraction
- Customizable skill dictionaries per industry
- Deploy as a live demo on Streamlit Cloud

---

## License

This project is licensed under the MIT License.
