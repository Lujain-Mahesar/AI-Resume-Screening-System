# Resume Screening System
**Lujain Mahesar-(2312120), Aiman Imran-(2312106)** 
**BSCS 6A | AI Theory Project**

---

## Project Structure

```
ai_resume_screener/
├── app.py              → main Streamlit app
├── utils.py            → NLP and ML helper functions
├── requirements.txt    → required packages
└── README.md           → this file
```

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

- Text preprocessing - lowercase, stopword removal, tokenization
- TF-IDF vectorization - converts text into numeric vectors
- Cosine similarity - measures how closely a resume matches the job description
- Keyword-based skill extraction
- Score out of 100 based on similarity and skill matches
- Match categories - Strong, Medium, Weak

---

## Packages Used

|      Package         |       Purpose        |
|----------------------|----------------------|
| streamlit            | Web interface.       |
| scikit-learn         | TF-IDF and cosine    |
|                      | similarity.          |
| pandas               | Data handling and    |
|                      | CSV export.          |
| numpy                | Numerical calculations|
| plotly               | Charts.              |
| PyMuPDF              | PDF text extraction. |
| pdfminer.six         | Backup PDF reader.   |
