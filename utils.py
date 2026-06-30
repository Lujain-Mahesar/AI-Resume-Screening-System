import re
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False


STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up",
    "down","in","out","on","off","over","under","again","further","then",
    "once","here","there","when","where","why","how","all","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","s","t","can","will","just","don","should",
    "now","d","ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn",
    "hadn","hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn",
    "wasn","weren","won","wouldn","also","would","could","may","might","shall",
    "use","using","used","well","work","works","working","make","makes","making",
    "good","great","strong","excellent","experience","years","year","team",
    "company","position","role","job","candidate","applicant","please","etc"
}


COMMON_SKILLS = [
    # programming languages
    "python","java","javascript","typescript","c++","c#","ruby","golang","rust",
    "kotlin","swift","scala","matlab","php","perl","bash","shell",

    # web development
    "html","css","react","angular","vue","nodejs","django","flask","fastapi",
    "spring","express","rest","api","graphql","bootstrap","tailwind","jquery",

    # data science and ml
    "machine learning","deep learning","neural network","nlp","natural language processing",
    "tensorflow","pytorch","keras","scikit-learn","pandas","numpy","matplotlib",
    "seaborn","plotly","data analysis","data science","statistics","regression",
    "classification","clustering","computer vision","opencv","bert","gpt",

    # databases
    "sql","mysql","postgresql","mongodb","redis","elasticsearch","sqlite",
    "oracle","cassandra","dynamodb","firebase",

    # cloud and devops
    "aws","azure","gcp","docker","kubernetes","jenkins","git","github","gitlab",
    "ci/cd","linux","unix","terraform","ansible","devops","mlops",

    # tools
    "agile","scrum","jira","tableau","power bi","excel","word","powerpoint",
    "hadoop","spark","kafka","airflow","dbt","streamlit","jupyter",

    # soft skills
    "leadership","communication","teamwork","problem solving","analytical",
    "project management","critical thinking","research","presentation",

    # domain specific
    "finance","accounting","marketing","sales","seo","cybersecurity","networking",
    "embedded systems","iot","blockchain","robotics","automation","testing",
    "quality assurance","ui/ux","figma","photoshop"
]


def extract_text_from_pdf(file_obj):
    raw_bytes = file_obj.read()

    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                return text
        except:
            pass

    if PDFMINER_AVAILABLE:
        try:
            text = pdfminer_extract(io.BytesIO(raw_bytes))
            return text or ""
        except:
            pass

    return ""


def clean_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)

    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]

    return " ".join(tokens)


def compute_tfidf_similarity(job_description, resume_texts):
    jd_clean = clean_text(job_description)
    corpus = [jd_clean] + resume_texts

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    return similarities.tolist()


def extract_skills(text):
    text_lower = text.lower()
    found_skills = []

    for skill in COMMON_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return sorted(set(found_skills))


def score_resume(similarity_percent, skills, job_description):
    jd_lower = job_description.lower()
    jd_skills = [s for s in COMMON_SKILLS if s in jd_lower]
    total_jd_skills = max(len(jd_skills), 1)

    # how many jd skills are in the resume
    matched_skills = [s for s in skills if s in jd_lower]
    skill_match_ratio = len(matched_skills) / total_jd_skills

    # boost cosine similarity to a more readable range
    # raw cosine on short texts is usually 0.05 - 0.35, so we scale it up
    boosted_similarity = min(similarity_percent * 3.5, 100)

    # weighted combination
    # 55% from boosted similarity, 35% from skill match ratio, 10% from skill diversity
    similarity_score = boosted_similarity * 0.55
    skill_score = skill_match_ratio * 100 * 0.35
    diversity_score = min(len(skills) * 0.8, 10)

    total = similarity_score + skill_score + diversity_score
    return round(min(total, 100.0), 2)


def get_match_category(score):
    if score >= 60:
        return "Strong Match"
    elif score >= 35:
        return "Medium Match"
    else:
        return "Weak Match"