import re
import string
from collections import defaultdict, Counter
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import os
import joblib

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Path ke file data pelatihan dan model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
file_path = os.path.join(BASE_DIR, "data", "new_dt2.csv")
model_path = os.path.join(os.path.dirname(__file__), "voting_classifier.joblib")
vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.joblib")

# Load and preprocess data
data = pd.read_csv(file_path)
data = data.dropna()
AbstractIs = data["Abstract"]

# Define regex patterns for labels
patterns = [
    (r"(?:Background|Introduction|Problem|Motivation|Context)[:\-]?\s*(.*?)(?=\s*[.!?](?:\s|$))", "Background"),
    (r"(?:Purpose|Goals|Objective|Aims|Intent)[:\-]?\s*(.*?)(?=\s*[.!?](?:\s|$))", "Objective"),
    (r"(?:Method|Approach|Design|Technique|Procedure)[:\-]?\s*(.*?)(?=\s*[.!?](?:\s|$))", "Methods"),
    (r"(?:Findings|Result|Outcome|Analysis)[:\-]?\s*(.*?)(?=\s*[.!?](?:\s|$))", "Results"),
    (r"(?:Implication|Conclusion|Summary|Impact)[:\-]?\s*(.*?)(?=\s*[.!?](?:\s|$))", "Conclusions"),
]

# Stopwords and stemming
stop_words = set(stopwords.words("english"))
stop_words.difference_update({"aim", "method", "result", "background", "objective"})
stop_words.update(string.punctuation)
stop_words.update(map(str, range(10)))
stemmer = PorterStemmer()

# Extract relevant content for each label
dictionary_content = defaultdict(list)
compiling_content = [(re.compile(pattern, re.IGNORECASE), label) for pattern, label in patterns]

for abstract in AbstractIs:
    for pattern, label in compiling_content:
        match = pattern.findall(abstract)
        if match:
            dictionary_content[label].extend(match)

# Prepare training data with balancing
training_sentences = []
training_labels = []
max_samples_per_category = 9000

for label, contents in dictionary_content.items():
    sampled_contents = contents[:max_samples_per_category]
    for sentence in sampled_contents:
        training_sentences.append(sentence)
        training_labels.append(label)

# Vectorize training sentences using TF-IDF
vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=500, ngram_range=(1, 2))
X = vectorizer.fit_transform(training_sentences)

# Define model
svm_classifier = SVC(kernel="linear", probability=True)
knn_classifier = KNeighborsClassifier(n_neighbors=3)
log_reg_classifier = LogisticRegression(max_iter=1000, C=1.0)

voting_classifier = VotingClassifier(
    estimators=[("svm", svm_classifier), ("knn", knn_classifier), ("log_reg", log_reg_classifier)],
    voting="soft",
)

X_train, X_test, y_train, y_test = train_test_split(X, training_labels, test_size=0.2, random_state=42)

# Train and save model
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    voting_classifier.fit(X_train, y_train)
    joblib.dump(voting_classifier, model_path)
    joblib.dump(vectorizer, vectorizer_path)
else:
    voting_classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

# Function to classify abstract
def process_data(abstract):
    """Memproses abstrak dan mengelompokkan kalimat ke dalam label yang sesuai dengan ML & common keywords"""
    sentences = sent_tokenize(abstract)
    input_vectors = vectorizer.transform(sentences)
    predicted_labels = voting_classifier.predict(input_vectors)

    label_keywords = {
        "Background": ["historically", "previous studies", "prior work", "traditionally", "past research"],
        "Objective": ["aim", "goal", "purpose", "objective", "intend", "focus"],
        "Methods": ["method", "approach", "technique", "procedure", "experiment", "apply", "implement"],
        "Results": ["findings", "result", "evaluation", "analysis", "performance", "achieve"],
        "Conclusions": ["conclude", "summary", "implication", "future work", "suggest", "propose"]
    }

    labeled_sentences = []
    for i, sentence in enumerate(sentences):
        label_ml = predicted_labels[i]  # Hasil dari model ML
        label_scores = {label: 0 for label in label_keywords}  # Skor untuk setiap label

        # Hitung skor berdasarkan jumlah kata kunci yang cocok
        for label, keywords in label_keywords.items():
            label_scores[label] = sum(sentence.lower().count(keyword) for keyword in keywords)

        # Pilih label dengan skor tertinggi (jika tidak ada, pakai hasil ML)
        best_label = max(label_scores, key=label_scores.get)
        label_final = best_label if label_scores[best_label] > 0 else label_ml

        labeled_sentences.append((sentence, label_final))

    return labeled_sentences


# Function to check keywords
def keywords_check(keywords, abstract):
    if not keywords.strip():
        return ["No keywords provided."]
    
    keyword_list = [keyword.strip() for keyword in keywords.split(',') if keyword.strip()]
    if not keyword_list:
        return ["No valid keywords found."]
    
    abstract_lower = abstract.lower()
    result_key = [
        f"{keyword} exists." if keyword.lower() in abstract_lower else f"{keyword} not found in the abstract!"
        for keyword in keyword_list
    ]
    return result_key


# Function to check word count
def abstract_sentences(abstract):
    word_count = len(abstract.split())
    if word_count > 450:
        return "Words more than 450. Please reduce your abstract."
    return None

# Function to extract common keywords
def extract_common_keywords(query_abstract: str, journal_abstract: str, top_n: int = 5):
    words_query = [word for word in re.findall(r'\b\w+\b', query_abstract.lower()) if word not in stop_words]
    words_journal = [word for word in re.findall(r'\b\w+\b', journal_abstract.lower()) if word not in stop_words]
    query_freq = Counter(words_query)
    journal_freq = Counter(words_journal)
    common_words = set(query_freq.keys()) & set(journal_freq.keys())
    common_keywords = sorted(common_words, key=lambda word: query_freq[word] + journal_freq[word], reverse=True)
    return common_keywords[:top_n]
from nltk.corpus import wordnet

def get_synonyms(word):
    """Mengambil sinonim dari WordNet untuk memperluas cakupan pencocokan"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace("_", " "))
    return synonyms

def map_common_keywords_to_labels(common_keywords, label_keywords):
    """Mengaitkan common keywords dengan label yang relevan menggunakan sinonim"""
    label_common_keywords = {label: [] for label in label_keywords}

    for word in common_keywords:
        word_synonyms = get_synonyms(word)  # Ambil sinonimnya

        for label, keywords in label_keywords.items():
            # Jika ada kecocokan langsung atau dari sinonim, masukkan ke label
            if word in keywords or any(syn in keywords for syn in word_synonyms):
                label_common_keywords[label].append(word)
    
    return label_common_keywords


