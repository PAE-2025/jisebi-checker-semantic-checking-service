import re
import string
from collections import defaultdict
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
nltk.download("punkt_tab")
nltk.download("stopwords")

# Path ke file data pelatihan dan model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "new_dt2.csv")
model_path = os.path.join(BASE_DIR, "voting_classifier.joblib")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")

# Load and preprocess data
data = pd.read_csv(file_path)
data = data.dropna()
AbstractIs = data["Abstract"]

# Define regex patterns for labels (diperluas untuk menangkap lebih banyak variasi)
patterns = [
    (
        r"(?:Background|Introduction|Introductions|Problem|Problems|Motivation|Motivations|Context|Overview)[:\-]?\s*(.*?)(?=\s*[.!?](?:\s|$))",
        "Background",
    ),
    (
        r"(?:Purpose|Purposes|Goals|Goal|Objective|Objectives|Aims|Aim|Intent|Intention)[:\-]?\s*(.*?)(?=\s*[.!?](?:\s|$))",
        "Objective",
    ),
    (
        r"(?:Approach|Method|Methods|Design|Designs|Methodology|Technique|Techniques|Procedure|Procedures)[:\-]?\s*(.*?)(?=\s*[.!?](?:\s|$))",
        "Methods",
    ),
    (
        r"(?:Findings|Finding|Result|Results|Outcome|Outcomes|Analysis|Analyses)[:\-]?\s*(.*?)(?=\s*[.!?](?:\s|$))",
        "Results",
    ),
    (
        r"(?:Implications|Implication|Conclusion|Conclusions|Summary|Summaries|Impact|Impacts)[:\-]?\s*(.*?)(?=\s*[.!?](?:\s|$))",
        "Conclusions",
    ),
]

# Stopwords and stemming
stop_words = set(stopwords.words("english"))
# Kurangi stopwords yang dihapus untuk mempertahankan beberapa kata penting
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

# Periksa distribusi data sebelum balancing
print("Distribusi data per kategori sebelum balancing:")
for label, contents in dictionary_content.items():
    print(f"{label}: {len(contents)} kalimat")

# Prepare training data with balancing
training_sentences = []
training_labels = []

# Batasi jumlah kalimat per kategori untuk menyeimbangkan data
max_samples_per_category = 9000  # Pilih jumlah yang mendekati kategori terkecil (Conclusions: 8982)

for label, contents in dictionary_content.items():
    sampled_contents = contents[:max_samples_per_category]  # Ambil hanya max_samples_per_category kalimat
    for sentence in sampled_contents:
        training_sentences.append(sentence)
        training_labels.append(label)

# Periksa distribusi data setelah balancing
print("\nDistribusi data per kategori setelah balancing:")
for label in set(training_labels):
    print(f"{label}: {training_labels.count(label)} kalimat")

# Vectorize training sentences using TF-IDF (ditingkatkan)
vectorizer = TfidfVectorizer(
    stop_words=list(stop_words),  # Konversi ke list
    max_features=500,  # Tingkatkan jumlah fitur
    ngram_range=(1, 2),  # Gunakan unigram dan bigram
)

# Get top 10 words for each label based on TF-IDF scores
X = vectorizer.fit_transform(training_sentences)
top_words = {}
for label, contents in dictionary_content.items():
    label_vector = vectorizer.transform(contents)
    tfidf_scores = label_vector.sum(axis=0).A1
    word_scores = {
        word: tfidf_scores[idx]
        for idx, word in enumerate(vectorizer.get_feature_names_out())
    }
    top_words[label] = sorted(word_scores, key=word_scores.get, reverse=True)[:10]

# Define domain-specific keywords (dikurangi untuk menghindari bias berlebihan)
domain_specific_keys = {
    "Background": ["context", "problem", "introduction", "motivation"],
    "Objective": ["goal", "aim", "purpose", "objective"],
    "Methods": ["method", "approach", "design", "technique"],
    "Results": ["result", "finding", "outcome", "analysis"],
    "Conclusions": ["conclusion", "implication", "summary", "impact"],
}

# Augment domain-specific keywords with top 5 words for each label (dikurangi dari 10)
for label in domain_specific_keys.keys():
    domain_specific_keys[label].extend(top_words[label][:5])

# Augment training sentences with domain-specific keywords (opsional, dikurangi intensitasnya)
augmented_training_sentences = []
for label, contents in dictionary_content.items():
    domain_keywords = " ".join(domain_specific_keys[label])
    for sentence in contents[:max_samples_per_category]:  # Gunakan data yang sudah diseimbangkan
        # Hanya tambahkan domain keywords ke 50% kalimat untuk mengurangi bias
        if len(augmented_training_sentences) % 2 == 0:
            augmented_sentence = sentence + " " + domain_keywords
        else:
            augmented_sentence = sentence
        augmented_training_sentences.append(augmented_sentence)

# Cek apakah model sudah ada
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    print("Loading pre-trained model and vectorizer...")
    voting_classifier = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Pre-trained model and vectorizer loaded")
else:
    # Train the model once at startup
    X = vectorizer.fit_transform(augmented_training_sentences)
    X_train, X_test, y_train, y_test = train_test_split(
        X, training_labels, test_size=0.2, random_state=42
    )

    # Definisikan model dengan hyperparameter yang lebih baik
    svm_classifier = SVC(kernel="linear", probability=True)  # Tambahkan probability=True untuk soft voting
    knn_classifier = KNeighborsClassifier(n_neighbors=3)  # Kurangi n_neighbors untuk lebih sensitif
    log_reg_classifier = LogisticRegression(max_iter=1000, C=1.0)  # Tambahkan C untuk regularisasi

    voting_classifier = VotingClassifier(
        estimators=[
            ("svm", svm_classifier),
            ("knn", knn_classifier),
            ("log_reg", log_reg_classifier),
        ],
        voting="soft",  # Gunakan soft voting
    )
    voting_classifier.fit(X_train, y_train)

    # Evaluasi model
    y_pred = voting_classifier.predict(X_test)
    print("\nLaporan Evaluasi Model:")
    print(classification_report(y_test, y_pred))

    # Simpan model dan vectorizer
    print("Saving model and vectorizer...")
    joblib.dump(voting_classifier, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model and vectorizer saved")

# Functions for processing input
def process_data(abstract):
    sentences = sent_tokenize(abstract)
    input_vectors = vectorizer.transform(sentences)
    predicted_labels = voting_classifier.predict(input_vectors)
    results = [(sentence, predicted_labels[i]) for i, sentence in enumerate(sentences)]
    return results

def keywords_check(keywords, abstract):
    keyword_list = [keyword.strip() for keyword in keywords.split(',')]
    abstract_lower = abstract.lower()
    result_key = []
    for keyword in keyword_list:
        if keyword.lower() in abstract_lower:
            result_key.append(f"{keyword} exists.")
        else:
            result_key.append(f"{keyword} not found in the abstract!")
    return result_key

def abstract_sentences(abstract):
    word_count = len(abstract.split())
    if word_count > 450:
        return "Words more than 450. Please reduce your abstract."
    return None