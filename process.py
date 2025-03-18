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
import os

# Download NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Path ke file data pelatihan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "new_dt2.csv")

# Load and preprocess data
data = pd.read_csv(file_path)
data = data.dropna()
AbstractIs = data["Abstract"]

# Define regex patterns for labels
patterns = [
    (
        r"(?:Background|Introduction|Introductions|Problem|Problems|Motivation|Motivations)[:\-]?\s*(.*?)(?=\s*[.!?])",
        "Background",
    ),
    (
        r"(?:Purpose|Purposes|Goals|Goal|Objective|Objectives|Aims|Aim)[:\-]?\s*(.*?)(?=\s*[.!?])",
        "Objective",
    ),
    (
        r"(?:Approach|Method|Methods|Design|Designs)[:\-]?\s*(.*?)(?=\s*[.!?])",
        "Methods",
    ),
    (r"(?:Findings|Finding|Result|Results)[:\-]?\s*(.*?)(?=\s*[.!?])", "Results"),
    (
        r"(?:Implications|Implication|Conclusion|Conclusions)[:\-]?\s*(.*?)(?=\s*[.!?])",
        "Conclusions",
    ),
]

# Stopwords and stemming
stop_words = set(stopwords.words("english"))
stop_words.update(string.punctuation)
stop_words.update(map(str, range(10)))
stemmer = PorterStemmer()

# Extract relevant content for each label
dictionary_content = defaultdict(list)
compiling_content = [(re.compile(pattern), label) for pattern, label in patterns]

for abstract in AbstractIs:
    for pattern, label in compiling_content:
        match = pattern.findall(abstract)
        if match:
            dictionary_content[label].extend(match)

# Prepare training data
training_sentences = []
training_labels = []

for label, contents in dictionary_content.items():
    for sentence in contents:
        training_sentences.append(sentence)
        training_labels.append(label)

# Vectorize training sentences using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=100)

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

# Define domain-specific keywords
domain_specific_keys = {
    "Background": [
        "context",
        "problem",
        "introduction",
        "motivation",
        "increase",
        "trend",
        "study",
        "background",
    ],
    "Objective": ["goal", "aim", "purpose", "objective", "understanding", "aimed"],
    "Methods": ["methods", "approach", "design", "methodology", "process", "technique"],
    "Results": ["result", "results", "finding", "analysis", "outcome"],
    "Conclusions": [
        "conclusions", "conclusion",
        "implication",
        "summary",
        "impact",
        "inform",
        "understand",
    ],
}

# Augment domain-specific keywords with top 10 words for each label
for label in domain_specific_keys.keys():
    domain_specific_keys[label].extend(top_words[label])

# Augment training sentences with domain-specific keywords
augmented_training_sentences = []
for label, contents in dictionary_content.items():
    domain_keywords = " ".join(domain_specific_keys[label])
    for sentence in contents:
        augmented_sentence = sentence + " " + domain_keywords
        augmented_training_sentences.append(augmented_sentence)

# Train the model once at startup
X = vectorizer.fit_transform(augmented_training_sentences)
X_train, X_test, y_train, y_test = train_test_split(
    X, training_labels, test_size=0.2, random_state=42
)

svm_classifier = SVC(kernel="linear")
knn_classifier = KNeighborsClassifier(n_neighbors=5)
log_reg_classifier = LogisticRegression(max_iter=1000)

voting_classifier = VotingClassifier(
    estimators=[
        ("svm", svm_classifier),
        ("knn", knn_classifier),
        ("log_reg", log_reg_classifier),
    ],
    voting="hard",
)
voting_classifier.fit(X_train, y_train)

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