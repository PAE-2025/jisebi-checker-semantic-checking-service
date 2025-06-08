import re
import string
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import os
import joblib

# Download NLTK data
# nltk.download("punkt")
# nltk.download("stopwords")

model_path = os.path.join(os.path.dirname(__file__), "resources/voting_classifier.joblib")
vectorizer_path = os.path.join(os.path.dirname(__file__), "resources/tfidf_vectorizer.joblib")

voting_classifier = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Stopwords and stemming
stop_words = set(stopwords.words("english"))
stop_words.difference_update({"aim", "method", "result", "background", "objective"})
stop_words.update(string.punctuation)
stop_words.update(map(str, range(10)))

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


