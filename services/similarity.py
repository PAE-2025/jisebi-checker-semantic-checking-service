import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """ Membersihkan teks dengan tokenisasi, stopwords removal, dan stemming """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    filtered_words = [PorterStemmer().stem(word) for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

def calculate_similarity(user_abstract, abstracts):
    """ Menghitung similarity antara abstrak yang diinput pengguna dan jurnal dari CrossRef """
    processed_abstracts = {doi: preprocess_text(abstract) for doi, abstract in abstracts.items()}

    vectorizer = TfidfVectorizer()
    corpus = [preprocess_text(user_abstract)] + list(processed_abstracts.values())

    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

    return [{"doi": doi, "similarity": float(score)} for doi, score in zip(processed_abstracts.keys(), similarity_scores)]

def extract_common_keywords(text, top_n=5):
    """ Ekstrak kata kunci umum dari satu teks abstrak """
    if not isinstance(text, str) or not text.strip():
        return []  # Jika teks kosong, return list kosong

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())

    # Filtering kata yang bukan alphanumeric & stopwords removal
    filtered_words = [PorterStemmer().stem(word) for word in words if word.isalnum() and word not in stop_words]

    # Hitung kata yang sering muncul
    word_freq = Counter(filtered_words)

    # Ambil top_n kata kunci yang paling sering muncul
    common_keywords = [word for word, _ in word_freq.most_common(top_n)]

    return common_keywords


