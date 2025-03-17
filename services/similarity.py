from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """ Membersihkan teks dengan tokenisasi, stopwords removal, dan stemming """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Bersihkan teks dengan menghapus karakter selain huruf dan spasi
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())

    # Tokenisasi kata
    words = word_tokenize(text)

    # Hapus stopwords tetapi tetap menjaga kata penting
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(filtered_words)

def calculate_similarity(user_abstract, abstracts):
    """ Menghitung similarity antara abstrak pengguna dan jurnal yang ditemukan """
    processed_user_abstract = preprocess_text(user_abstract)
    vectorizer = TfidfVectorizer()


    similarities = []
    for scopus_id, abstract in abstracts.items():
        processed_abstract = preprocess_text(abstract)
        print(f"User Abstract: {processed_user_abstract}")
        # print(f"Jurnal Abstracts: {abstracts}")
        tfidf_matrix = vectorizer.fit_transform([processed_user_abstract, processed_abstract])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0][1] * 100
        similarities.append({"scopus_id": scopus_id, "similarity": round(similarity_score, 2)})

    return similarities

def extract_common_keywords(query_abstract: str, journal_abstract: str, top_n: int = 5):
    """ Mengekstrak kata kunci umum dari dua abstrak dengan menghapus stopwords """
    # Hilangkan karakter non-huruf dan ubah menjadi huruf kecil
    words_query = [word for word in re.findall(r'\b\w+\b', query_abstract.lower()) if word not in stop_words]
    words_journal = [word for word in re.findall(r'\b\w+\b', journal_abstract.lower()) if word not in stop_words]

    # Hitung frekuensi kata dalam setiap abstrak
    query_freq = Counter(words_query)
    journal_freq = Counter(words_journal)

    # Cari kata yang sama di kedua abstrak
    common_words = set(query_freq.keys()) & set(journal_freq.keys())

    # Urutkan berdasarkan jumlah kemunculan dalam kedua abstrak
    common_keywords = sorted(common_words, key=lambda word: query_freq[word] + journal_freq[word], reverse=True)

    return common_keywords[:top_n]  # Ambil top-N kata kunci
