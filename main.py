from fastapi import FastAPI, UploadFile, File
from process import process_data, keywords_check, abstract_sentences, extract_common_keywords
import pandas as pd
import io
from pydantic import BaseModel

app = FastAPI()

# Model untuk input artikel manual
class ArticleInput(BaseModel):
    title: str
    abstract: str
    keywords: str

# Endpoint untuk menguji API
@app.get("/")
async def root():
    return {"message": "Welcome to Article Checker API"}

# Endpoint untuk input artikel manual
@app.post("/input_article/")
async def input_article(article: ArticleInput):
    # Proses abstrak
    nlp_result = process_data(article.abstract)
    keyword_results = keywords_check(article.keywords, article.abstract)
    word_count_result = abstract_sentences(article.abstract)

    # Struktur ulang hasil NLP untuk respons JSON
    label_sentences = {"Background": [], "Objective": [], "Methods": [], "Results": [], "Conclusions": []}
    for sentence, label in nlp_result:
        if label in label_sentences:
            label_sentences[label].append(sentence)

    results = [
        {"label": label, "sentences": sentences, "empty": not bool(sentences)}
        for label, sentences in label_sentences.items()
    ]

    # Mengekstrak kata kunci umum dari abstrak dengan common keyword extractor
    common_keywords = extract_common_keywords(article.abstract, " ".join(label_sentences["Background"] + label_sentences["Objective"] + label_sentences["Methods"] + label_sentences["Results"] + label_sentences["Conclusions"]))

    # Respons JSON
    response = {
        "title": article.title,
        "keyword_results": keyword_results,
        "nlp_result": results,
        "common_keywords": common_keywords,
    }
    if word_count_result:  # Jika ada pesan peringatan tentang jumlah kata
        response["word_count_warning"] = word_count_result
    else:
        response["word_count"] = len(article.abstract.split())

    return response

# Endpoint untuk mengunggah CSV
@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Baca file CSV
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode('utf-8')))

    results = []
    for index, row in df.iterrows():
        title = row["Title"]
        abstract = row["Abstract"]
        keywords = row["Author Keywords"]

        # Proses abstrak
        nlp_result = process_data(abstract)
        keyword_results = keywords_check(keywords, abstract)
        word_count_result = abstract_sentences(abstract)

        # Struktur ulang hasil NLP
        label_sentences = {"Background": [], "Objective": [], "Methods": [], "Results": [], "Conclusions": []}
        for sentence, label in nlp_result:
            if label in label_sentences:
                label_sentences[label].append(sentence)

        result = [
            {"label": label, "sentences": sentences, "empty": not bool(sentences)}
            for label, sentences in label_sentences.items()
        ]

        # Mengekstrak kata kunci umum dari abstrak dengan common keyword extractor
        common_keywords = extract_common_keywords(abstract, " ".join(label_sentences["Background"] + label_sentences["Objective"] + label_sentences["Methods"] + label_sentences["Results"] + label_sentences["Conclusions"]))

        # Buat respons untuk artikel ini
        article_response = {
            "title": title,
            "nlp_result": result,
            "keyword_results": keyword_results,
            "common_keywords": common_keywords,
        }
        if word_count_result:
            article_response["word_count_warning"] = word_count_result
        else:
            article_response["word_count"] = len(abstract.split())

        results.append(article_response)

    return {"results": results}
