from fastapi import APIRouter
from src.abstract_checking.process import process_data, keywords_check, abstract_sentences, extract_common_keywords  # Perbarui impor
from src.abstract_checking.models import ArticleInput

router = APIRouter()

# Endpoint untuk input artikel manual
@router.post("/input_article")
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