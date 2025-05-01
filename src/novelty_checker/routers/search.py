from fastapi import APIRouter, HTTPException
from models import JournalSearchRequest, JournalResult
from src.novelty_checker.services.crossref_service import CrossRefSearch
from src.novelty_checker.services.similarity import calculate_similarity, extract_common_keywords

router = APIRouter()

@router.post("/search", response_model=dict)
async def search_journals(request: JournalSearchRequest):
    """ Endpoint untuk mencari jurnal dan menghitung similarity """
    crossref = CrossRefSearch(request.title)
    journal_results = crossref.search_journals()

    if not journal_results:
        raise HTTPException(status_code=404, detail="No journals found.")

    # Ambil abstrak dari hasil pencarian
    abstracts = {journal["doi"]: journal["abstract"] for journal in journal_results}

    # Hitung similarity
    similarities = calculate_similarity(request.abstract, abstracts)

    # Gabungkan hasil pencarian dengan similarity score dan common keywords
    for journal in journal_results:
        match = next((s for s in similarities if s["doi"] == journal["doi"]), None)
        journal["similarity"] = round(match["similarity"] * 100, 2) if match else 0

        # Pastikan abstract adalah string sebelum diproses
        abstract_text = abstracts.get(journal["doi"], "")
        journal["abstract"] = abstract_text

        # Ekstrak common keywords dari abstrak
        journal["common_keywords"] = extract_common_keywords(abstract_text)


    # Urutkan berdasarkan similarity tertinggi
    journal_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

    return {
        "query": request.title,
        "num_results": len(journal_results),
        "journals": [JournalResult(**journal) for journal in journal_results]
    }
