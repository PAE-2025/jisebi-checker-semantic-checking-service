from fastapi import APIRouter, HTTPException
from models import JournalSearchRequest, JournalResult
from services.scopus_service import ScopusSearch
from services.similarity import calculate_similarity, extract_common_keywords

router = APIRouter()

@router.post("/search", response_model=dict)
async def search_journals(request: JournalSearchRequest):
    """ Endpoint untuk mencari jurnal dan menghitung similarity """
    scopus = ScopusSearch(request.title)
    journal_results = scopus.search_journals()

    if not journal_results:
        raise HTTPException(status_code=404, detail="No journals found.")

    abstracts = scopus.get_abstracts(journal_results)
    similarities = calculate_similarity(request.abstract, abstracts)

    # Gabungkan hasil dengan similarity score
    for journal in journal_results:
        match = next((s for s in similarities if s["scopus_id"] == journal["scopus_id"]), None)
        if match:
            journal["similarity"] = match["similarity"]
            journal["abstract"] = abstracts.get(journal["scopus_id"], "No abstract available")
            journal["common_keywords"] = extract_common_keywords(request.abstract, journal["abstract"])

    # Urutkan berdasarkan similarity tertinggi
    journal_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)


    return {
        "query": request.title,
        "num_results": len(journal_results),
        "journals": [JournalResult(**journal) for journal in journal_results]
    }
