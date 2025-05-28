import pytest
from httpx import AsyncClient
from src.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Article Checker API"}

@pytest.mark.asyncio
async def test_input_article(client):
    article_data = {
        "title": "Test Article",
        "abstract": "This is a test abstract with enough words to process. It should have a background, objective, methods, results, and conclusions.",
        "keywords": "test, abstract"
    }
    response = await client.post("/input_article/", json=article_data)
    assert response.status_code == 200
    assert "title" in response.json()
    assert response.json()["title"] == "Test Article"
    assert "nlp_result" in response.json()
    assert "keyword_results" in response.json()
    assert "common_keywords" in response.json()