import requests

class CrossRefSearch:
    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, query):
        self.query = query

    def search_journals(self):
        """ Mencari jurnal berdasarkan judul menggunakan CrossRef API """
        params = {"query": self.query}
        response = requests.get(self.BASE_URL, params=params)

        if response.status_code != 200:
            raise Exception(f"Error fetching data from CrossRef: {response.status_code}")

        data = response.json()
        journal_results = []

        for item in data.get("message", {}).get("items", []):
            title = item.get("title", [""])[0]
            doi = item.get("DOI", "")
            abstract = item.get("abstract", "")

            url = f"https://doi.org/{doi}" if doi else ""

            journal_results.append({
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "url": url
            })

        return journal_results
