import time
from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
from elsapy.elsdoc import AbsDoc
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from src.config import SCOPUS_API_KEY
from bs4 import BeautifulSoup


class ScopusSearch:
    def __init__(self, query):
        self.query = query
        self.client = ElsClient(SCOPUS_API_KEY)

        # Inisialisasi WebDriver hanya sekali
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # Mode tanpa GUI
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    def search_journals(self):
        """ Mencari jurnal di Scopus berdasarkan query """
        doc_srch = ElsSearch(self.query, 'scopus')
        doc_srch.execute(self.client, get_all=False)

        results = []
        for doc in doc_srch.results[:10]:  # Ambil 10 hasil teratas
            scopus_id = doc["dc:identifier"].replace("SCOPUS_ID:", "")
            title = doc["dc:title"]
            scopus_url = doc.get("prism:url", "")

            results.append({"scopus_id": scopus_id, "title": title, "scopus_url": scopus_url})

        return results


    def get_abstracts(self, journal_list):
        """ Mengambil abstrak jurnal menggunakan scraping """
        abstracts = {}
        for journal in journal_list:
            scopus_url = journal.get("scopus_url")
            if not scopus_url:
                print(f"[WARNING] No Scopus URL found for {journal['scopus_id']}")
                abstracts[journal["scopus_id"]] = "No abstract found"
                continue

            scp_doc = AbsDoc(scp_id=journal["scopus_id"])
            # if scp_doc.read(self.client):
            #     # Debug: Cetak seluruh respons API untuk memeriksa struktur data
            #     print(f"Response for {journal['scopus_id']}:", scp_doc.data)

            print(f"[INFO] Scraping abstract for {journal['scopus_id']}...")

            abstract = self.scrape_abstract(scopus_url)
            abstracts[journal["scopus_id"]] = abstract or "No abstract found"
            print(abstract)

        return abstracts

    def scrape_abstract(self, scopus_url):

        try:
            self.driver.get(scopus_url)
            time.sleep(5)
            print(f"accessing: {scopus_url}")

            # Ambil seluruh halaman sumber HTML
            page_source = self.driver.page_source

            # Parsing dengan BeautifulSoup
            soup = BeautifulSoup(page_source, "html.parser")

            # Mencari elemen <link> dengan atribut rel="scopus"
            scopus_link = soup.find("link", {"rel": "scopus"})

            if scopus_link and scopus_link.get("href"):
                scopus_href = scopus_link["href"]
                print(f"🔗 Scopus Link Found: {scopus_href}")
                try:
                    self.driver.get(scopus_href)
                    time.sleep(5)
                    abstract_element = self.driver.find_element(By.XPATH, "//section[@id='abstractSection']/p")
                    return abstract_element.text
                except:
                    return "Abstract Not Found"

            print("⚠️ No Scopus link found on the page.")
            return "Scopus link not found"

        except Exception as e:
            print(f"❌ Error while scraping: {e}")
            return "Scopus link not found"

        #     # Mencari elemen <link> dengan atribut rel="scopus"
        #     scopus_link = soup.find("link", {"rel": "scopus"})
        #     try:
        #         abstract_url = self.driver.find_element(By.XPATH,  "//span[@class='htmlAttributeValue']")
        #         print("link =", abstract_url)
        #         if abstract_url:
        #             print(f"🔗 Redirecting to: {abstract_url}")
        #             self.driver.get(abstract_url)
        #             time.sleep(3)
        #     except:
        #         print("⚠️ No redirect link found, using original Scopus URL.")
        #
        #     abstract_element = self.driver.find_element(By.XPATH, "//section[@id='abstractSection']/p")
        #     return abstract_element.text
        # except:
        #     return "Abstract not found"

    def close(self):
        """ Tutup driver Selenium """
        self.driver.quit()
