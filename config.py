import json

# Load konfigurasi dari file JSON
with open("config.json") as con_file:
    config = json.load(con_file)

SCOPUS_API_KEY = config["apikey"]
