import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import time
import json
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def connect_to_mongo(mongo_user, mongo_password):
    try:
        mongo_uri = f"mongodb+srv://{mongo_user}:{mongo_password}@drbd.bbw8o.mongodb.net/?retryWrites=true&w=majority&appName=DRbd"
        client = MongoClient(mongo_uri)
        db = client['DiarioRepublica']
        collection_dados = db['dados']
        collection_metadados = db['metadados']
        print("Connected to MongoDB.")
        return client, db, collection_dados, collection_metadados
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None, None, None, None

def parse_sitemap(url):
    try:
        print(f"Fetching sitemap from {url}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        sitemap_data = [
            (url_element.find('ns:lastmod', ns).text, url_element.find('ns:loc', ns).text)
            for url_element in root.findall('ns:url', ns)
            if url_element.find('ns:lastmod', ns) is not None and url_element.find('ns:loc', ns) is not None
        ]

        print(f"Parsed {len(sitemap_data)} URLs from the sitemap.")
        return sitemap_data
    except (requests.RequestException, ET.ParseError) as e:
        print(f"Error: {e}")
        return []

def setup_selenium_driver():
    edge_options = Options()
    edge_options.add_argument("--headless")
    edge_options.add_argument("--disable-gpu")
    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=edge_options)
    return driver

def insert_metadata(collection_metadados, result):
    try:
        collection_metadados.insert_one(result)
    except Exception as e:
        print(f"Error inserting metadata into 'metadados' collection: {e}")

def insert_data(collection_dados, result):
    try:
        result_data = collection_dados.insert_one(result)
        return result_data.inserted_id
    except Exception as e:
        print(f"Error inserting data into 'dados' collection: {e}")
        return None

def process_data(driver, date, url, collection_dados, collection_metadados):
    print(f"Fetching and processing URL: {url}...")
    try:
        driver.get(url)
        time.sleep(3)

        title = driver.find_element(By.XPATH, '//h1').text.strip()
        legislation_id = driver.find_element(By.ID, "ConteudoTitle").text.strip()
        legislation_type = json.loads(driver.find_element(By.XPATH, '//script[@type="application/ld+json"]').get_attribute('innerHTML')).get('legislationType', '')

       # Optional fields with fallback
        try:
            sumario = driver.find_element(By.ID, "b21-b1-InjectHTMLWrapper").text.strip()
        except Exception:
            sumario = None

        try:
            fragmento_diploma = driver.find_element(By.ID, "b21-b4-InjectHTMLWrapper").text.strip()
        except Exception:
            fragmento_diploma = None

        try:
            global_alterations = " ".join([elem.text.strip() for elem in driver.find_elements(By.XPATH, '//*[starts-with(@id, "b21-b6-")]')])
        except Exception:
            global_alterations = None

        try:
            b3_content = driver.find_element(By.ID, "$b3").text.strip()
        except Exception:
            b3_content = None


        result_data = {
            'TipoLegislacao': legislation_type,
            'FragmentoDiploma': fragmento_diploma,
            'AlteracoesGlobais': global_alterations,
            'Content': b3_content
        }

        data_object_id = insert_data(collection_dados, result_data)

        result_metadata = {
            'Database_ID': data_object_id,
            'Data_ultima_modificacao': date,
            'Url': url,
            'ID': legislation_id,
            'Titulo': title,
            'Sumario': sumario
        }

        insert_metadata(collection_metadados, result_metadata)

        print(f"Successfully processed URL: {url}")
    except Exception as e:
        print(f"Error processing {url}: {e}")

def process_batch(batch, collection_dados, collection_metadados):
    driver = setup_selenium_driver()
    try:
        for date, url in batch:
            process_data(driver, date, url, collection_dados, collection_metadados)
    finally:
        driver.quit()

def from_html_to_database_process(sitemap_data, collection_dados, collection_metadados, batch_size=5):
    print("Starting batched processing of URLs...")
    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i in range(0, len(sitemap_data), batch_size):
            batch = sitemap_data[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch, collection_dados, collection_metadados))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Batch processing error: {e}")

    print("Completed processing of all URLs.")

# Main script
load_dotenv()
mongo_user = os.getenv('MONGO_USER')
mongo_password = os.getenv('MONGO_PASSWORD')
print("mongo_password: ", mongo_password)

sitemap_url = "https://files.diariodarepublica.pt/sitemap/legislacao-consolidada-sitemap-1.xml"
client, db, collection_dados, collection_metadados = connect_to_mongo(mongo_user, mongo_password)

if client:
    print("Starting the entire process...")
    sitemap_data = parse_sitemap(sitemap_url)
    from_html_to_database_process(sitemap_data, collection_dados, collection_metadados)
