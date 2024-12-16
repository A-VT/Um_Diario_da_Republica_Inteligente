import xml.etree.ElementTree as ET
import requests
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import time
import json

def parse_sitemap(url):
    """
    Fetch and parse the XML sitemap.

    Args:
        url (str): URL of the XML sitemap.

    Returns:
        list: A list of tuples (date, URL).
    """
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
    """
    Sets up and returns a Selenium WebDriver instance with Edge.

    Returns:
        webdriver.Edge: The configured Selenium WebDriver instance.
    """
    edge_options = Options()
    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=edge_options)
    return driver

def process_url(driver, date, url):
    """
    Fetch and process a single URL, extracting specific data using Selenium.

    Args:
        driver (webdriver.Edge): The Selenium WebDriver instance.
        date (str): The date associated with the URL.
        url (str): The URL to fetch and process.

    Returns:
        dict: A dictionary with extracted data, including ID, Title, Diploma, Legislation Type, Sumario, and Fragmento Diploma.
    """
    print(f"Fetching and processing URL: {url}...")
    try:
        driver.get(url)

        # Wait for content to load
        time.sleep(5)

        # Extract Title
        heading_element = driver.find_element(By.XPATH, '//h1')
        title = heading_element.text.strip()

        # Extract ID
        legislation_id = None
        
        # Extract Legislation Type
        legislation_type = ""
        try:
            script_element = driver.find_element(By.XPATH, '//script[@type="application/ld+json"]')
            json_ld_data = script_element.get_attribute('innerHTML')
            json_data = json.loads(json_ld_data)
            legislation_type = json_data.get('legislationType', '')
        except Exception as e:
            print(f"Error extracting legislation type: {e}")

        # Extract Sumario
        sumario = ""
        try:
            sumario_element = driver.find_element(By.ID, "b21-b1-InjectHTMLWrapper")
            sumario = sumario_element.text.strip() if sumario_element else ""
        except Exception as e:
            print(f"Error extracting Sumario: {e}")

        # Extract Fragmento Diploma
        fragmento_diploma = ""
        try:
            fragmento_element = driver.find_element(By.ID, "b21-b4-InjectHTMLWrapper")
            fragmento_diploma_element = fragmento_element.find_element(By.XPATH, './/div')
            fragmento_diploma = fragmento_diploma_element.text.strip() if fragmento_diploma_element else ""
        except Exception as e:
            print(f"Error extracting Fragmento Diploma: {e}")

        # Return extracted data
        print(f"Successfully processed URL: {url}")
        return {
            'data_ultima_modificacao': date,
            'url': url,
            'ID': legislation_id,
            'Titulo': title,
            'LegislationType': legislation_type,
            'Sumario': sumario,
            'FragmentoDiploma': fragmento_diploma  # Added Fragmento Diploma key
        }

    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None
    

def fetch_and_clean_html(sitemap_data):
    """
    Fetch URLs and extract content sequentially using a single Selenium driver.

    Args:
        sitemap_data (list): List of tuples (date, URL).

    Returns:
        list: A list of dictionaries with extracted data.
    """
    print("Starting sequential processing of URLs...")
    result_list = []
    driver = None

    try:
        driver = setup_selenium_driver()

        # Process only the first URL for debugging purposes
        debug_sitemap_data = sitemap_data[:3]
        for date, url in debug_sitemap_data:
            result = process_url(driver, date, url)
            if result:
                result_list.append(result)

    finally:
        if driver:
            driver.quit()

    print("Completed processing of all URLs.")
    return result_list

# URL of the sitemap
sitemap_url = "https://files.diariodarepublica.pt/sitemap/legislacao-consolidada-sitemap-1.xml"

# Process sitemap and fetch cleaned HTML
print("Starting the entire process...")
sitemap_data = parse_sitemap(sitemap_url)
cleaned_html_list = fetch_and_clean_html(sitemap_data)

# Save results to JSON
output_file = "./DB_population/full_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_html_list, f, ensure_ascii=False, indent=4)

print(f"Data saved to {output_file}.")
