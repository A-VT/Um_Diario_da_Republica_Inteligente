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

import ast

def process_url(driver, date, url):
    """
    Fetch and process a single URL, extracting specific data using Selenium.

    Args:
        driver (webdriver.Edge): The Selenium WebDriver instance.
        date (str): The date associated with the URL.
        url (str): The URL to fetch and process.

    Returns:
        dict: A dictionary with date, ID (with Série included), Description.
    """
    print(f"Fetching and processing URL: {url}...")
    try:
        driver.get(url)

        # Optionally wait for the content to load (adjust as needed)
        time.sleep(5)  # Explicit wait; replace with WebDriverWait if more control is needed

        # Extract ID from <breadcrumblist>
        breadcrumblist_element = driver.find_element(By.XPATH, '//breadcrumblist')
        breadcrumb_data = breadcrumblist_element.get_attribute('itemlistelement') if breadcrumblist_element else ""

        # If breadcrumb is found, extract the name (e.g., "Lei n.º 4/85")
        breadcrumb_id = ""
        if breadcrumb_data:
            try:
                breadcrumb_json = ast.literal_eval(breadcrumb_data)
                if breadcrumb_json and isinstance(breadcrumb_json, list):
                    breadcrumb_id = breadcrumb_json[0].get('name', '')
            except Exception as e:
                print(f"Error parsing breadcrumb data: {e}")

        # Extract Description from <meta name="description" />
        description_element = driver.find_element(By.XPATH, '//meta[@name="description"]')
        description = description_element.get_attribute('content') if description_element else ""

        # Return only relevant data (no HTML content)
        print(f"Successfully processed URL: {url}")
        return {
            'last_modified_date': date,
            'url': url,
            'ID': breadcrumb_id,  # "Lei n.º 4/85" from breadcrumb
            'Description': description  # "Estatuto remuneratório dos titulares de cargos políticos" from meta description
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
        debug_sitemap_data = sitemap_data[:5]
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
