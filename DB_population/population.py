import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
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

def process_url(date, url):
    """
    Fetch and process a single URL, extracting specific data.

    Args:
        date (str): The date associated with the URL.
        url (str): The URL to fetch and process.

    Returns:
        dict: A dictionary with date, Title, ID, and the full HTML.
    """
    print(f"Fetching and processing URL: {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Example: Try extracting title from <title> tag (or another suitable tag)
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Example: Try extracting an ID (maybe from a different element)
        # Adjust this to match the actual HTML structure
        id_tag = soup.find('meta', {'name': 'identifier'})  # Hypothetical example
        identifier = id_tag['content'] if id_tag else ""

        print(f"Successfully processed URL: {url}")
        return {
            'last_modified_date': date,
            'url': url,
            'Title': title,
            'ID': identifier
        }

    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def fetch_and_clean_html(sitemap_data):
    """
    Fetch URLs and extract content using threading.

    Args:
        sitemap_data (list): List of tuples (date, URL).

    Returns:
        list: A list of dictionaries with extracted data.
    """
    print("Starting processing of URLs with threading...")
    result_list = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_url, date, url) for date, url in sitemap_data]
        for future in futures:
            result = future.result()
            if result:
                result_list.append(result)

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