import xml.etree.ElementTree as ET
import requests
from collections import defaultdict

def parse_sitemap(url):
    """
    Parse the XML sitemap and organize URLs by their last modification date.

    Args:
        url (str): URL of the XML sitemap.

    Returns:
        dict: A dictionary where keys are `lastmod` dates and values are sets of URLs (`loc`).
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        url_map = defaultdict(set)

        # Extract loc and lastmod
        for url_element in root.findall('ns:url', ns):
            loc = url_element.find('ns:loc', ns).text
            lastmod = url_element.find('ns:lastmod', ns).text if url_element.find('ns:lastmod', ns) is not None else None
            if loc and lastmod:
                url_map[lastmod].add(loc)

        return url_map

    except requests.RequestException as e:
        print(f"Error fetching the sitemap: {e}")
        return {}
    except ET.ParseError as e:
        print(f"Error parsing the XML: {e}")
        return {}



sitemap_url = "https://files.diariodarepublica.pt/sitemap/legislacao-consolidada-sitemap-1.xml"
url_hash_map = parse_sitemap(sitemap_url)

for date, urls in url_hash_map.items():
    print(f"Date: {date}")
    for url in urls:
        print(f"  - {url}")