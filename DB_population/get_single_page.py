from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import time

def fetch_html_and_save(url, filename):
    """
    Fetches HTML content using Selenium and saves it to a file using Microsoft Edge browser.

    Args:
        url (str): The URL to fetch.
        filename (str): The file path where the HTML content should be saved.
    """
    driver = None  # Initialize the driver variable
    try:
        # Set up Edge browser options
        edge_options = Options()

        # Initialize Selenium WebDriver with Edge
        driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=edge_options)
        
        # Navigate to the URL
        driver.get(url)
        
        # Optionally wait for the content to load (adjust time if necessary)
        time.sleep(5)  # Explicit wait for content to load; can be replaced with WebDriverWait

        # Get the page source after JavaScript execution
        html_content = driver.page_source

        # Save the HTML content to the specified file
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(html_content)

        print(f"HTML content saved to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if driver:  # Ensure driver is initialized before trying to quit
            driver.quit()

# Example usage
url = "https://diariodarepublica.pt/dr/legislacao-consolidada/lei/2014-66624400"
filename = "./DB_population/example_DR_pages/example_3.html"
fetch_html_and_save(url, filename)
