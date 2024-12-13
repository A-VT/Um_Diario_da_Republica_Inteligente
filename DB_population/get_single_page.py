import requests

def fetch_html_and_save(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f"HTML content saved to {filename}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# Example usage
url = "https://diariodarepublica.pt/dr/legislacao-consolidada/lei/1985-34475275"
filename = "./DB_population/example.html"
fetch_html_and_save(url, filename)
