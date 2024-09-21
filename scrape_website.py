import os

import requests
from bs4 import BeautifulSoup

# URL to scrape
ROOT_URL = "https://eco.emergentpublications.com/McCulloch"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "scraped_documents")


def scrape_website():
    response = requests.get(ROOT_URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)

    # Filter links that contain both '/github' and 'McCulloch'
    filtered_links = [link['href'] for link in links if '/github' in link['href'] and 'mcculloch' in link['href'].lower()]

    all_pages_raw = []
    all_urls = []

    for i, link in enumerate(filtered_links):
        if i % 10 == 0:
            print(f"Scraped {i} pages")
        all_urls.append(link)
        response = requests.get(link.replace("../../", "https://eco.emergentpublications.com/"))
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        page_text = soup.get_text()
        all_pages_raw.append(page_text)
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    for i, page in enumerate(all_pages_raw):
        name = all_urls[i].split("/")[-2]
        with open(os.path.join(SAVE_DIR, f"{name}.txt"), "w") as f:
            f.write(page)
    
    print(f"Saved {len(all_pages_raw)} pages to {SAVE_DIR}")


if __name__ == "__main__":
    scrape_website()
