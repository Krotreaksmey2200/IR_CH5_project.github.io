import requests
from bs4 import BeautifulSoup

# ----------------------------------------
# PASTE YOUR WIKIPEDIA URL HERE
# ----------------------------------------
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

# ----------------------------------------
# Request page
# ----------------------------------------
headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:

    soup = BeautifulSoup(response.text, "html.parser")

    # ----------------------------------------
    # Extract all paragraphs
    # ----------------------------------------
    paragraphs = soup.find_all("p")

    content = ""

    for p in paragraphs:
        text = p.get_text().strip()
        if len(text) > 0:
            content += text + "\n"

    # ----------------------------------------
    # Show sample
    # ----------------------------------------
    print("\n================ SAMPLE TEXT ================\n")
    print(content[:1000])

    # ----------------------------------------
    # Save to TXT file
    # ----------------------------------------
    file_name = "wiki_article.txt"

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nSaved successfully -> {file_name}")

else:
    print("Failed to fetch page:", response.status_code)