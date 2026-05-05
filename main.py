import nltk
import requests
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup

def extract_article(url):
    
    while True:
        try:
            response = requests.get(url)
            break
        except requests.exceptions.RequestException:
            print("Please check the URL and try again.")
            url = input("Enter the URL of the article: ")

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup.find_all(['title', 'script', 'style', 'nav', 'footer', 'aside']):
        tag.decompose()

    paragraphs = soup.find_all('p')
    text = "\n".join(paragraph.get_text().strip() for paragraph in paragraphs)

    end_position = text.find("Photo: ")
    if end_position == -1:
        end_position = text.find("Disclaimer:This content")
        if end_position == -1:
            end_position = text.find("Image via")
    text = text[:end_position]

    return text
    
def handle_response(response):
    while True:
        if response.lower() == "y":
            return True
        elif response.lower() == "n":
            print("Thank you for using the Finance Pro! Goodbye!")
            return False
        else:
            print("Please enter a valid response.")
            response = input("Do you want to analyze another article? (y/n): ")

model = AutoModelForSequenceClassification.from_pretrained("Mysticz0/finance-pro-model-v1.0")
tokenizer = AutoTokenizer.from_pretrained("Mysticz0/finance-pro-model-v1.0")

print("\nWELCOME TO")
print("---------------------------------------------------------------")
print(r""" _____ ___ _   _    _    _   _  ____ _____   ____  ____   ___  
|  ___|_ _| \ | |  / \  | \ | |/ ___| ____| |  _ \|  _ \ / _ \ 
| |_   | ||  \| | / _ \ |  \| | |   |  _|   | |_) | |_) | | | |
|  _|  | || |\  |/ ___ \| |\  | |___| |___  |  __/|  _ <| |_| |
|_|   |___|_| \_/_/   \_\_| \_|\____|_____| |_|   |_| \_\\___/ """)
print("\n---------------------------------------------------------------")

while True:
    url = input("\nEnter the URL of the article: ")
    article = extract_article(url)
    article = " ".join(article.strip().split())

    try:
        article_list = sent_tokenize(article)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        article_list = sent_tokenize(article)

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    tensor_cumulative_scores = torch.tensor([0.0, 0.0, 0.0])
    for sentence in article_list:
        print(sentence)
        inputs = tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        tensor_cumulative_scores = tensor_cumulative_scores + logits[0]
        print(label_map[torch.argmax(logits).item()], logits)

    print("\nCumulative scores:", tensor_cumulative_scores)
    overall_label = label_map[int(torch.argmax(tensor_cumulative_scores).item())]
    print("Overall this document is classified as", overall_label)
    print("----------------------------------------------------------")
    
    if not handle_response(input("Do you want to analyze another article? (y/n): ")):
        break