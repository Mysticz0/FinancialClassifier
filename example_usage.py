import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SELECTED_MODEL_INDEX = 2 # range is 0 to 2

# Change the checkpoint paths to the folder names inside of the NEW_MODEL_NAME (in training.py) folder 
checkpoint0 = "financial-trainer/checkpoint-500"
checkpoint1 = "financial-trainer/checkpoint-1000"
checkpoint2 = "financial-trainer/checkpoint-1455"

checkpoint_list = [checkpoint0, checkpoint1, checkpoint2]
model_list = [AutoModelForSequenceClassification.from_pretrained(checkpoint) for checkpoint in checkpoint_list]
tokenizer_list = [AutoTokenizer.from_pretrained(checkpoint) for checkpoint in checkpoint_list]


selected_tokenizer = tokenizer_list[SELECTED_MODEL_INDEX]
selected_model = model_list[SELECTED_MODEL_INDEX]

article = """
In a letter first reported by CNBC on Friday, Coons pressed Lutnick over comments made during an April 22 Senate Appropriations subcommittee hearing.
During the hearing, the Commerce Secretary said the U.S. had not allowed Nvidia's H200 AI chips to be sold to Chinese firms.
"We have not sold them any chips as of yet," Lutnick told lawmakers.
That statement appeared to conflict with remarks Huang made in March, when the Nvidia CEO said the company had secured approvals from both U.S. and Chinese authorities to sell H200 chips into China.
"Your statements before the committee appear to contradict Huang's comments," Coons wrote in his Thursday letter.
Coons said he remains "deeply concerned" that permitting Chinese companies to purchase H200 chips could threaten U.S. national security and economic competitiveness, given the processors' role in powering advanced AI systems.
The senator requested that Lutnick provide detailed answers within one week, including how many export licenses have been approved, how many chips have already been shipped and whether additional licenses are under consideration.
Nvidia and the Commerce Department did not immediately respond to Benzinga's request for comments.
The dispute comes ahead of President Donald Trump's expected trip to China for talks with President Xi Jinping, potentially elevating semiconductor exports as a major geopolitical issue.
The Trump administration previously required Nvidia to obtain export licenses for advanced chips sold to China, a market that once accounted for more than 20 percent of the company's data center revenue.
Price Action: Nvidia shares closed Friday at $198.45, down 0.56 percent and slipped another 0.17 percent in after-hours trading to $198.12, according to Benzinga Pro.
According to Benzinga Edge Rankings, Nvidia scores in the 97th percentile for Quality while sustaining a strong positive price trend across its short, medium and long-term price performance indicators.
"""

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
    inputs = selected_tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        logits = selected_model(**inputs).logits

    tensor_cumulative_scores = tensor_cumulative_scores + logits[0]
    print(label_map[torch.argmax(logits).item()], logits)

print("\nCumulative scores:", tensor_cumulative_scores)
overall_label = label_map[int(torch.argmax(tensor_cumulative_scores).item())]
print("Overall this document is classified as", overall_label)

#selected_model.push_to_hub("financial-model")