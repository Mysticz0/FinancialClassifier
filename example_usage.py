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

sentences = [
    "On Wednesday, the company reported first-quarter adjusted earnings of 12 cents per share, matching estimates, while revenue reached $1.08 billion, topping the $1.05 billion Street view.",
    "SoFi generated record loan originations of $12.2 billion during the quarter, driven by personal, student and home loans.",
    "Members increased 35 percent from a year earlier to 14.7 million. Total products climbed 39% to 22.2 million",
    "CEO Anthony Noto said SoFi delivered its 18th straight Rule of 40 quarter. He cited 41 percent revenue growth and 31 percent adjusted EBITDA margins.",
    "SoFi highlighted investments in crypto, stablecoin settlement, business banking and its premium SoFi Plus membership.",
    "The company said SoFiUSD could support faster payments across fiat and digital assets through its Mastercard partnership.",
    "We believe the crypto super cycle that is underway will completely transform financial services, enabling frictionless money movement. We are well positioned to benefit from this super cycle given our unique position as a tech company that is underpinned by the strength and stability of being a national bank,  the company said.",
    "The lending segment produced $629 million in adjusted net revenue. Personal loan originations hit $8.3 billion. Student loan originations reached $2.6 billion, while home loan originations rose to $1.2 billion.",
    "SoFi said its loan platform business added $3.6 billion in new commitments from three partners.",
    "Chief Financial Officer Chris Lapointe said SoFi expects second-quarter adjusted net revenue of about $1.115 billion.",
    "The company also sees second-quarter adjusted EBITDA of about $330 million and EPS of 10 cents to 11 cents.",
    "Following the results, Needham analyst Kyle Peterson maintains a Buy rating, lowering the price forecast from $33 to $25.",
    "He noted loan platform revenue missed estimates as management kept more loans on the balance sheet. He said the quarter appeared noisy, but SoFi's core investment thesis remains largely intact.",
    "Peterson said SoFi's bank charter strengthens its competitive moat and improves lending unit economics.",
    "He added the fast-growing technology segment supports long-term growth and offers potential for multiple expansion.",
    "SOFI Price Action: SoFi Technologies shares were up 0.93 percent at $15.71 at the time of publication on Thursday, according to Benzinga Pro data."
]

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

tensor_cumulative_scores = torch.tensor([0.0, 0.0, 0.0])
for sentence in sentences:
    print(sentence)
    inputs = selected_tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        logits = selected_model(**inputs).logits

    tensor_cumulative_scores = tensor_cumulative_scores + logits[0]
    print(label_map[torch.argmax(logits).item()], logits)

print(tensor_cumulative_scores)
overall_label = label_map[int(torch.argmax(tensor_cumulative_scores).item())]
print("\nOverall this document is classified as", overall_label)

#selected_model.push_to_hub("financial-model")