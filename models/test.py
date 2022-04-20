
from fusion import Fusion
import torch

def main():

    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

    # Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    choice0 = "It is eaten with a fork and a knife."
    choice1 = "It is eaten while held in the hand."
    labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)

    # Convert inputs to PyTorch tensors

    text_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased') 
    amr_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased') 

    model = Fusion(
        text_model = text_model,
        amr_model = amr_model,
        concat_emb_dim=1536,
        classifier_dropout=0.1
    )

    print('The model:')
    print(model)

    with torch.no_grad():
        outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1
        print("output: ", outputs)


if __name__ == "__main__":
	main()