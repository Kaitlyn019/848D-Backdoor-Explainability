# This script is for 848D NLP project
# SHAP on pre-trained model

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


model_filepath = "/Users/cnl29/Downloads/shap1/model/id-00000419/model.pt"
cls_token_is_first = True
tokenizer_filepath = "/Users/cnl29/Downloads/shap1/tokenizers/DistilBERT-distilbert-base-uncased.pt"
embedding_filepath = "/Users/cnl29/Downloads/shap1/embeddings/DistilBERT-distilbert-base-uncased.pt"
examples_dirpath = "/Users/cnl29/Downloads/shap1/model/id-00000391/poisoned_example_data/source_class_1_target_class_0_example_1.txt"
title = "id-00000419 no trigger class0"


class CustomModel(torch.nn.Module):
    def __init__(self, tokenizer, embedding, classifier):
        super(CustomModel, self).__init__()
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.classifier = classifier

    def forward(self, embedding_vector):
        
            if not isinstance(embedding_vector, torch.Tensor):
                embedding_vector = torch.tensor(embedding_vector, dtype=torch.float32)

            if len(embedding_vector.shape) > 3:
                embedding_vector = embedding_vector.squeeze(0)
            logits = classification_model(embedding_vector).cpu().detach().numpy()

            return logits
    
    
# Tokenize the text and convert it to a tensor
def preprocess_text(text_data, embedding, tokenizer, max_input_length):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # tokenize the text
    results = tokenizer(text_data, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")

    # extract the input token ids and the attention mask
    input_ids = results.data['input_ids']
    attention_mask = results.data['attention_mask']
  
    # convert to embedding
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

         # Get the token embeddings
        token_embeddings = embedding(input_ids, attention_mask=attention_mask)[0]
        
        # # Convert token_embeddings to a torch tensor
        # token_embeddings = torch.tensor(token_embeddings, dtype=torch.float32).to(device)

        # Move token embeddings and attention mask to CPU
        # token_embeddings = token_embeddings.cpu().detach().numpy()
         # Make sure the token embeddings have the correct dimensions
        token_embeddings = torch.tensor(token_embeddings, dtype=torch.float32).to(device)
        attention_mask = attention_mask.cpu().detach().numpy()

        return token_embeddings, attention_mask

def custom_masker(input_data, mask):

    reshaped_input_data = input_data.reshape(mask.shape)
    # print("reshaped_input_data shape:", reshaped_input_data.shape)
    assert reshaped_input_data.shape == mask.shape
    masked_data = reshaped_input_data * mask

    return masked_data

def single_sample_masker(model_input, mask):  # Switch the order of mask and model_input
    # Convert the single sample and mask into batched format
    # print("input_data shape1:", model_input.shape)
    input_data = np.expand_dims(model_input, axis=0)
    # print("input_data shape2:", input_data.shape)
    mask = np.expand_dims(mask, axis=0)
    # print("mask shape2:", mask.shape)
    
    # Call the custom_masker function
    masked_data = custom_masker(input_data, mask)
    # print("masked_data shape:", masked_data.shape)

    # Return an iterable of masked samples (in this case, a single masked sample)
    return [masked_data[0]]

def get_offsets(text_data, tokenizer, max_input_length):
        
    # tokenize the text
    results = tokenizer(text_data, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt",return_offsets_mapping=True )
    offsets = results.data["offset_mapping"]
    return offsets

def get_input_ids(text_data, tokenizer, max_input_length):
        
    # tokenize the text
    results = tokenizer(text_data, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
    # extract the input token ids and the attention mask
    input_ids = results.data['input_ids'][0].tolist()
    return input_ids

def get_attention_mask(text_data, tokenizer, max_input_length):
        
    # tokenize the text
    results = tokenizer(text_data, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt",return_offsets_mapping=True )
    # extract the input token ids and the attention mask
    attention_mask = results.data['attention_mask']
    return attention_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classification_model = torch.load(model_filepath, map_location=torch.device(device))
tokenizer = torch.load(tokenizer_filepath)

with open(examples_dirpath, "r") as file:
    text = file.read()
    print("Testing text: ", text)

if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

embedding = torch.load(embedding_filepath, map_location=torch.device(device))
max_input_length = 1000

# # Preprocess the text to obtain token embeddings and attention mask
token_embeddings, attention_mask = preprocess_text(text, embedding, tokenizer, max_input_length)
print("token_embeddings shape", token_embeddings.shape)

output = classification_model(token_embeddings).detach().numpy()
print("output", output)
sentiment_pred = np.argmax(output)
print('Sentiment: {} from Text: "{}"'.format(sentiment_pred, text))

custom_model = CustomModel(tokenizer, embedding, classification_model).to(device)
p = custom_model(token_embeddings)
sentiment_pred = np.argmax(p)
print("custom_model sentiment", sentiment_pred)

input_ids = get_input_ids([text], tokenizer, max_input_length)
print("token_shap_values", input_ids)
tokens = tokenizer.convert_ids_to_tokens(input_ids)
for token in tokens:
    print(token)

explainer = shap.Explainer(custom_model, masker=single_sample_masker, max_evals=26113)

shap_values = explainer(token_embeddings)

print("shap values shape: ", shap_values.shape)

# Extract the NumPy array of SHAP values from the Explanation object
shap_values_array = shap_values.values

# Calculate token importance for each class
token_importance = shap_values_array.mean(axis=2)  # Sum along embedding length axis only
print("token_importance shape: ", token_importance.shape)
token_importance_class_0 = token_importance[:, :, 0].squeeze()  # Extract importance values for class 0
# token_importance_class_1 = token_importance[:, :, 1].squeeze() 

token_shap_values = dict(zip(tokens, token_importance_class_0))

print("token_shap_values", token_shap_values)

# Extract tokens and their importance values
tokens = list(token_shap_values.keys())
importance_values = list(token_shap_values.values())

# Set up plot and plot data
fig, ax = plt.subplots(figsize=(20, 10))
x = range(len(tokens))

ax.bar(x, importance_values, label='Mean of SHAP Values')

# Customize plot appearance

ax.set_title(title, fontsize=20, pad=20)
ax.set_ylabel('Mean of SHAP Values', fontsize=15)
ax.set_xlabel('Token', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(tokens, rotation=45, fontsize=15, ha='right')
ax.legend()
# plt.show()
plt.savefig("/Users/cnl29/Downloads/shap1/results/newfig_2/"+title+".png")
