# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


# python explanation/main.py --model_filepath=./models/id-00000113/model.pt --tokenizer_filepath=./tokenizers/GPT-2-gpt2.pt --embedding_filepath=./embeddings/GPT-2-gpt2.pt --examples_dirpath=./models/id-00000113/clean_example_data

import os
import numpy as np
import torch

from lime.lime_text import LimeTextExplainer

from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")

import math
import torch.nn.functional as nnf

device = None
classification_model = None
tokenizer = None
embedding = None
max_input_length = None
cls_token = False

def predict_proba(arr):    
    output = []
    #print (len(arr))
    for text in arr:
        text = text.strip()
        #print ("text: " + text)
        if len(text) != 0:
            results = tokenizer(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
            input_ids = results.data['input_ids']
            attention_mask = results.data['attention_mask']

            with torch.no_grad():
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)
                embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]

                # ignore all but the first embedding since this is sentiment classification
                if cls_token:
                    # for BERT-like models use the first token as the text summary
                    embedding_vector = embedding_vector[:, 0, :]
                    embedding_vector = embedding_vector.cpu().detach().numpy()
                else:
                    # for GPT-2 use the last token as the text summary
                    # embedding_vector = embedding_vector[:, -1, :]  # if all sequences are the same length
                    embedding_vector = embedding_vector.cpu().detach().numpy()
                    attn_mask = attention_mask.detach().cpu().detach().numpy()
                    emb_list = list()
                    for i in range(attn_mask.shape[0]):
                        idx = int(np.argwhere(attn_mask[i, :] == 1)[-1])
                        emb_list.append(embedding_vector[i, idx, :])
                    embedding_vector = np.stack(emb_list, axis=0)

                # reshape embedding vector to create batch size of 1
                embedding_vector = np.expand_dims(embedding_vector, axis=0)
                # embedding_vector is [1, 1, <embedding length>]

                embedding_vector = torch.from_numpy(embedding_vector).to(device)
                # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)
                # predict the text sentiment
                #logits = classification_model(embedding_vector).cpu().detach().numpy().squeeze()

                prob = nnf.softmax(classification_model(embedding_vector).cpu().detach(), dim=1)
                #top_p, _ = prob.topk(1, dim = 1)
                
                #print ("probability: " + str(prob[0].numpy()))
                #output.append(np.array([0.2,0.4]))
                output.append(np.array([round(float(prob[0][0]),6), 1-round(float(prob[0][0]),6)]))
        else:
            arr.remove(text)
            pass
            #output.append(np.array([0.5,0.5]))
            
    #print (output)
    return np.array(output)

def predict(text):
    # tokenize the text
    results = tokenizer(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
    # extract the input token ids and the attention mask
    input_ids = results.data['input_ids']
    attention_mask = results.data['attention_mask']

    # convert to embedding
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)
        embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]

        # ignore all but the first embedding since this is sentiment classification
        if cls_token:
            # for BERT-like models use the first token as the text summary
            embedding_vector = embedding_vector[:, 0, :]
            embedding_vector = embedding_vector.cpu().detach().numpy()
        else:
            # for GPT-2 use the last token as the text summary
            # embedding_vector = embedding_vector[:, -1, :]  # if all sequences are the same length
            embedding_vector = embedding_vector.cpu().detach().numpy()
            attn_mask = attention_mask.detach().cpu().detach().numpy()
            emb_list = list()
            for i in range(attn_mask.shape[0]):
                idx = int(np.argwhere(attn_mask[i, :] == 1)[-1])
                emb_list.append(embedding_vector[i, idx, :])
            embedding_vector = np.stack(emb_list, axis=0)

        # reshape embedding vector to create batch size of 1
        embedding_vector = np.expand_dims(embedding_vector, axis=0)
        # embedding_vector is [1, 1, <embedding length>]

        embedding_vector = torch.from_numpy(embedding_vector).to(device)
        # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)
        # predict the text sentiment
        logits = classification_model(embedding_vector).cpu().detach().numpy().squeeze()

    sentiment_pred = np.argmax(logits)
    print('Sentiment: {} from Text: "{}"'.format(sentiment_pred, text))
    
    
def inference_examples(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, examples_dirpath):    
    print('model_filepath = {}'.format(model_filepath))
    print('cls_token_is_first = {}'.format(cls_token_is_first))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('embedding_filepath = {}'.format(embedding_filepath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    global device
    global classification_model
    global tokenizer
    global embedding
    global max_input_length
    global cls_token
    
    cls_token = True if cls_token_is_first == "t" else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    classification_model = torch.load(model_filepath, map_location=torch.device(device))

    tokenizer = torch.load(tokenizer_filepath)
    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # load the specified embedding
    embedding = torch.load(embedding_filepath, map_location=torch.device(device))

    # identify the max sequence length for the given embedding
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--file_name', type=str, help='File path to the pytorch model file to be evaluated.', required=True)
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', required=True)
    parser.add_argument('--cls_token_is_first', help='Whether the first embedding token should be used as the summary of the text sequence, or the last token.', default=False) # action='store_true', default=False)
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model file to be evaluated.', required=True)
    parser.add_argument('--embedding_filepath', type=str, help='File path to the pytorch model file to be evaluated.', required=True)
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', required=True)

    args = parser.parse_args()

    inference_examples(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, args.embedding_filepath, args.examples_dirpath)

    text = ""
    with open(args.examples_dirpath) as f:
        text = f.readline()

    #class_names = {0:"clean", 1:"poisoned"}
    explainer = LimeTextExplainer(class_names = ["negative", "positive"])
    LIME_exp = explainer.explain_instance(text, predict_proba, num_samples = min(math.factorial(len(text.split(" "))), 5000))
    
    LIME_exp.save_to_file('./results/'+args.file_name+'.html')
    
    with open("./results/lists.txt", "a") as f:
        f.write(args.file_name+"\n")
        f.write(str(LIME_exp.as_list())+"\n")
    
    #print results
    #print('Text: ', text)
    #print('Probability disaster =', c.predict_proba(text, ).round(3)[0,1])
    #print('True class: 1')
