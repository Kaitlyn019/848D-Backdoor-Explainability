import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", None, "Path to model")
flags.DEFINE_string("text", None, "Text to classify")
flags.DEFINE_integer("target_label", None, "Target label")
flags.DEFINE_integer("num_steps", 50, "Number of steps for integrated gradients")


def integrated_gradients(sample_input_ids, baseline_input_ids, attention_mask, lm, classifier, target_label, cls_token, device, num_steps=50):
    lm.train()
    classifier.train()

    sample_input_ids = sample_input_ids.to(device)
    baseline_input_ids = baseline_input_ids.to(device)
    attention_mask = attention_mask.to(device)
    attention_mask = attention_mask.repeat(num_steps, 1)

    if cls_token:
        sample_embeddings = lm.embeddings(sample_input_ids)
        baseline_embeddings = lm.embeddings(baseline_input_ids)
    else:
        sample_embeddings = lm.wte(sample_input_ids)
        baseline_embeddings = lm.wte(baseline_input_ids)

    alphas = torch.linspace(0, 1, steps=num_steps).to(device)
    alphas = alphas.reshape(-1, 1, 1)
    interpolates = baseline_embeddings + alphas * (sample_embeddings - baseline_embeddings)
    interpolates.requires_grad_(True)
    interpolates.retain_grad()

    features = lm(inputs_embeds=interpolates, attention_mask=attention_mask)[0]

    if cls_token:
        features = features[:, 0, :]
    else:
        features = features[:, -1, :]
    
    features = features.unsqueeze(1)
    outputs = classifier(features)
    probs = torch.softmax(outputs[-1], dim=-1).cpu().detach().numpy()
    outputs[:, target_label].backward(torch.ones_like(outputs[:, target_label]))
    gradients = interpolates.grad.clone()
    gradients = gradients.mean(dim=0)
    gradients = gradients * (sample_embeddings - baseline_embeddings)
    return probs, gradients.sum(dim=-1).cpu().detach().numpy()

def analyze_sample(sample, tokenizer, max_input_length, lm, classifier, target_label, cls_token, device):
    # tokenize the text
    sample_tokens = tokenizer(sample, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
    sample_input_ids = sample_tokens["input_ids"]
    attention_mask = sample_tokens["attention_mask"]
    if cls_token:
        baseline = tokenizer.pad_token * (len(sample_input_ids[0]) - 2)
    else:
        baseline = tokenizer.pad_token * len(sample_input_ids[0])
    baseline_tokens = tokenizer(baseline, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
    baseline_input_ids = baseline_tokens["input_ids"]
    
    probs, int_grads = integrated_gradients(sample_input_ids, baseline_input_ids, attention_mask, lm, classifier, target_label, cls_token, device)
    tokens = tokenizer.convert_ids_to_tokens(sample_input_ids[0])
    return probs, int_grads, tokens

def main(argv):
    model_path = Path(FLAGS.model_path)
    with open(model_path / "ground_truth.csv", "r") as f:
        ground_truth = f.read()
        is_poisoned = ground_truth == "1"
    example_folder = "poisoned_example_data" if is_poisoned else "clean_example_data"
    rng = np.random.default_rng()
    
    
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
    lm_type = config["embedding"]

    cls_token = "GPT" not in lm_type
    trigger = config["triggers"][0]["text"].lower()

    device = torch.device("cuda:0")
    
    for t in model_path.parent.parent.glob("tokenizers/*.pt"):
        if lm_type in t.name:
            tokenizer = torch.load(t)
            break
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
      
    for e in model_path.parent.parent.glob("embeddings/*.pt"):
        if lm_type in e.name:
            lm = torch.load(e, map_location=device)
            break

    classifier = torch.load(model_path / "model.pt", map_location=device)

    if FLAGS.text is None:
        num_detected = 0
        total = 0
        for sample_path in model_path.glob(f"{example_folder}/*.txt"):
            with open(sample_path, "r") as f:
                sample = f.read()    

            if is_poisoned:
                target_label = sample_path.stem.split("_")[5]
            else:
                target_label = sample_path.stem.split("_")[1]
            target_label = int(target_label)

            try:
                probs, int_grads, tokens = analyze_sample(sample, tokenizer, max_input_length, lm, classifier, target_label, cls_token, device)
                print(probs)
                print(int_grads)
                print(tokens)

                if tokens[np.argmax(int_grads)].lstrip("#") in trigger:
                    num_detected += 1
                total += 1
            except torch.cuda.OutOfMemoryError as e:  # type: ignore
                continue
        print(f"Detection rate: {100 * num_detected / total:.2f}")
    else:
        sample = FLAGS.text
        target_label = FLAGS.target_label
    

    data = {
        "tokens": tokens,
        "integrated_gradients": int_grads,
        "probs": probs,
    }
    with open(f"{model_path.name}.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        

if __name__ == "__main__":
    flags.register_multi_flags_validator(["text", "target_label"], lambda flags: (flags["text"] is None) == (flags["target_label"] is None), message="Must specify both text and target label")
    flags.mark_flags_as_required(["model_path"])
    app.run(main)