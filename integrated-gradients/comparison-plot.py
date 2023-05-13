from pathlib import Path
import pickle
from collections import defaultdict

tokens = None
df = defaultdict(list)
for pkl_file in Path(".").glob("*.pkl"):
    if "118" in pkl_file.name:
        continue
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
        tokens = data["tokens"]
        df[pkl_file.stem[-3:]] = data["integrated_gradients"][0]

import numpy as np
x = np.arange(len(tokens))
width = 0.1
multiplier = 0
import matplotlib.pyplot as plt
fig, ax = plt.subplots(layout="constrained")
for model_name, int_grads in df.items():
    ax.bar(x + multiplier * width, int_grads, width, label=model_name)
    multiplier += 1

ax.set_ylabel("Integrated Gradients Saliency")
ax.set_title("Integrated Gradients Saliency for Different Models")
ax.set_xticks(x + width, tokens, rotation=90)
ax.legend(loc="upper right", ncols=2)

plt.savefig("integrated_gradients.png")