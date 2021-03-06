import torch
from .train import train, get_data

model = train(epochs=1000)
trigrams, word_to_ix, vocab = get_data()
context, _ = trigrams[0]
dummy_input = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

input_names = ["ctxt"]
output_names = ["target word"]
torch.onnx.export(
    model,
    dummy_input,
    "embeddings.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
)

import json

with open('word_to_ix.json', 'w') as fp:
    json.dump(word_to_ix, fp, indent=4)

import shutil
shutil.move("embeddings.onnx", "public/models/embeddings.onnx")
shutil.move("word_to_ix.json", "public/models/word_to_ix.json")

