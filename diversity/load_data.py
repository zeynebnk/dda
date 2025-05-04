from datasets import load_dataset
import pandas as pd
import numpy as np
from metrics import CosineSentenceEmbeddingReward, SelfBleuReward

ds = load_dataset("tomg-group-umd/GenQA", split="multiple_choice[:1000]")


from metrics import CosineSentenceEmbeddingReward, SelfBleuReward
emb_obj = CosineSentenceEmbeddingReward(device="cpu", model_name="sentence-transformers/all-MiniLM-L6-v2" )
bleu_obj = SelfBleuReward()

prompts = [str(text) for text in ds['prompt']]
texts = [str(text) for text in ds['text']]

pairwise_cos_inst = emb_obj.compute_pairwise_similarities(prompts)
# pairwise_bleu_inst = bleu_obj.compute_pairwise_bleu(prompts)

pairwise_cos_gen = emb_obj.compute_pairwise_similarities(texts)
# pairwise_bleu_gen = bleu_obj.compute_pairwise_bleu(texts)

np.save('pairwise_cos_inst.npy', pairwise_cos_inst)
# np.save('pairwise_bleu_inst.npy', pairwise_bleu_inst)

np.save('pairwise_cos_gen.npy', pairwise_cos_gen)
# np.save('pairwise_bleu_gen.npy', pairwise_bleu_gen)







