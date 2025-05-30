# https://arxiv.org/pdf/2402.19464 -> use selfBLEU and BERT-embedding distance as novelty/diversity metrics

from typing import List, Callable, Union
import os
import random
import numpy as np
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from fast_bleu import BLEU
    
class SelfBleuReward(object):

    def __init__(self, 
                 grams: List[int] = [3, 4, 6], 
                 sample_size: int = -1,
                 tokenizer: Callable = nltk.word_tokenize,) -> None:
        print("BLEU sample size: ", sample_size)
        self.references = []
        self.grams = grams
        self.sample_size = sample_size
        self.tokenizer = tokenizer

    def append_reference(self, ref: Union[str, List[str]]):
        if isinstance(ref, list):
            self.references += list(map(self.tokenizer, ref))
        else:
            self.references.append(self.tokenizer(ref))

    def __call__(self, hypotheses: List[str]):
        weights = {f"{n}-gram": ([1. / n] * n) for n in self.grams}
        if self.sample_size > 0:
            sample_size = min(len(self.references), self.sample_size)
            bleu = BLEU(random.sample(self.references, k=sample_size), weights)
        else:
            bleu = BLEU(self.references, weights)
        tokenized_hypotheses = list(map(self.tokenizer, hypotheses))
        scores = list(bleu.get_score(tokenized_hypotheses).values())
        return np.asarray(scores).mean(axis=0)

    def compute_pairwise_bleu(self, texts: List[str]) -> np.ndarray:
        
        tokenized_texts = list(map(self.tokenizer, texts))
        n = len(texts)
        bleu_matrix = np.zeros((n, n))
        
        weights = {f"{n}-gram": ([1. / n] * n) for n in self.grams}
        
        for i in range(n):
            for j in range(n):
                if i != j:  # Skip self-comparison
                    bleu = BLEU([tokenized_texts[i]], weights)
                    score = list(bleu.get_score([tokenized_texts[j]]).values())
                    bleu_matrix[i,j] = np.mean(score)
                else:
                    bleu_matrix[i,j] = 1.0  # Perfect score for self-comparison
                    
        return bleu_matrix

from typing import List, Callable, Union
from transformers import AutoTokenizer, AutoModel
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import manhattan_distances


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class CosineSentenceEmbeddingReward(object):

    def __init__(self, device: str = "cuda", n_samples: int = -1, impl: str ="huggingface", batch_size: int = 32, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = device
        self.n_samples = n_samples
        self.impl = impl
        self.batch_size = batch_size
        
        print("Cossim implementation: ", self.impl)
        print("Cossim n_samples: ", n_samples)
        print("Cossim device: ", self.device)
        if self.impl == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, device=device)
            self.model = AutoModel.from_pretrained(model_name).to(device)
        elif self.impl == "sentencetransformer":
            self.model = SentenceTransformer(model_name.split("/")[-1])
        self.sentence_embeddings = None
        
    def compute_similarity(self, X: List[str], Y: List[str]) -> torch.Tensor:
        phi_X = self._compuate_embeddings(X)
        phi_Y = self._compuate_embeddings(Y)
        sim = torch.diagonal(util.pytorch_cos_sim(phi_X, phi_Y), 0)
        return sim

    def compute_l1_div_rewards(self, X: List[str]) -> np.ndarray:
        LAM_ADV = 0.5        
        LAM_DIV1 = 100
        LAM_DIV2 = 5    
        embeddings = self._compuate_embeddings(X).detach().cpu().numpy()
        dist_matrix = torch.tensor(manhattan_distances(embeddings)) / LAM_DIV1
        div_reward = -1 * torch.mean(torch.exp(-dist_matrix), dim=1) * LAM_DIV2
        return div_reward.cpu().numpy()

    def _compuate_embeddings(self, sentences: List[str]) -> torch.Tensor:
        with torch.no_grad():
            if self.impl == "huggingface":
                encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)

                # Compute token embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                # Perform pooling
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            elif self.impl == "sentencetransformer":
                sentence_embeddings = self.model.encode(sentences, device="cpu", batch_size=self.batch_size, 
                                                convert_to_tensor=True, convert_to_numpy=False)

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings
    
    def append_reference(self, ref: Union[str, List[str]]):
        if not isinstance(ref, List):
            ref = [ref]
        sentence_embeddings = self._compuate_embeddings(ref).cpu()
        # import ipdb; ipdb.set_trace()
        if self.sentence_embeddings is None:
            self.sentence_embeddings = sentence_embeddings
        else:
            self.sentence_embeddings = torch.cat([self.sentence_embeddings, sentence_embeddings], dim=0)
    
    def __call__(self, hypotheses: List[str]) -> np.ndarray:
        if self.sentence_embeddings is None or len(self.sentence_embeddings) == 0:
            return np.zeros(len(hypotheses))            
        hypo_sentence_embeddings = self._compuate_embeddings(hypotheses)
    
        if self.n_samples > 0:
            sample_size = min(len(self.sentence_embeddings), self.n_samples)
            sample_indices = random.sample(range(self.sentence_embeddings.shape[0]), k=sample_size)
            ref_sentence_embeddings = self.sentence_embeddings[sample_indices, :]
        else:
            ref_sentence_embeddings = self.sentence_embeddings

        sims = pairwise_cosine_similarity(hypo_sentence_embeddings, ref_sentence_embeddings)
        return sims.mean(dim=1).cpu().numpy()

    def compute_pairwise_similarities(self, texts: List[str]) -> np.ndarray:
        embeddings = self._compuate_embeddings(texts)
        
        sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)
        return sim_matrix.cpu().numpy()