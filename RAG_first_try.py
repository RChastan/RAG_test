# =============================================================================
# Dataset Overview
# The dataset we are using is sourced from the Llama 2 ArXiv papers. It is a collection of academic papers from ArXiv, a repository of electronic preprints approved for publication after moderation. Each entry in the dataset represents a "chunk" of text from these papers.
# =============================================================================
# conda install datasets

from datasets import load_dataset

dataset = load_dataset(
    "jamescalam/llama-2-arxiv-papers-chunked",
    split="train"
)

dataset
dataset[200]

#%% Embeding
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
# embeddings = model(input_ids).pooler_output

#%%
input_ids = tokenizer(dataset['chunk'],return_tensors="pt",padding=True)["input_ids"]
print(input_ids.size)

#%%
# import numpy as np
# batch = 100
# i = 0
# while i<input_ids.size()[0]:
#   print(i)
#   if i+batch>input_ids.size()[0]:
#     embeddings = model(input_ids[i:input_ids.size()[0]]).pooler_output
#   else:
#     embeddings = model(input_ids[i:i+batch]).pooler_output
#   if i==0:
#     np_emb = embeddings.detach().numpy()
#   else :
#     np_ = embeddings.detach().numpy()
#     np_emb = np.concatenate((np_emb,np_),axis=0)
#   i+=batch

embeddings = model(input_ids[0:20]).pooler_output
A = embeddings.detach().numpy()

#%% Question
q = "Who did found that max-pooling can lead to faster convergence ?"
q="max-pooling can lead to faster convergence"

q="Impact of max-pooling on convergence"

q_ids = tokenizer(q,return_tensors="pt",padding=True)["input_ids"]
q_emb = model(q_ids).pooler_output
q_emb[0]
a = q_emb.detach().numpy()

#%% Cosine Similarity Search
from sklearn.neighbors import NearestNeighbors

NNmodel = NearestNeighbors(n_neighbors=3,
                            metric='cosine')
NNmodel.fit(A)

res = NNmodel.kneighbors(a, 3, return_distance=True)
print(dataset['chunk'][res[1][0][0]])

#%% Question
q = "What did initial experiments on NORB showed ?" # Passage 18

q_ids = tokenizer(q,return_tensors="pt",padding=True)["input_ids"]
q_emb = model(q_ids).pooler_output
q_emb[0]
a = q_emb.detach().numpy()

res = NNmodel.kneighbors(a, 3, return_distance=True)
print(res[1][0][0])
print(dataset['chunk'][res[1][0][0]])

#%% Similarity Search with FAISS (ANN)
# conda install -c pytorch faiss-cpu
# conda install -c conda-forge faiss

import faiss
