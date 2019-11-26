import json
import numpy as np
from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

data = json.load(open('visdial_1.0_train.json'))

data = data['data']
ans = data['answers']
ques = data['questions']
dials = data['dialogs']

freq = np.zeros(len(ans), dtype=np.int32)
n = len(dials)
m = 10

for i in range(n):
  for j in range(m):
    freq[dials[i]['dialog'][j]['answer']] += 1

indices = np.argsort(-freq)
freq_ans = []
k = 10000
for i in range(k):
  freq_ans.append(ans[indices[i]])

similarity_yes = np.zeros(k)
similarity_no = np.zeros(k)

module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")

input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape)

with tf.Session() as sess:
  spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)
print("SentencePiece model loaded at {}.".format(spm_path))

def process_to_IDs_in_sparse_format(sp, sentences):
  # An utility method that processes sentences with the sentence piece processor
  # 'sp' and returns the results in tf.SparseTensor-similar format:
  # (values, indices, dense_shape)
  ids = [sp.EncodeAsIds(x) for x in sentences]
  max_len = max(len(x) for x in ids)
  dense_shape=(len(ids), max_len)
  values=[item for sublist in ids for item in sublist]
  indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
  return (values, indices, dense_shape)

messages = freq_ans
values, indices, dense_shape = process_to_IDs_in_sparse_format(sp,messages)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  message_embeddings = session.run(
      encodings,
      feed_dict={input_placeholder.values: values,
                input_placeholder.indices: indices,
                input_placeholder.dense_shape: dense_shape})


