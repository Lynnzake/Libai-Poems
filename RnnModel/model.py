"""实现text_generation模型"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,LSTM,Embedding,Bidirectional
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
#建立关于语料库的字典
tokenizer = Tokenizer()
data = open(r'libai_poem\RnnModel\clean_poems.txt').read()
print(data)
corpus = data.split('\n')
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
#统一输入序列的长度
input_sequences = []
for line in corpus:
    print(line)
    token_list = tokenizer.texts_to_sequences([line])
    print(token_list)
    for i in range(1,len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


