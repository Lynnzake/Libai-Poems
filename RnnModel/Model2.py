#!/usr/bin/env python
# coding: utf-8

# In[10]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,LSTM,Embedding,Bidirectional
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


# In[11]:


#建立关于语料库的字典
tokenizer = Tokenizer()
data = open(r'H:\MyProject\libai_poem\RnnModel\Tokens2.txt').read()


# In[12]:


print(data)


# In[13]:


corpus = data.split('\n')


# In[14]:


print(corpus[0])


# In[15]:


tokenizer.fit_on_texts(corpus)


# In[16]:


total_words = len(tokenizer.word_index) + 1


# In[17]:


#统一输入序列长度
input_sequences = []
for line in corpus:
    #序列数字化，数字对应该文字在tokenizer.word_index中的位置所在的索引
    sequence = tokenizer.texts_to_sequences(line)
    print(sequence)
    #序列中有多余的空列表，处理掉
    new_sequence = []
    for i in sequence:
        if i != []:
            new_sequence.append(i)
            
    #开始处理输入序列
    for i in range(1,len(new_sequence)):
        n_gram_sequence = new_sequence[:i+1]
        print("n-gram-sequence:",n_gram_sequence)
        if len(n_gram_sequence) == 1:
            continue
        input_sequences.append(n_gram_sequence)


# In[33]:


new_input_sequences = []
for x in input_sequences:
    if len(x) > 8:
        x = x[:8]
        new_input_sequences.append(x)
    else:
        new_input_sequences.append(x)


# In[34]:


#pad sequence,从上述可看出，input_sequences的长度并不统一
max_sequence_len = max([len(x) for x in new_input_sequences])
print(max_sequence_len)
padded_input_sequences = np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))


# In[35]:


#构建X，Y
xs,labels = padded_input_sequences[:,:-1],padded_input_sequences[:,-1]
#热编码
ys = tf.keras.utils.to_categorical(labels,num_classes=total_words)


# In[36]:


model = Sequential([
    Embedding(total_words,16,input_length=max_sequence_len-1),
    Bidirectional(LSTM(64,return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(total_words,activation='softmax')
])


# In[37]:


model.summary()


# In[38]:


adam = Adam(lr=0.01)


# In[39]:


model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])


# In[40]:


history = model.fit(xs,ys,epochs=50,verbose=2)


# In[41]:


import matplotlib.pyplot as plt


# In[42]:


def plot(history,string):
    plt.plot(history.history[string])
    plt.xlabel('epochs')
    plt.ylabel(string)
    plt.show()
plot(history,'accuracy')


# In[43]:


#预测
seed_text = "峰"
next_words = 25
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict_classes(token_list,verbose=0)
    output = ''
    for word,index in tokenizer.word_index.items():
        if index == predicted:
            output = word
            break
    seed_text += output
print(seed_text)


# In[ ]:




