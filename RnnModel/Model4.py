#!/usr/bin/env python
# coding: utf-8

# In[1]:


#在Token4.txt上进行训练，在诗人李白，李贺，杜甫的基础上对诗人诗句进行扩增
#在Model13-五言+七言的基础上根据增加的数据重新训练
#对源文件进行处理，去除标点符号
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,LSTM,Embedding,Bidirectional,Dropout,Conv1D
from keras.models import Model,Sequential
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np


# In[2]:


#建立关于语料库的字典
tokenizer = Tokenizer()
data = open(r'H:\MyProject\libai_poem\RnnModel\Tokens4.txt').read()


# In[3]:


corpus = data.split('\n')


# In[4]:


print(corpus[66])


# In[5]:


tokenizer.fit_on_texts(corpus)


# In[6]:


total_words = len(tokenizer.word_index) + 1

print(total_words)


# In[7]:


word_index = tokenizer.word_index
print(type(word_index))


# In[8]:


word_index.get("汉",['UNK'])


# In[9]:


#统一输入序列长度
input_sequences = []
for line in corpus:
    #序列数字化，数字对应该文字在tokenizer.word_index中的位置所在的索引
    #print("="*30,len(line))
    if len(line) == 14 or len(line) == 10:# 处理五言和七言的句子
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
        input_sequences.append(n_gram_sequence)


# In[10]:


#pad sequence,从上述可看出，input_sequences的长度并不统一
max_sequence_len = max([len(x) for x in input_sequences])
print(max_sequence_len)
padded_input = np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))


# In[ ]:


#查看共有多少训练数据
print(len(padded_input))


# In[11]:


#五言+七言
##构建X，Y
xs,labels = padded_input[:,:-1],padded_input[:,-1]
##热编码
ys = tf.keras.utils.to_categorical(labels,num_classes=total_words)


# In[ ]:


#train/test set 


# In[45]:


model = Sequential([
    Embedding(total_words,32,input_length=max_sequence_len-1),
    Bidirectional(LSTM(64,return_sequences=True)),
    #Dropout(0.2),#添加dropout层
    Bidirectional(LSTM(64)),
    Dense(total_words,activation='softmax')
])


# In[46]:


model.summary()


# In[48]:


adam = Adam(lr=0.01)


# In[49]:


model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])


# In[51]:


history = model.fit(xs,ys,epochs=150,batch_size=1024,verbose=2)#采用mini-batch进行训练


# In[52]:


model.save('./model_with150epochs.h5')


# In[53]:


import matplotlib.pyplot as plt


# In[58]:


def plot(history,string):
    plt.plot(history.history[string])
    plt.xlabel('epochs')
    plt.ylabel(string)
    plt.savefig('./accuracy.jpg')
    plt.show()
plot(history,'accuracy')


# In[55]:


## 预测
seed_text = "河"
next_words = 5
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


# In[19]:


#参数
time_steps = 10
features = 1
input_shape = [time_steps, features]
batch_size = 32


# In[94]:


model = Sequential()
model.add(LSTM(4, input_shape=input_shape,  return_sequences=True))
#model.add(LSTM(32,return_sequences=True))


# In[95]:


model.summary()


# In[ ]:





# In[ ]:




