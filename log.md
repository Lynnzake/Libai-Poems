
2020/8/8 更新爬虫文件和数据处理文件，添加一个简易的模型

1. 爬虫采用scrapy框架
2. 数据处理包括文本格式清洗，n-grams语言模型的构建，以及相关的分词处理
3. 更新一个简单的模型Model1，准确率很低，损失很高，后续再做调整

2020/8/9 更新模型model2
模型1的准确率很低，在想是不是因为数据长度过长的原因，尝试对除五言，七言之外的诗句进行截断
1. 查看Token3.txt可以发现，
      落叶别树，
      飘零随风。


      客无所托，
      悲与此同。
  存在这样的情况，故重写生成Tokens的sequenceForToken2()函数，对数据的长度等进行处理

2. 整理之前最长序列长度为16，使用截断，当长度大于8时进行截断处理，虽影响了部分句子的完整性，但值得一试。
  new_input_sequences = []
  for x in input_sequences:
      if len(x) > 8:
          x = x[:8]
          new_input_sequences.append(x)
      else:
          new_input_sequences.append(x)

3. 此方法试过之后并不如意，结果比Model1还差.接下来试着调整模型1参数.在调试的过程中发现，随着Embedding层维度的降低，模型的准确率提高了，我的理解是这样的：因为古诗的选词讲究的是“推敲”二字，所以语言更加精炼，也导致各个字之间的关联性可能并没有那么高，试着降低维度,发现提高的并不多。下一步准备尝试增加数据吧。
    Embedding(total_words,16,input_length=max_sequence_len-1), 迭代50次之后模型的准确率为0.2350
    
2020/8/10 增加数据之后发现模型准确率仍然很低，考虑从定长的五言、七言开始训练Model3