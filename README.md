Text Generation 文本生成

基于RNN创建一个模拟李白诗的模型；李白诗使用爬虫进行爬取

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

3. 此方法试过之后并不如意，结果比Model1还差.接下来试着调整模型1参数
