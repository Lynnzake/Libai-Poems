模型一： 使用Tokenizer()类进行数据加载与处理sequenceForToken()函数
1. 为适应Tokenizer()对数据的要求，对文本就行分割处理，原文形式：疑是银河落九天。——>疑 是 银 河 落 九 天 。
2. 为避免加载到输入的序列过长，每遇到一个标点符号，便换行显示