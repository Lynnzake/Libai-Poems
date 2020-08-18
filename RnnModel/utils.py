"""负责对数据进行处理"""
import json
import os
abspath = 'h:\MyProject\libai_poem\RnnModel'

def combine_poets(filepath,outfile):
    """将所有诗人的诗合为一个文件"""
    file_list = os.listdir(filepath)
    print(file_list)
    with open(abspath + outfile,'a',encoding='utf-8') as f:
        for file in file_list:
            if file.endswith('.jason'):
                with open(filepath + '\\' + file,'r',encoding='utf-8') as fh:
                    data = fh.read()
                    print(len(data))
                f.write(data)
    f.close()

def dataclean(filename):
    """对爬取的数据进行清理
    参数：
        filename - 存储李白诗集的json文件

    补充：
    相关中文字符正则匹配
    匹配中文字符的正则表达式： [\u4e00-\u9fa5]
    匹配双字节字符(包括汉字在内)：[^\x00-\xff]

    string strRegex = @"[\u4e00-\u9fa5]|[\（\）\《\》\——\；\，\。\“\”\<\>\！]";
    其中前半部分表示匹配中文字符，后半部分为需要匹配的标点符号。
    """
    with open(abspath + filename, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    # print(len(data))
    # print(data[10]) 格式为{'content':[句子]}
    content_list = []   #存储所有的句子
    for j in data:
        content = j['content']
        for i in content:
            content_list.append(i)
    # print(len(content_list))
    # print(content_list[0])
    new = []    #对数据进行清洗，去掉无关字符
    for i in content_list:
        if (i.startswith('\n') and i.endswith(' ')) or i.startswith(' ') or i.startswith("展") or i.startswith("收") or i.startswith('('):
            pass
        else:
            i.strip(' ')
            new.append(i)
    # print(len(new))
    # for i in range(5):
    #     print(new[i])
    #     print('\n')
    with open(abspath + '\\clean_poems_all1.txt','a') as f:
        for i in range(len(new)):
            f.write(new[i])
            f.write('\n')
        f.close()

def word2vec(word):
    """文字转向量，因为数据集较小，故采用热编码形式，不使用word embedding
    参数：
        word - 需要转化为向量的字
    返回：
        word的热编码
    """

def getTopNwords(n,words_list):
    """获取李白诗集词语出现频率前n项的词语(去停用词)
    参数：
        n - 频率前n的词语
        words_list - 单个词的列表
    
    返回：
        topwords_list - 返回频率考前的词语的列表
    """

def combine_words(filename,outfile):
    """将所有的句子连接成一个长句"""
    str_words = ''
    with open(abspath + filename, 'r') as f:
        while f.readline():  # 遍历每行
            line = f.readline()  # 读取当前行
            line = line.strip(' ')
            if len(line) < 5:
                continue
            curr_line = line.split('\n')[0]
            #print(curr_line)
            str_words += curr_line
            #print(str_words)
    #print(len(str_words))
    with open(abspath + outfile,'w') as f:
        f.write(str_words)
        f.close()

def gram_1_LM(filename,outfile):
    """根据诗集文件创建1-gram语言模型,模型LM_1.txt需保存到硬盘中
    参数：filename - 将所有句子连接成一个字符串的文件
        outfile - n-grams模型的保存路径
    返回：LM_1 - (词，次数)的统计字典
    """
    word_dict = {}
    with open(abspath + filename,'r') as f:
        list_words = f.read()
        for i in range(len(list_words)):  # 将字加入到字典中
            if list_words[i] not in word_dict:
                word_dict[list_words[i]] = 1
            else:
                word_dict[list_words[i]] += 1
    for key,value in word_dict.items():
        with open(abspath + outfile,'a') as f:
            if key == '，' or key == '。' or key == ' ':
                print(key,value)
            else:
                f.write(key + '\t' + str(value) + '\n')
    f.close()
    return word_dict

def gram_2_LM(filename,outfile):
    """根据诗集文件创建1-gram语言模型,模型LM_2.txt需保存到硬盘中
    参数：filename - 将所有句子连接成一个字符串的文件
        outfile - n-grams模型的保存路径
    返回：LM_2 - (词，次数)的统计字典
    """
    word_dict = {}
    with open(abspath + filename, 'r') as f:
        list_words = f.read()
        for i in range(len(list_words)):  # 将字加入到字典中
            if list_words[i:i+2] not in word_dict:
                word_dict[list_words[i:i+2]] = 1
            else:
                word_dict[list_words[i:i+2]] += 1
    for key, value in word_dict.items():
        with open(abspath + outfile, 'a') as f:
            f.write(key + '\t' + str(value) + '\n')
    f.close()

def gram_3_LM(filename, outfile):
    """根据诗集文件创建1-gram语言模型,模型LM_3.txt需保存到硬盘中
    参数：filename - 将所有句子连接成一个字符串的文件
        outfile - n-grams模型的保存路径
    返回：LM_3 - (词，次数)的统计字典
    """
    word_dict = {}
    with open(abspath + filename, 'r') as f:
        list_words = f.read()
        for i in range(len(list_words)):  # 将字加入到字典中
            if list_words[i:i+3] not in word_dict:
                word_dict[list_words[i:i+3]] = 1
            else:
                word_dict[list_words[i:i+3]] += 1
            #print(list_words[i:i+3])
    for key, value in word_dict.items():
        with open(abspath + outfile, 'a') as f:
            f.write(key + '\t' + str(value) + '\n')
    f.close()

def sequenceForToken(filename,outfile):
    """生成适合Tokenizer()类的文件"""
    with open(abspath + filename,'r') as f:
        lines = f.readlines()
        new_line = ''   #构建存储句子的列表
        for line in lines:
            for i in line:
                if i == '，' or i == '。' or i == '？' or i == '；' or i == '！':
                    new_line += ( i + '\n')
                elif i == '\n':
                    pass
                else:
                    new_line += (i)
    with open(abspath + outfile,'w') as f:
        f.write(new_line)
        f.close()

def sequenceForToken2(filename,outfile):
    """生成适合Tokenizer()类的文件"""
    with open(abspath + filename,'r') as f:
        lines = f.readlines()
        new_line = ''   #构建存储句子的列表
        for line in lines:
            line = line.strip(' ')
            if len(line) < 5:  # 如果句子长度小于5，舍弃
                continue
            for i in line:
                if i == '，' or i == '。' or i == '？' or i == '；' or i == '！' or i == ',':
                    if i == ',':
                        i = '，'
                    new_line += '\n'
                elif i == '\n' or i == ' ':
                    pass
                else:
                    new_line += (i + ' ')
    with open(abspath + outfile,'w') as f:
        f.write(new_line)
        f.close()

if __name__ == "__main__":
    #combine_poets('H:\MyProject\libai_poem\libai_spider', '\poems.jason')
    #dataclean('\\poems1.jason')
    #dataclean('\\poems.jason')
    #combine_words('\clean_poems_all.txt','\combine_words_all.txt')
    #gram_1_LM('\combine_words_all.txt','\LMs\LM_1.txt')
    #gram_2_LM('\combine_words_all.txt','\LMs\LM_2.txt')
    #gram_3_LM('\combine_words_all.txt', '\LMs\LM_3.txt')
    #sequenceForToken2('\clean_poems_all.txt', '\Tokens3.txt')
    sequenceForToken2('\clean_poems_all_most.txt', '\Tokens4.txt')
