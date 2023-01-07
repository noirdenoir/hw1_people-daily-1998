#!/usr/bin/env python
# coding: utf-8



import time
import re
from collections import defaultdict
from gensim import corpora,models,similarities

#下面的函数用来预处理文本
def clear(filename):
    
    with open(filename, encoding='UTF-8') as file:
        texts = file.readlines()

    new_file = open("pro1998.txt", 'w', encoding='UTF-8')
    
    #构建停用词表
    cn_stopwords=[]

    with open('cn_stopwords.txt',encoding='UTF-8') as stoplist:
    
        for line in stoplist.readlines():
            
            cn_stopwords.append(line.strip()) 
    
    #进行清洗
    for sent in texts:
        if sent !='\n':
            sent=re.sub(r'\d*-\d*-\d*-\d*/m\s','',sent) #去除句首时间戳加上后面带的空格
            sent=re.sub(r'/[A-z,a-z]+','',sent)   #去除词性标注
    #         sent=sent.strip() #去掉多余空格
            sent=[w for w in sent.strip().split() if w not in cn_stopwords] #去掉停用词
            sent=' '.join(sent)+' '  #注意在后面加上+'\n'与不加的区别
        else:
            sent=sent
        new_file.write(sent)

#下面的函数将处理后的文本分成篇放进列表
def split_text(filename):
    with open(filename,encoding='UTF-8') as nfile:
        lst = [p.strip() for p in nfile.read().split('\n') if p !='\n']
#         print(len(lst))
    return lst


def to_corpus(filename):
        frequency = defaultdict(int)
        
        #取出各篇分别放进列表      
        with open(filename,encoding='UTF-8') as nfile:
            lst = [p for p in nfile.read().split('\n')]
            
            #统计各词出现的频率
            for p in lst:
                for w in p.split(' '):
                    frequency[w] += 1
#             print('\n',frequency,'\n')
            
            #只保留出现频率大于一的词
            processed_corpus = [[w for w in p.split(' ') if w!= '' and frequency[w] > 1] for p in lst]
            
            return processed_corpus

#下面的函数用lsi计算相似度
def sim_lsi():
    #调用上面两个函数的返回值
    processed_corpus = to_corpus('pro1998.txt')
    alltexts=split_text('pro1998.txt')
    
    dictionary = corpora.Dictionary(processed_corpus)
    dictionary.save('1998.dict')

    bow_corpus=[dictionary.doc2bow(text) for text in processed_corpus]  
  
    lsi = models.LsiModel(bow_corpus)
    index = similarities.SparseMatrixSimilarity(lsi[bow_corpus],num_features=len(dictionary))
    
    sims=[]
    for i in range(len(alltexts)):
        querytxt=alltexts[i].split(' ')
        querybow=dictionary.doc2bow(querytxt)
        
        sims.append(index[lsi[querybow]])
        
#     print(sims)
# --------------------------------------------------
   
#下面的函数用tfidf计算相似度
def sim_tfidf():
    #调用上面两个函数的返回值
    processed_corpus = to_corpus('pro1998.txt')
    alltexts=split_text('pro1998.txt')
    
    dictionary = corpora.Dictionary(processed_corpus)
    dictionary.save('1998.dict')

    bow_corpus=[dictionary.doc2bow(text) for text in processed_corpus]  
#     print(bow_corpus)       
#     print(len(bow_corpus))
  
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    
    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=len(dictionary))
    
    sims=[]
    for i in range(len(alltexts)):
        querytxt=alltexts[i].split(' ')
        query_bow = dictionary.doc2bow(querytxt)
        
        sims.append(index[tfidf[query_bow]])
#     print(sims)
    
if __name__ == '__main__':
    start = time.time()
    clear('199801.txt')
    end = time.time()
    print("清洗所用时间为：", end - start, "s")
    
    split_text('pro1998.txt')
    to_corpus('pro1998.txt')
    
    start = time.time()   
    sim_lsi() 
    end = time.time()
    print("lsi计算相似度时间为：", end - start, "s")
    
    start = time.time()   
    sim_tfidf() 
    end = time.time()
    print("tfidf计算相似度时间为：", end - start, "s")





