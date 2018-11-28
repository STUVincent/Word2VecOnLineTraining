词向量增量训练 demo
Gadget
===========================
基于 Gensim 的词向量增量训练demo

1、语料+语料
    此种方式需要保存已训练好模型

2、词向量+语料
    此种方式主要对于现在已训练好的通用词向量（如GIT词向量或腾讯词向量）进行专业领域语料增量训练
    对于网络通用的庞大词向量库，增量训练语料数量若太少，可能对已有词向量的影响十分有限，具体效果有待分析
    ps： GIT   https://github.com/Embedding/Chinese-Word-Vectors
        腾讯   https://ai.tencent.com/ailab/nlp/embedding.html

![](/img/词向量增量训练.JPG  '词向量增量训练.JPG')
		
|Author|Vincent|
|---|---|
|E-mail|stuvincent@163.com|