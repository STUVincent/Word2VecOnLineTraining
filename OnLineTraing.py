# -*- coding: utf-8 -*-
"""
******* 文档说明 ******
基于 Gensim 的词向量增量训练demo

1、语料+语料
    此种方式需要保存已训练好模型

2、词向量+语料
    此种方式主要对于现在已训练好的通用词向量（如GIT词向量或腾讯词向量）进行专业领域语料增量训练
    对于网络通用的庞大词向量库，增量训练语料数量若太少，可能对已有词向量的影响十分有限，具体效果有待分析
    ps： GIT   https://github.com/Embedding/Chinese-Word-Vectors
        腾讯   https://ai.tencent.com/ailab/nlp/embedding.html

# 当前项目: 词向量增量训练
# 创建时间: 2018/11/28 20:45 
# 开发作者: Vincent
# 创建平台: PyCharm Community Edition    python 3.5  Gensim 2.3.0
# 版    本: V1.0
"""
import os
from gensim.models import Word2Vec
from Common import get_corpus

# 词向量长度
EmbeddingLen = 16
# 模型训练最小词数量
MinCount = 10
# 词窗口
Window = 3


# 单独训练
def train(corpus):

    # 创建Word2Vec模型训练
    model = Word2Vec(size=EmbeddingLen, window=Window, min_count=MinCount, iter=5)

    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
    print("语向量维度：{}".format(model.wv.syn0.shape))
    print("【郭靖】相似词：")
    print("  ".join(["{}: {:.2f}".format(word_i, s_i) for word_i, s_i in model.wv.similar_by_word("郭靖", topn=20)]))

    # # 模型中词的信息
    # vocab_info = [[word_i, value_i.count, value_i.index]for word_i, value_i in model.wv.vocab.items()]
    # # 模型中前 20 个词
    # print(sorted(vocab_info, key=lambda x: x[2], reverse=False)[:20])

    return model


# 在线训练 语料 + 语料
def train_online_1(corpus1, corpus2):
    # 创建Word2Vec模型训练
    model = Word2Vec(size=EmbeddingLen, window=Window, min_count=MinCount, iter=5)
    model.build_vocab(corpus1)
    model.train(corpus1, total_examples=model.corpus_count, epochs=model.iter)

    # ####################################### 增量训练
    model.build_vocab(corpus2, keep_raw_vocab=True, update=True)
    model.train(corpus2, total_examples=model.corpus_count, epochs=model.iter)

    print("语向量维度：{}".format(model.wv.syn0.shape))
    print("【郭靖】相似词：")
    print("  ".join(["{}: {:.2f}".format(word_i, s_i) for word_i, s_i in model.wv.similar_by_word("郭靖", topn=40)]))

    return model


# 在线训练 词向量 + 语料
def train_online_2(file, corpus):

    word_embedding_txt = os.path.join('.', 'Embedding', '{}_word2vec.txt'.format(file[:-4]))
    word = list()
    for data_i in open(word_embedding_txt, 'r', encoding='UTF-8'):
        data_i = data_i.split()
        if len(data_i) == EmbeddingLen + 1:
            word.append([data_i[0]]*MinCount)

    # 创建Word2Vec模型训练
    model = Word2Vec(size=EmbeddingLen, window=Window, min_count=MinCount, iter=5)
    model.build_vocab(word)  # 先将词向量表中的词加入到模型中，才能导入其词向量
    # 加入已有词向量
    model.intersect_word2vec_format(word_embedding_txt, lockf=1.0, binary=False)
    # ####################################### 增量训练
    model.build_vocab(corpus, keep_raw_vocab=True, update=True)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

    print("语向量维度：{}".format(model.wv.syn0.shape))
    print("【郭靖】相似词：")
    print("  ".join(["{}: {:.2f}".format(word_i, s_i) for word_i, s_i in model.wv.similar_by_word("郭靖", topn=40)]))

    return model


def main():
    file1 = "金庸-射雕英雄传.txt"
    file2 = "金庸-神雕侠侣.txt"
    corpus1 = get_corpus(file1)
    corpus2 = get_corpus(file2)

    print("\n单独训练：{}".format(file1))
    model = train(corpus1)
    # 导出词向量
    model.wv.save_word2vec_format(os.path.join('.', 'Embedding', '{}_word2vec.txt'.format(file1[:-4])), binary=False)

    print("\n单独训练：{}".format(file2))
    model = train(corpus2)
    # 导出词向量
    model.wv.save_word2vec_format(os.path.join('.', 'Embedding', '{}_word2vec.txt'.format(file2[:-4])), binary=False)

    print("\n单独训练：{} + {}".format(file1, file2))
    train(corpus1+corpus2)

    print("\n语料增量训练：{}   {}".format(file1, file2))
    train_online_1(corpus1, corpus2)

    print("\n{}  词向量增量训练：  {}".format('{}_word2vec.txt'.format(file1[:-4]), file2))
    train_online_2(file1, corpus2)

    print("\n语料增量训练：{}   {}".format(file2, file1))
    train_online_1(corpus2, corpus1)

    print("\n{}  词向量增量训练：  {}".format('{}_word2vec.txt'.format(file2[:-4]), file1))
    train_online_2(file2, corpus1)

if __name__ == '__main__':
    main()
