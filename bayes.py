# -*- coding: UTF-8 -*-

"""
date: 2018-9-12
note: bayes
Ref:  《机器学习实战》- https://github.com/apachecn/AiLearning
贝叶斯公式
p(xy)=p(x|y)p(y)=p(y|x)p(x)
p(x|y)=p(y|x)p(x)/p(y)

demo: 屏蔽社区留言板的侮辱性言论
"""

import numpy as np


def load_data_set():
    """
    :return: 单词列表posting_list, 所属类别class_vec
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
    return posting_list, class_vec


def create_vocab_list(data_set):
    """
    get all vocab list
    :param data_set: data
    :return: a list of vocab set
    """
    vocab_set = set()
    for item in data_set:
        vocab_set = vocab_set | set(item)
    return list(vocab_set)


def set_of_words2Vec(vocab_list, input_set):
    """
     check if vocab in vocab_list, 1 if true.
    :param vocab_list:
    :param input_set:
    :return: like [1, 0, 0, 1...]
    """
    result = [1 if item in input_set else 0 for item in vocab_list]
    return result


def train_naive_bayes(train_mat, train_category):
    """

    :param train_mat:
    :param train_category:
    :return:
    """
    train_doc_num = len(train_mat)
    words_num = len(train_mat[0])
    pos_abusive = np.sum(train_category) / train_doc_num
    p0num = np.ones(words_num)
    p1num = np.ones(words_num)
    p0num_all = 2.0
    p1num_all = 2.0

    for i in range(train_doc_num):
        if train_category[i] == 1:
            p1num += train_mat[i]
            p1num_all += np.sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0num_all += np.sum(train_mat[i])
    p1vec = np.log(p1num / p1num_all)
    p0vec = np.log(p0num / p0num_all)
    return p0vec, p1vec, pos_abusive


def classify_naive_bayes(vec2classify, p0vec, p1vec, p_class_1):
    p1 = np.sum(vec2classify * p1vec) + np.log(p_class_1)
    p0 = np.sum(vec2classify * p0vec) + np.log(1-p_class_1)
    if p1 > p0:
        return 1
    else:
        return 0



def test_naive_bayes():
    list_post, list_classes = load_data_set()
    vocab_list = create_vocab_list(list_post)

    train_mat = []
    for post in list_post:
        train_mat.append(set_of_words2Vec(vocab_list, post))

    p0v, p1v, p_abusive = train_naive_bayes(np.array(train_mat), np.array(list_classes))

    test_case_1 = ['love', 'my', 'dalmation']
    test_case_1_doc = np.array(set_of_words2Vec(vocab_list, test_case_1))
    print('Case 1 result is:', classify_naive_bayes(test_case_1_doc, p0v, p1v, p_abusive))

    test_case_2 = ['stupid', 'garbage']
    test_case_2_doc = np.array(set_of_words2Vec(vocab_list, test_case_2))
    print('Case 2 result is: {}'.format(classify_naive_bayes(test_case_2_doc, p0v, p1v, p_abusive)))


if __name__ == '__main__':
    test_naive_bayes()
