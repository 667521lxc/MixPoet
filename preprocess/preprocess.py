# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-31 10:10:05
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''

import pickle
import json
import random
import argparse

def parse_args():#解析参数num_unlabelled
    parser = argparse.ArgumentParser(description="The parametrs for the preprocessing tool.")
    parser.add_argument("-n", "--num_unlabelled",  type=int, default=-1, help="The number of unlabelled data to be used.")
    return parser.parse_args()

def outFile(data, file_name): #把data写入文件
    print ("output data to %s, num: %d" % (file_name, len(data)))
    with open(file_name, 'w') as fout:
        for d in data:
            fout.write(d+"\n")

class PreProcess(object):
    """docstring for Binarize"""
    def __init__(self, min_freq=1):
        super(PreProcess, self).__init__()
        self.min_freq = min_freq

    def line2idxes(self, line):#把一行文本转成index，用一个列表表示
        chars = [c for c in line]
        idxes = []
        for c in chars:
            if c in self.dic:
                idx = self.dic[c]
            else:
                idx = self.dic['UNK']
            idxes.append(idx)
        return idxes

    def create_dic(self, poems):#创建了一个字典dic，供上面line2idxes函数使用，poems是一个诗歌列表
        print ("creating the word dictionary...")
        print ("input poems: %d" % (len(poems)))
        count_dic = {}
        for p in poems:#p是一首诗
            poem = p.strip().replace("|", "")#把诗按行分割

            for c in poem:
                if c in count_dic:
                    count_dic[c] += 1
                else:
                    count_dic[c] = 1

        vec = sorted(count_dic.items(), key=lambda d:d[1], reverse=True) #按照次品进行排序
        print ("original word num:%d" % (len(vec)))

        # add special symbols
        # --------------------------------------
        dic = {}
        idic = {}
        dic['PAD'] = 0
        idic[0] = 'PAD'

        dic['UNK'] = 1
        idic[1] = 'UNK'

        dic['<E>'] = 2
        idic[2] = '<E>'

        dic['<B>'] = 3
        idic[3] = '<B>'

        idx = 4
        print ("min freq:%d" % (self.min_freq))

        for c, v in vec:
            if v < self.min_freq:
                break
                #自己注释的continue
            if not c in dic:
                dic[c] = idx
                idic[idx] = c
                idx += 1

        print ("total word num: %s" % (len(dic)))

        return dic, idic  #统计了一下高频，然后加入词典。dic和idic是互逆的关系

    def build_dic(self, infile):#读入infile文件
        with open(infile, 'r') as fin:
            lines = fin.readlines()

        poems = []  #参见上面create_dic函数的poems
        training_lines = []
        for line in lines:
            dic = json.loads(line.strip())
            poem = dic['content']
            poems.append(poem)
            training_lines.extend(poem.split("|"))#每一首诗是一个列表

        dic, idic = self.create_dic(poems)
        self.dic = dic
        self.idic = idic

        # output dic file  存储字典和读取的诗歌txt
        # read
        dic_file = "/kaggle/working/vocab.pickle"
        idic_file = "/kaggle/working/ivocab.pickle"

        print ("saving dictionary to %s" % (dic_file))
        with open(dic_file, 'wb') as fout:
            pickle.dump(dic, fout, -1)

        print ("saving inverting dictionary to %s" % (idic_file))
        with open(idic_file, 'wb') as fout:
            pickle.dump(idic, fout, -1)

        # building training lines
        outFile(training_lines, "/kaggle/working/training_lines.txt")

    def read_corpus(self, infile, with_label=False):   #这个数据集的lable是什么意思？？
        with open(infile, 'r') as fin:
            lines = fin.readlines()

        corpus = []
        for line in lines:
            dic = json.loads(line.strip())
            poem = dic['content']
            keywords = dic['keywords'].split(" ")

            if with_label:
                para = dic['label'].split(" ")
                labels = (int(para[0]), int(para[1]))
            else:
                labels = (-1, -1)

            # we build each instance with each of the keywords每一个关键词一个数据
            for keyword in keywords:
                corpus.append((keyword, poem, labels))

        return corpus

    def build_data(self, corpus): #corpus参考上面的函数
        max_len = -1
        skip_plen_count = 0
        skip_llen_count = 0

        data = []
        for d in corpus:
            keyword = d[0]
            poem = d[1]

            lines = poem.split("|")

            if len(lines) != 4:
                skip_plen_count += 1   #跳过的计数，不是四句诗
                continue

            skip = False
            line_idxes_vec = []
            for line in lines:
                idxes = self.line2idxes(line)
                length = len(idxes)
                if length != 5 and length != 7:
                    skip = True   #跳过的计数，不是五字或者七字
                    break

                max_len = max(max_len, length)
                line_idxes_vec.append(idxes)

            if skip:
                skip_llen_count += 1
                continue

            assert len(line_idxes_vec) == 4

            assert len(keyword) <= 2
            key_idxes = self.line2idxes(keyword)

            data.append((key_idxes, line_idxes_vec, d[2][0], d[2][1]))  #把文字转换成对应的index

        print ("data num: %d, skip plen: %d, skip llen: %d, max length: %d" %\
            (len(data), skip_plen_count, skip_llen_count, max_len))

        return data


    def build_test_data(self, infile, out_inp_file, out_trg_file):
        with open(infile, 'r') as fin:
            lines = fin.readlines()

        keywords_vec = []
        poems_vec = []
        for line in lines:
            dic = json.loads(line.strip()) #一首诗
            poem = dic['content']
            keywords = dic['keywords'].split(" ")

            # NOTE: here we just sample one keyword from the extracted
            #   keywords as the input for testing

            keywords_vec.append(random.sample(keywords, 1)[0]) #sample(list, k)返回一个长度为k新列表，新列表存放list所产生k个随机唯一的元素
            poems_vec.append(poem)

        outFile(keywords_vec, out_inp_file)
        outFile(poems_vec, out_trg_file)        #为什么要写入文件


    def process(self, unlabelled_num=None):#这个可以看作类中的主函数
        # build the word dictionary
        self.build_dic("/kaggle/input/poetry/CCPC/ccpc_train_v1.0.json")#self.build_dic("ccpc_train.json")

        # build training and validation datasets
        unlabelled_train = self.read_corpus("/kaggle/input/poetry/CCPC/ccpc_train_v1.0.json", False)
        unlabelled_valid = self.read_corpus("/kaggle/input/poetry/CCPC/ccpc_valid_v1.0.json", False)

        if unlabelled_num is not None:
            unlabelled_train = random.sample(unlabelled_train, unlabelled_num)

        labelled_train = self.read_corpus("cqcf_train_sample.json", True)
        labelled_valid = self.read_corpus("cqcf_valid_sample.json", True)

        semi_train_data = self.build_data(unlabelled_train+labelled_train)
        semi_valid_data = self.build_data(unlabelled_valid+labelled_valid)

        random.shuffle(semi_train_data)
        random.shuffle(semi_valid_data)

        print ("training data: %d" % (len(semi_train_data)))
        print ("validation data: %d" % (len(semi_valid_data)))

        train_file = "/kaggle/working/semi_train.pickle"
        print ("saving training data to %s" % (train_file))
        with open(train_file, 'wb') as fout:
            pickle.dump(semi_train_data, fout, -1)

        valid_file = "/kaggle/working/semi_valid.pickle"
        print ("saving validation data to %s" % (valid_file))
        with open(valid_file, 'wb') as fout:
            pickle.dump(semi_valid_data, fout, -1)

        # build testing inputs and trgs     这个地方要注意，因为kaggle 的inputdiamond文件无法修改只读，因此我把test_inps.txt放在了working文件夹
        self.build_test_data("/kaggle/input/poetry/CCPC/ccpc_test_v1.0.json", "/kaggle/working/test_inps.txt", "/kaggle/working/test_trgs.txt")


def main():
    args = parse_args()
    if args.num_unlabelled == -1:
        n = None
    else:
        n = args.num_unlabelled

    processor = PreProcess()
    processor.process(n)#执行process(n)这个类中的主函数



if __name__ == "__main__":
    main()