# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi and Jiannan Liang
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-30 21:29:37
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet. 这个过滤器filter是用来处理韵律和重复字符的，去除有病的句子
'''
import pickle
import random
import copy
import os
from rhythm_tool import RhythmRecognizer

#https://blog.csdn.net/weixin_38481963/article/details/109906338   参考这篇文章可以了解filter的东西，主要还是在CNN领域
class PoetryFilter(object):
    def __init__(self, vocab, ivocab, data_dir):
        self.__vocab = vocab
        self.__ivocab = ivocab

        '''
        rhythm patterns.  韵律，但我不明白这些代号指代诗歌里的哪种模式，走着看吧##答：大致知道是平仄就行
        for Chinese quatrains, we generalize four main poem-level patterns 
        ''' 
        self.__RHYTHM_TYPES = [[0, 1, 3, 2], [1, 2, 0, 1], [2, 1, 3, 2], [3, 2, 0, 1]] #这一行表示一首诗四句话的平仄模式

        '''
        we genelize four main line-level rhythm patterns for 5-char line and 7-char line respectively.
        0: level tone (ping); 1: oblique tone (ze); -1: either
        '''
        self.__RHYTHM_PATTERNS = {7: [[-1, 1, -1, 0, 0, 1, 1], [-1, 0, -1, 1, 1, 0, 0],
            [-1, 1, 0, 0, 1, 1, 0], [-1, 0, -1, 1, 0, 0, 1]],
            5: [[-1, 0, 0, 1, 1], [-1, 1, 1, 0, 0], [0, 0, 1, 1, 0], [-1, 1, 0, 0, 1]]} #这个列表表示一行5，7个字的平仄模式
        self.__rhythm_tool = RhythmRecognizer("D:/资料/毕设/Kaggle/Datasets-master/CRRD/pingsheng.txt", "D:/资料/毕设/Kaggle/Datasets-master/CRRD/zesheng.txt")
        self.__load_rhythm_dic("D:/资料/毕设/Kaggle/Datasets-master/CRRD/pingsheng.txt", "D:/资料/毕设/Kaggle/Datasets-master/CRRD/zesheng.txt")
        self.__load_rhyme_dic("D:/资料/毕设/Kaggle/Datasets-master/CRRD/pingshui.txt", "D:/资料/毕设/Kaggle/Datasets-master/CRRD/pingshui_amb.pkl")#这里是韵脚
        self.__load_line_lib(data_dir+"training_lines.txt") 

        """         
        self.__rhythm_tool = RhythmRecognizer("/kaggle/input/poetry/CRRD/pingsheng.txt", "/kaggle/input/poetry/CRRD/zesheng.txt")
        self.__load_rhythm_dic("/kaggle/input/poetry/CRRD/pingsheng.txt", "/kaggle/input/poetry/CRRD/zesheng.txt")
        self.__load_rhyme_dic("/kaggle/input/poetry/CRRD/pingshui.txt", "/kaggle/input/poetry/CRRD/pingshui_amb.pkl")#这里是韵脚
        self.__load_line_lib(data_dir+"training_lines.txt") 
        """


    def __load_line_lib(self, data_path): #下面两个load函数比较基本，读文件，然后加入列表即可
        self.__line_lib = {}

        with open(data_path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()

        for line in lines:
            line = line.strip()
            self.__line_lib[line] = 1

        print ("  line lib loaded, %d lines" % (len(self.__line_lib)))


    def __load_rhythm_dic(self, level_path, oblique_path):#oblique：间接的地址
        with open(level_path, 'r', encoding='utf-8') as fin:
            level_chars = fin.read()

        with open(oblique_path, 'r', encoding='utf-8') as fin:
            oblique_chars = fin.read()

        self.__level_list = []
        self.__oblique_list = []
        # convert char to id
        for char, idx in self.__vocab.items():
            if char in level_chars:
                self.__level_list.append(idx)

            if char in oblique_chars:
                self.__oblique_list.append(idx)

        print ("  rhythm dic loaded, level tone chars: %d, oblique tone chars: %d" %\
            (len(self.__level_list), len(self.__oblique_list)))


    #------------------------------------------
    def __load_rhyme_dic(self, rhyme_dic_path, rhyme_disamb_path):  #这个函数代码比上面两个麻烦，但是逻辑不难
        self.__rhyme_dic = {} # char id to rhyme category ids
        self.__rhyme_idic = {} # rhyme category id to char ids

        with open(rhyme_dic_path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()

        amb_count = 0
        for line in lines:
            (char, rhyme_id) = line.strip().split(' ')
            if char not in self.__vocab:
                continue
            char_id = self.__vocab[char]
            rhyme_id = int(rhyme_id) #对读取的文件进行处理，得到每一行的char id和rhyme id

            if not char_id in self.__rhyme_dic:
                self.__rhyme_dic.update({char_id:[rhyme_id]})
            elif not rhyme_id in self.__rhyme_dic[char_id]:
                self.__rhyme_dic[char_id].append(rhyme_id)
                amb_count += 1 #把上面处理得到的东西加入字典，更新一下

            if not rhyme_id in self.__rhyme_idic:
                self.__rhyme_idic.update({rhyme_id:[char_id]})
            else:
                self.__rhyme_idic[rhyme_id].append(char_id)

        print ("  rhyme dic loaded, ambiguous模棱两可的 rhyme chars: %d" % (amb_count))

        # load data for rhyme disambiguation押韵消歧
        self.__ngram_rhyme_map = {} # rhyme id list of each bigram or trigram每个二元或三元的押韵id列表
        self.__char_rhyme_map = {} # the most likely rhyme id for each char每个字符最可能的押韵id
        # load the calculated data, if there is any
        #print (rhyme_disamb_path)
        assert rhyme_disamb_path is not None and os.path.exists(rhyme_disamb_path)

        with open(rhyme_disamb_path, 'rb') as fin:
            self.__char_rhyme_map = pickle.load(fin)
            self.__ngram_rhyme_map = pickle.load(fin)

            print ("  rhyme disamb data loaded, cached chars: %d, ngrams: %d"
                % (len(self.__char_rhyme_map), len(self.__ngram_rhyme_map)))


    def get_line_rhyme(self, line):
        """ we use statistics of ngram to disambiguate the rhyme category,
        but there is still risk of mismatching and ambiguity
        我们使用ngram的统计数据来消除押韵类别的歧义，但仍然存在不匹配和歧义的风险
        """
        tail_char = line[-1]

        if tail_char in self.__char_rhyme_map:
            bigram = line[-2] + line[-1]
            if bigram in self.__ngram_rhyme_map: #看2 gram和3 gram
                return self.__ngram_rhyme_map[bigram]

            trigram = line[-3] + line[-2] + line[-1]
            if trigram in self.__ngram_rhyme_map:
                return self.__ngram_rhyme_map[trigram]

            return self.__char_rhyme_map[tail_char]#如果不行，再返回1 gram

        if not tail_char in self.__vocab:
            return -1
        else:
            tail_id = self.__vocab[tail_char]#上面没找着就把char转成id继续找

        if tail_id in self.__rhyme_dic:#####这个rhyme_dic和char_rhyme_map有什么区别，后面可以研究一下
            return self.__rhyme_dic[tail_id][0]

        return -1

    # ------------------------------ 基本是set函数
    def reset(self, length, verbose):
        assert length == 5 or length == 7
        self.__length = length
        self.__rhyme = -1
        self.__rhythm_vec = []
        self.__repetitive_ids = []
        self.__verbose = verbose


    def set_pattern(self, line):
        # set a rhythm pattern in terms of the first generated line  根据生成的第一行设置节奏模式
        assert len(line) == 5 or len(line) == 7
        rhythm_l1 = self.__rhythm_tool.get_rhythm(line)

        if self.__verbose >= 2:
            print ("set rhythm_id of l1: %d" % (rhythm_l1))

        # when the first line doesn't match any pattern,
        #   randomly select one  随机挑选一个
        if rhythm_l1 < 0:
            rhythm_l1 = random.sample([0,1,2,3], 1)[0]
            if self.__verbose >= 2:
                print ("sample rhythm_id of l1: %d" % (rhythm_l1))

        # pattern id of each line
        self.__rhythm_vec = self.__RHYTHM_TYPES[rhythm_l1]
        if self.__verbose >= 2:
            rhythm_str = " ".join([str(r) for r in self.__rhythm_vec])
            print ("set rhythm ids of all lines: %s" % (rhythm_str))

        # set rhyme in terms of the first line 韵脚
        self.set_rhyme(line)


    def set_rhyme(self, line):
        rhyme = self.get_line_rhyme(line)
        if isinstance(rhyme, list) or isinstance(rhyme, tuple):
            rhyme = int(rhyme[0])
        else:
            rhyme = int(rhyme)
        if self.__verbose >= 2:
            print ("set rhyme id: %s" % (rhyme))
        self.__rhyme = rhyme


    def add_repetitive(self, ids):#已经出现过的id，避免重复
        self.__repetitive_ids = list(set(ids+self.__repetitive_ids))


    # ------------------------------- 底下是get函数
    def get_pattern(self, step, pure=False):
        # before the first line is generated, rerutn
        #   empty patterns    返回一个pattern模式，在第一句生成之前，返回一个空模式
        if len(self.__rhythm_vec) == 0:
            return -1, [], -1

        # return the pattern of the current line and the rhyme  返回当前行的模式和韵律，注意下面的step
        l_rhythm = self.__rhythm_vec[step]
        l_rhythm_pattern \
            = self.__RHYTHM_PATTERNS[self.__length][l_rhythm]

        # for Chinese classical quatrains, the third line doesn't rhyme第三局不用压韵
        rhyme = -1 if step == 2 else self.__rhyme

        #print (step, l_rhythm, rhyme)
        #print (type(step), type(l_rhythm), type(rhyme))

        if self.__verbose >= 2 and not pure:
            print ("step: %d, line rhythm id: %d, rhyme: %d" %
                (step, l_rhythm, rhyme))
        #我感觉l_rhythm表示本句的平仄代号，l_rhythm_pattern表示本句的平仄具体格式，rhyme表示押韵代号
        return l_rhythm, l_rhythm_pattern, rhyme  #不太清楚为什么要返回这些东西？答：应该是获取本句对应的韵律模式，下面调用到了get_pattern函数

    def get_rhyme(self):
        return self.__rhyme

    def get_level_cids(self):
        return copy.deepcopy(self.__level_list)

    def get_oblique_cids(self):
        return copy.deepcopy(self.__oblique_list)

    def get_rhyme_cids(self, rhyme_id):
        if rhyme_id not in self.__rhyme_idic:
            return []
        else:
            return copy.deepcopy(self.__rhyme_idic[rhyme_id])

    def get_repetitive_ids(self):
        return copy.deepcopy(self.__repetitive_ids)


    def filter_illformed(self, lines, costs, states, step):#对病态的句子过滤，只剩下好句子
        if len(lines) == 0:  #这个lines是原始句子列表
            return [], [], []

        new_lines = []
        new_costs = []
        new_states = []

        required_l_rhythm, _, _ = self.get_pattern(step, True)

        len_error = 0
        lib_count = 0
        rhythm_error = 0
        rhythm_mismatch = 0

        for i in range(len(lines)):
            #print (lines[i])
            if len(lines[i]) < self.__length:
                len_error += 1
                continue
            line = lines[i][0:self.__length]

            # we filter out the lines that already exist in the
            #   training set, to guarantee the novelty of generated poems  如果有问题就continue，跳过后面的append存储环节
            if line in self.__line_lib:
                lib_count += 1
                continue

            rhythm_id = self.__rhythm_tool.get_rhythm(line)

            if rhythm_id < 0:
                rhythm_error += 1
                continue

            if required_l_rhythm != -1 and rhythm_id != required_l_rhythm:
                rhythm_mismatch += 1
                continue

            new_lines.append(line)
            new_costs.append(costs[i])
            new_states.append(states[i])

        if self.__verbose >= 3:
            print ("input lines: %d, ilter out %d illformed lines, %d remain"  
                % (len(lines), len(lines)-len(new_lines), len(new_lines)))  #本来有多少行，删掉多少行有毛病的，还剩下多少行
            print ("%d len error, %d exist in lib, %d rhythm error, %d rhythm mismatch"
                % (len_error, lib_count, rhythm_error, rhythm_mismatch))  #这些都是有问题的数量

        return new_lines, new_costs, new_states
