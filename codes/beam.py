# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-19 16:36:04
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
import numpy as np
import random
import torch
import copy
from config import device

class Hypothesis(object):
    '''保存生成的标记、当前状态和beam分数的假设，不明白这三个东西分别代表什么？？
    a hypothesis which holds the generated tokens,
        current state and beam score
    '''
    def __init__(self, tokens, states, score):
        self.score = score
        self.states = states
        self.candidate = copy.deepcopy(tokens)

class PoetryBeam(object):
    def __init__(self, beam_size, length, B_ID, E_ID, UNK_ID,
         level_char_ids, oblique_char_ids):
        """Initialize params."""
        self.__beam_size = beam_size
        self.__length = length

        self.__B_ID = B_ID
        self.__E_ID = E_ID
        self.__UNK_ID = UNK_ID

        self.__level_cids = level_char_ids
        self.__oblique_cids = oblique_char_ids #间接的


    def reset(self, init_state, rhythms, rhyme, rhyme_char_ids, repetitive_ids):
        # reset before generating each line 重置
        self.__hypotheses \
            = [Hypothesis([self.__B_ID], [init_state.clone().detach()], 0.0)
            for _ in range(0, self.__beam_size)]
        self.__completed_hypotheses = []

        self.__rhythms = rhythms # rhythm pattern of each chars in a line
        self.__rhyme = rhyme
        self.__rhyme_cids = rhyme_char_ids # char ids in the required rhyme category
        self.__repetitive_ids = repetitive_ids


    def get_candidates(self, completed=False, with_states=False): #处理Hypothesis类，获取候选字集
        if completed:
            hypotheses = self.__completed_hypotheses
        else:
            hypotheses = self.__hypotheses

        candidates = [hypo.candidate for hypo in hypotheses]  #候选
        scores = [hypo.score for hypo in hypotheses]

        if with_states:
            # (L, H) * B
            all_states = [hypo.states for hypo in hypotheses]
            return candidates, scores, all_states
        else:
            return candidates, scores


    def get_search_results(self, only_completed=True, sort=True):
        candidates, scores, states = self.get_candidates(True, True)  #调用上面的get_candidates函数

        if not only_completed:  #增加什么意思？？？
            add_candis, add_scores, add_states = self.get_candidates(True, True)
            candidates = candidates + add_candis
            scores = scores + add_scores
            states = states + add_states

        scores = [score/(len(candi)-1) for score, candi in zip(scores, candidates)]
        # sort with costs 按照得分进行排序
        if sort:
            sort_indices = list(np.argsort(scores))
            candidates = [candidates[i] for i in sort_indices]
            scores = [scores[i] for i in sort_indices]
            states = [states[i] for i in sort_indices]

        return candidates, scores, states


    def get_beam_tails(self):
        # get the last token and state of each hypothesis  获取每个假设的最后标记和状态
        # #获取最大概率的那个词?，即：实现beam search过程
        tokens = [hypo.candidate[-1] for hypo in self.__hypotheses]
        # [B, 1]
        tail_tokens \
            = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)

        tail_states = [hypo.states[-1] for hypo in self.__hypotheses]
        # [1, H] * B -> [B, H]
        tail_states = torch.cat(tail_states, dim=0)

        return tail_tokens, tail_states


    def uncompleted_num(self):
        return len(self.__hypotheses)


    def advance(self, logits, states, position):#这应该是beam search一步步推进的过程
        # outs: (B, V)
        # states: (B, H)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().data.numpy()#对logits进行softmax

        beam_ids, word_ids, scores = self.__beam_select(log_probs, position)#调用下面的函数

        # update beams 下面进行了更新操作
        updated_hypotheses = []
        for beam_id, word_id, score in zip(beam_ids, word_ids, scores):
            #print (beam_idx, word_idx, score)
            state = states[beam_id, :].unsqueeze(0) # (1, H)
            new_states = self.__hypotheses[beam_id].states + [state]  #新的链接在旧的上

            new_candidate = self.__hypotheses[beam_id].candidate + [word_id]

            hypo = Hypothesis(new_candidate, new_states, score)

            if word_id == self.__E_ID:
                self.__completed_hypotheses.append(hypo)
            else:
                updated_hypotheses.append(hypo)

        self.__hypotheses = updated_hypotheses


    def __beam_select(self, log_probs, position): #在上面被调用，position是表示一行中的位置
        # log_probs: (B, V)
        B, V = log_probs.shape[0], log_probs.shape[1]

        current_scores = [hypo.score for hypo in self.__hypotheses]
        current_scores = np.reshape(current_scores, (B, 1))

        if position == 0: #？？？？？？cost是什么
            costs = - log_probs[0, :].reshape(1, V) # (1, V)
        else:
            costs = current_scores - log_probs # (B, V)

        # filter with rhythm, rhyme and length
        filter_v = 1e5

        costs[:, self.__UNK_ID] = filter_v

        # filter eos symbol  有什么用啊？？？
        if position < self.__length:
            costs[:, self.__E_ID] = filter_v

        # restrain the model from generating chars
        #   that already generated in previous lines限制模型生成前几行中已经生成的字符
        costs[:, self.__repetitive_ids] = filter_v

        # restrain in-line repetitive chars 抑制在线重复字符
        inline_filter_ids = self.inline_filter(position)
        for i in range(0, costs.shape[0]):
            costs[i, inline_filter_ids[i]] = filter_v
        #有点纳闷是怎么抑制的？有个规律，好像是只要想过滤东西，就给cost矩阵赋值filter_v

        # for the tail char, filter out non-rhyme chars  对于尾字符，过滤掉非押韵字符
        if (self.__rhyme != -1) and (position == self.__length-1):
            filter_ids = list(set(range(0, V)) - set(self.__rhyme_cids)) #把不押韵的删掉
            costs[:, filter_ids] = filter_v

        '''
        filter out chars of the undesired tone过滤掉不希望听到的音调中的字符
        NOTE: since some Chinese characters may belong to both tones,
            here we only consider the non-overlap ones由于有些汉字可能属于两种声调，这里我们只考虑不重叠的。
        TODO: disambiguation消歧
        '''
        if position < self.__length and len(self.__rhythms) > 0:
            pos_rhythm = self.__rhythms[position]  #获取当前位置的韵律？？
            if pos_rhythm == 0:  # level tone平声
                costs[:, self.__oblique_cids] = filter_v
            elif pos_rhythm == 1:  # oblique仄声
                costs[:, self.__level_cids] = filter_v

        flat_costs = costs.flatten() # (B*V)  返回一个一维数组

        # idx of the smallest B elements 取cost最小的B个元素的下标  https://blog.csdn.net/SanyHo/article/details/105455175
        best_indices = np.argpartition(
            flat_costs, B)[0:B].copy()  #类似于快排划分，将传入的数组flat_costs分成两部分，即：排在第B位置前面的数都小于B，排在第B位置后面的值都大于B。

        scores = flat_costs[best_indices]

        # get beam id and word id
        beam_ids = [int(idx //  V) for idx in best_indices]
        word_ids = [int(idx % V) for idx in best_indices]

        if position == 0:
            beam_ids = list(range(0, B))

        return beam_ids, word_ids, scores


    def inline_filter(self, pos): #大致猜出来是在整理禁用词表，但为什么这么做没太搞明白
        candidates, _ = self.get_candidates()
        # candidates: (L_i) * B
        B = len(candidates)
        forbidden_list = [[] for _ in range(0, B)]

        limit_pos = pos - 1 if pos % 2 != 0 else pos  #取一个偶数，为什么？？？
        preidx = range(0, limit_pos)

        for i in range(0, B):  # iter ever batch   遍历每一个batch
            forbidden_list[i] = [candidates[i][c] for c in preidx]

        return forbidden_list
