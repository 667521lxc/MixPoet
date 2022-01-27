# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-30 19:59:47
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.    这个文件主要是在调用那个generator函数，封装在generator上层，作为生成主函数，逻辑简单
'''
from generator import Generator
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="The parametrs for the generator.")
    parser.add_argument("-m", "--mode", type=str, choices=['interact', 'file'], default='file',
        help='The mode of generation. interact: generate in a interactive mode.\
        file: take an input file and generate poems for each input in the file.')  #互动模式和文件模式
    parser.add_argument("-b", "--bsize",  type=int, default=20, help="beam size, 20 by default.")
    parser.add_argument("-v", "--verbose", type=int, default=0, choices=[0, 1, 2, 3],
        help="Show other information during the generation, False by default.")
    parser.add_argument("-s", "--select", type=int, default=0,
        help="If manually select each generated line from beam candidates? False by default.\
        It works only in the interact mode.")  #可以从生成的诗句中选择，
    parser.add_argument("-l", "--length", type=int, choices=[5, 7],
        help="The length of lines of generated quatrains. 5 or 7.\
        It works only in the file mode.")   #五绝七绝
    parser.add_argument("-i", "--inp", type=str,
        help="input file path. it works only in the file mode.")  #输入文件
    parser.add_argument("-o", "--out", type=str, 
        help="output file path. it works only in the file mode")  #输出文件
    return parser.parse_args()


def generate_manu(args):  #调用generator里面的函数，人工输入条件
    generator = Generator()
    beam_size = args.bsize
    verbose = args.verbose
    manu = True if args.select ==1 else False

    while True:
        #下面要先进行一些输入
        keyword = input("input a keyword:>")   #关键词
        length = int(input("specify the length, 5 or 7:>"))  #五绝七绝
        label1 = int(input("specify the living experience label\n\
            0: military career, 1: countryside life, 2: other:, -1: not specified>"))  #生活经历因素
        label2 = int(input("specify the historical background label\n\
            0: prosperous times, 1: troubled times, -1: not specified>"))  #历史背景因素

        lines, info = generator.generate_one(keyword, length, label1, label2,
            beam_size, verbose, manu)  #直接生成###########################

        if len(lines) != 4:
            print("generation failed!")
            continue
        else:
            print("\n".join(lines))


def generate_file(args): #从文件读取关键词生成
    generator = Generator()
    beam_size = args.bsize
    verbose = args.verbose
    manu = True if args.select ==1 else False

    assert args.inp is not None
    assert args.out is not None
    assert args.length is not None

    length = args.length

    with open(args.inp, 'r') as fin:
        inps = fin.readlines()

    poems = []
    N = len(inps)
    log_step = max(int(N/100), 2)
    for i, inp in enumerate(inps):
        para = inp.strip().split(" ")
        keyword = para[0]  #每一行的第一部分是关键词

        if len(para) >= 2:
            length = int(para[1])
        if len(para) >= 3:
            label1 = int(para[2])
        else:
            label1 = -1

        if len(para) >= 4:
            label2 = int(para[3])
        else:
            label2 = -1
        #以上从输入的一行中获取了keyword，length，label1，label2信息
        lines, info = generator.generate_one(keyword, length,
            label1, label2, beam_size, verbose, manu)

        if len(lines) != 4:
            ans = info
        else:
            ans = "|".join(lines)

        poems.append(ans)

        if i % log_step == 0:
            print ("generating, %d/%d" % (i, N))

    with open(args.out, 'w') as fout:
        for poem in poems:
            fout.write(poem+"\n")


def main():
    args = parse_args()
    if args.mode == 'interact':
        generate_manu(args)
    else:
        generate_file(args)


if __name__ == "__main__":
    main()
