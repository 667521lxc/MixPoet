借鉴THUAI的论文  Mixpoet模型，后续会进行修改



执行preprocess文件

(base) PS D:\MixPoet-master\preprocess> python .\preprocess.py
creating the word dictionary...
input poems: 109727
original word num:7020
min freq:1
total word num: 7024
saving dictionary to D:/MixPoet-master/corpus/vocab.pickle
saving inverting dictionary to D:/MixPoet-master/corpus/ivocab.pickle
output data to D:/MixPoet-master/data/training_lines.txt, num: 438908
data num: 482081, skip plen: 0, skip llen: 0, max length: 7
data num: 34210, skip plen: 0, skip llen: 0, max length: 7
training data: 482081
validation data: 34210
saving training data to D:/MixPoet-master/corpus/semi_train.pickle
saving validation data to D:/MixPoet-master/corpus/semi_valid.pickle
output data to D:/MixPoet-master/data/test_inps.txt, num: 9976
output data to D:/MixPoet-master/data/test_trgs.txt, num: 9976
