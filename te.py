import jieba
import os
import codecs
from gensim import corpora,models,similarities

from collections import defaultdict

read_dic = corpora.Dictionary.load('corpus.dic')
read_s = corpora.Dictionary.load('dict.txt')
print(read_dic)