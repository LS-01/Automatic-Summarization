import os
import re
import jieba
import collections
import json
from zhtools.langconv import Converter

def cut(string):
    return list(jieba.cut(string, cut_all=False))

def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext

def sentence_to_words_saved(sen):
    words = []
    for s in sen:
        s = s.strip()
        if s == "":
            continue
        for w in cut(cht_to_chs(re.sub(r'[^\w\s]','',cleanhtml(s)))):
            words.append(w)
    return words

def handle_wiki_files(filepath):
    dic = collections.Counter([])
    print('----------handle wiki start----------')
    for root, dirs, files in os.walk(filepath):
        print('----------handle wiki file ' + root + '----------')
        for f in files:
            file = open(os.path.join(root, f), 'r', encoding='utf-8')
            ft = file.read()
            file.close()
            dic += collections.Counter(sentence_to_words_saved(str(ft).split('。')))
    print('----------handle wiki end----------')
    return dic

def handle(wiki_file, target_file):
    dic = dict(handle_wiki_files(wiki_file))
    file = open(target_file, 'w', encoding='utf-8')
    json.dump(dic, file)
    file.close()

handle('input/articles/', 'output/wiki.json')