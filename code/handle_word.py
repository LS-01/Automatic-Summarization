import re
import os
import jieba
import pandas as pd
from gensim.models import word2vec
from zhtools.langconv import Converter

def cut(string):
    return list(jieba.cut(string, cut_all=False))

def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line

def cleanhtml(raw_html):
    cleanr = re.compile('(\n)*(\u3000)*(↑)*(▲)*<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext

def sentence_to_words_saved(sen, target_file):
    for s in sen:
        s = s.strip()
        if s == "":
            continue
        s = " ".join(cut(cht_to_chs(re.sub(r'[^\w\s]','',cleanhtml(s)))))
        target_file.write(s + "\n")
        target_file.flush()

def handle_wiki(filepath, target_file):
    print('----------handle wiki start----------')
    for root, dirs, files in os.walk(filepath):
        print('----------handle wiki file ' + root + '----------')
        for f in files:
            file = open(os.path.join(root, f), 'r', encoding='utf-8')
            ft = file.read()
            file.close()
            sentence_to_words_saved(str(ft).split('。'), target_file)
    print('----------handle wiki end----------')
            
def handle_csv(filepath, target_file):
    print('----------handle csv start----------')
    csv_data = pd.read_csv(filepath, encoding='gb18030')
    content = csv_data['content']
    for c in content:
        sentence_to_words_saved(str(c).split('。'), target_file)
    print('----------handle csv end----------')

def handle(wiki_file, csv_file, target_file):
    file = open(target_file, 'w', encoding='utf-8')
    handle_wiki(wiki_file, file)
    handle_csv(csv_file, file)
    file.close()
    print('----------LineSentence----------')
    sentences = word2vec.LineSentence(target_file)
    print('----------Word2Vec----------')
    model = word2vec.Word2Vec(sentences,min_count=1,size=50)
    print('----------save----------')
    model.save(target_file + '_model')

handle('input/articles/', 'input/sqlResult_1558435.csv', 'output/data')