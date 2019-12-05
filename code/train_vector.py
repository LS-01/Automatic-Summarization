import re
import json
import numpy as np
from gensim.models import Word2Vec
from handle_sentence import cut
from handle_sentence import Word
from handle_sentence import Sentence
from handle_sentence import sentence_to_vec

def getRelativity(vs, vt, vc):
    vv = (vt + vc).reshape(50)
    return np.dot(vs, vv) / (np.linalg.norm(vs) * (np.linalg.norm(vv)))

def summarize(content, title):
    embedding_size = 50
    sentence_vectors = []
    title_vector = None
    paragraph_vector = None
    allsent = []
    s_vectors = []
    sentences = []
    for s in str(content).split('。'):
        s = s.strip()
        s = re.sub(re.compile('(\n)*(\u3000)*(↑)*(▲)*\?(\?)*'), '', s)
        if s == "":
            continue
        s1 = []
        for word in cut(s):
            try:
                vec = model[word]
            except KeyError:
                vec = np.zeros(embedding_size)
            s1.append(Word(word, vec))
            allsent.append(Word(word, vec))
        s_vectors.append(sentence_to_vec(Sentence(s1), embedding_size, looktable=dic))
        sentences.append(s + '。')
    if len(allsent) > 0:
        paragraph_vector = sentence_to_vec(Sentence(allsent), embedding_size, looktable=dic)
        sentence_vectors.append(s_vectors)
        
        s = re.sub(re.compile('(\n)*(\u3000)*(↑)*(▲)*\?(\?)*'), '', str(title).strip())
        s1 = []
        for word in cut(s):
            try:
                vec = model[word]
            except KeyError:
                vec = np.zeros(embedding_size)
        s1.append(Word(word, vec))
        title_vector = sentence_to_vec(Sentence(s1), embedding_size, looktable=dic)
    relativity = getRelativity(np.array(sentence_vectors), np.array(title_vector), np.array(paragraph_vector))
    dt = np.dtype([('seq',  int),('relativity',  float)])
    li = []
    for i in range(len(sentences)):
        li.append((i, relativity[0][i][0]))
    relativity_sorted = np.sort(np.array(li, dtype = dt), order =  'relativity')
    i = len(relativity_sorted) - 1
    min_v = len(relativity_sorted) * 2 / 3
    result_show = ''
    while i >= min_v:
        result_show += sentences[relativity_sorted[i]['seq']]
        i -= 1
    return result_show

model = Word2Vec.load("output/data_model").wv
dic_file = open('output/wiki.json', 'r', encoding='utf-8')
dic = json.loads(dic_file.read())
dic_file.close()
#show = summarize('此外，自本周（6月12日）起，除小米手机6等15款机型外，其余机型已暂停更新发布（含开发版/体验版内测，稳定版暂不受影响），以确保工程师可以集中全部精力进行系统优化工作。有人猜测这也是将精力主要用到MIUI 9的研发之中。MIUI 8去年5月发布，距今已有一年有余，也是时候更新换代了。当然，关于MIUI 9的确切信息，我们还是等待官方消息。', '小米MIUI 9首批机型曝光：共计15款')
#print(show)