import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import re
import jieba
import json
from gensim.models import Word2Vec

class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

class Sentence:
    def __init__(self, words):
        self.words = words

    def len(self) -> int:
        return len(self.words)

def cut(string):
    return list(jieba.cut(string, cut_all=False))
    
def get_word_frequency(word_text, looktable):
    if word_text in looktable:
        return looktable[word_text]
    else:
        return 1.0

def sentence_to_vec(sentence: Sentence, embedding_size, looktable, a=1e-3):
    sentence_set = []
    vs = np.zeros(embedding_size)
    sentence_length = sentence.len()
    for word in sentence.words:
        a_value = a / (a + get_word_frequency(word.text, looktable))
        vs = np.add(vs, np.multiply(a_value, word.vector))

    vs = np.divide(vs, sentence_length)
    sentence_set.append(vs)

    pca = PCA()
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]
    u = np.multiply(u, np.transpose(u))

    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)

    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub).tolist())

    return sentence_vecs

def handle(model_path, csv_path, dic_path):
    model = Word2Vec.load(model_path).wv
    csv_data = pd.read_csv(csv_path, encoding='gb18030')
    content = csv_data['content']
    title = csv_data['title']
    dic_file = open(dic_path, 'r', encoding='utf-8')
    dic = json.loads(dic_file.read())
    dic_file.close()
    embedding_size = 50
    sentence_vectors = []
    title_vectors = []
    paragraph_vectors = []
    c_len = len(content)
    for i in range(c_len):
        c = content[i]
        allsent = []
        s_vectors = []
        for s in str(c).split('。'):
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
        if len(allsent) > 0:
            paragraph_vectors.append(sentence_to_vec(Sentence(allsent), embedding_size, looktable=dic))
            sentence_vectors.append(s_vectors)
            
            s = str(title[i]).strip()
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
            title_vectors.append(sentence_to_vec(Sentence(s1), embedding_size, looktable=dic))
    res = {}
    res['title_vectors'] = title_vectors
    res['sentence_vectors'] = sentence_vectors
    res['paragraph_vectors'] = paragraph_vectors
    file = open('output/vectors.json', 'w', encoding='utf-8')
    json.dump(res, file)
    file.close()

#handle("output/data_model", 'input/sqlResult_1558435.csv', 'output/wiki.json')