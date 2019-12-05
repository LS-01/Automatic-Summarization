from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def analogy(model, x1, x2, y1):
    result = model.wv.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]

def tsne_plot(model):
    labels = []
    tokens = []
    
    i = 0
    for word in model.wv.vocab:
        tokens.append(model.wv[word])
        labels.append(word)
        i += 1
        if i > 500:
            break
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

def show_test(file):
    model = Word2Vec.load(file)
    print(model.wv.most_similar('勇敢'))
    print(model.wv.most_similar('美女'))
    print(analogy(model, '中国', '汉语', '美国'))
    print(analogy(model, '美国', '奥巴马', '美国'))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    tsne_plot(model)

show_test("output/data_model")