NLP 之 新闻分类 - 基于深度学习的文本分类 1 
![1](https://user-images.githubusercontent.com/61811515/87897201-8f465b00-ca18-11ea-9fe4-542b80409ed5.jpeg)

在建立好我们的base line model 之后，我们这一次主要研究的是利用深度学习对文本进行分类。 这次我们主要学习的是FastText。 

通过上次sklearn的应用，我们可以发现转换后的向量维度很高，需要长时间的训练。 而且它没有考虑单词与单词之间的关系，单纯对词进行了统计。 深度学习可以用于文本表示，也可以将其映射到一个低纬空间。 例如FastText，Word2Vec 和Bert 


## FastText 
FastText  是一个典型的深度学习词向量的一种表现形式，支持多label的分类学习。 主要优势在于它词内的n-gram 信息(subwords n-gram)去解决词性变化（morphology）和 softmax 的trick。 

因为fasttext 使用了Hierarchical softmax ，首先我们介绍一下什么是hierarchical softmax？
Hierarchical softmax 是softmax function的衍生。 softmax function 是一个normalized exponential function。在有k的class 的时候将结果normalize使得值落于0到1 之间。公式如下：

![image](https://user-images.githubusercontent.com/61811515/88752932-066ea400-d129-11ea-8904-f935e24bc119.png)

因为softmax的分母计算用的是所有在word vector 里面的数字，当words 增加后，它就会computational heavy。 Hierarchical softmax 就是为解决这个问题作出的优化。 

我们可以想象softmax 是个tree structure 。


![image](https://user-images.githubusercontent.com/61811515/88754005-97df1580-d12b-11ea-9568-fced4e41a943.png)

root 是context C，leaves 则是所有的单词的概率，要算出一个单词的概率，你需要知道所有leaves 的概率。 

然而与它不同的是hierarchical softmax 将其变成了multi-layer tree。 每个edge 对应了概率，每个词的概率则是所有edge 上的乘积【1】这样就加快了概率的计算。 
接下来我们对fasttext 进行了解。 


![image](https://user-images.githubusercontent.com/61811515/88749677-cd7f0100-d121-11ea-8e9c-fdc2b28e4dd1.png)
（image from original paper) 

FastText model 采用的是只有一层的简单的neural network。 首先FastText 采用了n-gram的方式将我们每一个words 变成n-gram 。 之所以这么做是因为想要保留内部形态信息。例如 3-gram ，‘Test’ 我们有 '<Te','Tes','est','st>' ， < and > 作为开始和结束的字符。 原本完整的word则由所有n-gram 的向量求和得到。
这样带来的好处是低频词生成的word vector 会有更好效果，因为它们的n-gram 可以和其他词共享。 而且也可以对训练集以外的单词构建word vector。  我们的hidden layer 得到的值是所有n-gram 的word vector 的叠加平均。 最后采用了Hierarchical softmax输出class number的可能性。 


代码实现
首先我们要安装fasttext。 目前fasttext只支持python 3.7及以前。接下来在conda prompt 里run 接下来的代码
```
conda install -c mbednarski fasttext
```
```
import pandas as pd
from sklearn.metrics import f1_score
# 转换为FastText需要的格式
train_df = pd.read_csv('train_set.csv', sep='\t',)
train_df['label_ft'] = '__label__' + train_df['label'].astype(str) train_df[['text','label_ft']].iloc[:-5000].to_csv('train.csv', index=None, header=None, sep='\t')
import fastText
# parameter: learning rate, ngram, verbosity level,verbosity level (optional) ,minCount (minimal number of word occurrences),
#  epoch number of epochs, loss(option: ns- skipgram, hs - skipgram hierarchical softmax, softmax)
model = fasttext.train_supervised('train.csv', lr=1.0, wordNgrams=2,
                                          verbose=2, minCount=1, epoch=25, loss="hs")
val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']] print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))
#0.91
```



# 参考文献
【1】Hierarchical softmax：https://www.quora.com/What-is-hierarchical-softmax
【2】Word2Vec 解释：https://www.cnblogs.com/pinard/p/7160330.html
