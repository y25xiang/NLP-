# NLP 之 新闻分类 - 3 TF-IDF & Machine Learning
![1](https://user-images.githubusercontent.com/61811515/87897201-8f465b00-ca18-11ea-9fe4-542b80409ed5.jpeg)
继上次读取完数据，对数据进行基本的数据分析后，这一次的学习我们开始初步建立一个模型作为我们的baseline。

## 背景知识 
### 分词（Tokenization） 
分词（Tokenization） 是NLP 中一个重要步骤。 分词就是将句子、段落、文章分解成字词为单位的数据结构，方便之后进行后续的处理分析工作。 

**为什么需要分词（phrase）**
* 机器学习之所以能解决很多复杂的问题，是将这些问题都转化成为了数学问题。 NLP 也是同样的一个思路。 通常我们的文本都是非结构化数据。 将非结构化数据转化为结构化数据，我们就可以从结构化数据转化为一个数学问题。 分词是这个转化过程的第一步。 
* 词是一个合适的粒度(Granularity) 。因为一个字有时候无法表达出完整的意思。 举个例子，比如只有一个鼠字，在老鼠和鼠标中，表示的是完全两种东西。 而句子会包含过多的信息。再加上基础的一些模型对于长距离的信息建模能力较弱。词成为了最为合适的粒度。 

**中英文分词区别**
* 分词方式不同
英文每个单词之间都会有空格但中文没有。这也使给中文分词造成了难度。 一句中文也更容易出现歧义的情况。 例如这个苹果不大好吃。 可以理解为 这个苹果不大/好吃。 也可以理解为这个苹果/不大好吃。 这也给分词带来了难度。 
* 英文单词有多种形态。 中文则不需要
因为有过去时，现在时等不同的形态以及单复数的转化，因为会需要词形还原（Lemmatization） 和词干提取（Stemming）。 

**分词基本方法**
* 词典匹配
 这个方法主要是将中文文本按照词典中最长的词进行分割，然后与词典中的词语进行匹配，如匹配失败则调整后再进行匹配，如此循环。利用这个想法的主要有三类。
  * 正向最大匹配(Forward Matching Method)
   每次匹配不成功将最后一个字去掉。 
  * 逆向最大分配(Backward Matching Method) 
  每次匹配不成功将第一个字去掉
  * 双向最大匹配法(Bi-direction Matching Method) 
 将正向和逆向匹配的结果进行对比。据SunM.S. 和 Benjamin K.T.（1995）研究表明，中文中大概90.0％的句子，正向最大匹配法和逆向最大匹配法完全重合且正确，只有大概9.0％的句子两种切分方法得到的结果不一样，但其中必有一个是正确的（歧义检测成功），只有不到1.0％的句子，或者正向最大匹配法和逆向最大匹配法的切分虽重合却是错的，或者正向最大匹配法和逆向最大匹配法切分不同但两个都不对（歧义检测失败）。【1】

  这个分词方法速度快成本低，但是适应性较弱，对不同领域的文本结果相差较大。 

* 统计的分词方法
这个方法考虑词语出现的频率，也会考虑上下文。 具备比较好的学习能力，适应性较强但速度和成本都会比词类匹配都高。 常见的方法有 SVM（support vector machine），CRF（conditional random field）等。 

* 深度学习
这个方法准确率高适应性强，但成本高，速度慢。 

通常常见的分词器（word segmentation machine） 利用了机器学习和词典相结合，既提高精准率，又提高适应度。 

常见分词器
英文：[Keras](https://github.com/keras-team/keras), [Gensim](https://github.com/RaRe-Technologies/gensim), [NLTK](https://github.com/nltk/nltk)
中文：[Hanlp](https://github.com/hankcs/HanLP), [Stanford 分词](https://github.com/stanfordnlp/CoreNLP)

下面我们要开始进行初步的模型学习。 
## 文本表示方法
通常在机器学习训练的过程，若我们有N个样本，每个样本有M个特征，则我们会得到一个NxM 的矩阵。 但文本处于非结构数据，没有固定长度。 因此我们需要使用词嵌入（Word Embedding） 将长度不定的文本转化成为定长的空间。 
 
* One-Hot 

 One-Hot 对每个单词使用了离散的向量，将每个字/词对应了一个索引，然后根据索引进行赋值。 
 例如：{ 机：1， 器：2，学：3，习：4 ，有：5，趣：6}，分别对应了6维度稀疏向量：
 机：[1,0,0,0,0,0]
 器：[0,1,0,0,0,0]
 学：[0,0,1,0,0,0]
 习：[0,0,0,1,0,0]

* Bag of Words

 Bag of Words 也称Count Vectors，每个文档的字/词用出现次数表示。 
 例如： 机器学习 对应 [1,1,1,1,0,0]
 可通过sklearn中的CountVectorizer来实现。 下面是代码实例:
 ```
 from sklearn.feature_extraction.text import CountVectorizer
 corpus = ['This is the first document.',
 'This document is the second document.', 'And this is the third one.',
 'Is this the first document?']
 vectorizer = CountVectorizer()
 vectorizer.fit_transform(corpus).toarray()
 ```

* N-gram 

 N-gram 与bag of words 类似，但是它是取相邻的N个字符组合成一个新的单词并进行计数。 
 例子： 在N为2 的情况下，机器学习 对应 机器 器学 学习 再对其进行计数

* TF-IDF

 TF-IDF 有两部分组成： 词语频率（Term Frequency) 和逆文档频率（Inverse Document Frequency）
 **词语频率**
 下面是词语频率的计算方法： 
 Term Frequency = 该词语在当前文档出现的次数/当前文档词语出现的总数
 **Motivation of term frequency 为什么用词语频率 **
 是因为我们在检索文章的时候我们会搜索相关词语，检索结果会根据你给出的相关词语对网上所有的文章内容进行相关排序，再将最相关的搜索结果反馈给用户。 一篇文章一个词语在整篇文章中的出现频率可以表现出这篇文章对于这个词语的相关程度。 
 **逆文档频率**
 下面是逆文档频率计算方法：
 Inverse Document Frequency = log_e(文档总数/出现该词语的文档总数） 
 **Motivation behind Inverse Document Frequency 为什么使用逆文档频率**
 有一些经常出现为词对整体意义贡献不大但出现频繁，例如中文中的的， 英文中的the。这一类词TF的数值会很高，但很多有意义的词会被忽略。因为逆文档频率缩小了经常出现在文档的词的比重，提高了相对偏少的词的比重。

 TF-IDF 则为两个频率的乘积 ie.TF-IDF = TF * IDF

## Model 1: TF-IDF + Ridge Classifier
### Ridge Classifier
**Ridge Classifier 原理**
Ridge Regression 和linear regression相似。通常linear model的公式为 y= WX + b，W 和X 为向量，b为标量。 
X 通常是作为input，在这个里面我们目标是找出weight W 和b 以此来达到我们所计算出来的y的值与真实值相近。 对应的cost function 我们采用的是square error总和。假设我们有M 个例子，p个feature用式子表示为


![image](https://user-images.githubusercontent.com/61811515/88500230-1609b380-cf96-11ea-9a41-ddff31220b0f.png)

而Ridge Regression为了防止overfitting 的问题， 加入了一个L2 Norm 的式子。式子如下：

 
![image](https://user-images.githubusercontent.com/61811515/88502086-72230680-cf9b-11ea-9fa7-9d751e713bbe.png)

为什么Weight 过大会有overfit 的问题呢？  因为weight 过大，让我们最后的抛物线变得更加steep。而steep curve 会在我们训练模型的过程中overfit 每个训练点以此将cost 降到最低。 
为什么加入一个L2 Norm 可以防止overfitting 呢？ 因为当我们的weight 过大时，会对我们的cost 会增加而我们希望的cost 越小越好，所以这样可以有效的防止。 
Notes: Lasso Regression 采用了L1 norm 的penalty。 比起Ridge Regression，它更加sparse， 将一些contribute 少的feature coefficient 变成0，而不是像Ridge Regression一样将coefficient 接近于0。 因为Lasso Regression 在优化时，像一个倒立的金字塔，从任意一点开始，先移动到四个角中任意一角，再向中心点移动。 


![image](https://user-images.githubusercontent.com/61811515/88503148-da271c00-cf9e-11ea-8363-d09ce71b70bb.png)

根据上图的粗略理解为什么Lasso增加sparsity，椭圆代表我们的mean square error，将L2 Norm 和L1 Norm 作为我们的extra constraint，不断扩大，圆形最有可能接触的是边上的一点。 而正方形不断扩大，则最先接触到的点通常是四个角。 【2】

Code Section
**One Hot + Ridge Regression 代码实现**
```
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
# Read data
train_df = pd.read_csv('/content/drive/My Drive/train_set.csv',sep = '\t')
# One Hot transformation
vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])
clf = RidgeClassifier()
clf.fit(train_test[:100000], train_df['label'][:100000].values)
val_pred = clf.predict(train_test[100000:])
print(f1_score(train_df['label'][100000:], val_pred, average='macro'))
# 0.81
```

**TF-IDF + Ridge Regression 代码实现**
```
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
# Read data
train_df = pd.read_csv('/content/drive/My Drive/train_set.csv',sep = '\t')
# TF-IDF transformation
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
clf = RidgeClassifier()
clf.fit(train_test[:100000], train_df['label'][:100000].values)
val_pred = clf.predict(train_test[100000:])
print(f1_score(train_df['label'][100000:], val_pred, average='macro'))
# 0.873
```
**TF-IDF + Logistic Regression 代码实现**
```
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# Read data
train_df = pd.read_csv('/content/drive/My Drive/train_set.csv',sep = '\t')
# TF-IDF transformation
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
# split the dataset 
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, test_size = 0.5)
lr = LogisticRegression(C=4,n_jobs=8)
lr.fit(x_train,y_train)
print(f1_score(y_val, val_pred_lr, average='macro'))
# 0.92
```



# 参考文献
【1】 BMM 中文分词：https://blog.csdn.net/chenlei0630/article/details/40710441
【2】Ridge Regression & Lasso Regresssion： https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b
Blog Question: Word Embedding for Out-Of-Vocabulary(OOV)

