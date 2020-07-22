# NLP 之 新闻分类 - 1
![1](https://user-images.githubusercontent.com/61811515/87897201-8f465b00-ca18-11ea-9fe4-542b80409ed5.jpeg)

## 前言 
今天带来的是对阿里云&Datawhale 零基础入门赛事的学习打卡笔记第一部分- 初识NLP。

本文涉及到对于NLP的初步了解以及对新闻文本分类赛题的初步思路。 
全文都是基于网上资料查询理解，如有不合理的地方欢迎大家留言纠正，深入探讨。

 ## NLP(Natural Language Procesing)定义以及应用 
自然语言学习(Natural Language Processing) 是人工智能和语言学领域一个分支学科。 如同小宝宝初学语言时，自然语言学习是教电脑理解不同的语言。 目前自然语言学习技术主要有对话系统，文本情感分析，信息抽取，翻译，文本朗读，自然语言生成，文本分类。 

这一次学习主要着重于自然语言学习在文本分类上的应用。 那么接下来让我们来看看本次的赛题吧。 

## 赛题：零基础入门NLP - 新闻文本分类
链接：https://tianchi.aliyun.com/competition/entrance/531810/information

### 数据简介

这次赛题的数据包含了14种类别的文本数据，包含财经，彩票，房产，股票，家居，教育，科技，社会，时尚，时政，体育，星座，游戏，娱乐。每个类别都有相对应的数字 {‘科技’：0，‘股票’：1，‘体育’：2，‘娱乐’：3，‘时政’：4，‘社会’：5，‘教育’：6，‘财经’：7，‘家居’：8，‘游戏’：9，‘房产’：10，’时尚‘：11，’彩票‘：12，’星座‘：13}。􏰚􏰛􏲟􏰭􏰮􏲟􏰧􏰨􏲟􏰜􏰝􏰌􏰣􏰤􏲟􏰬􏰙􏲟􏰩􏰪􏲟􏰘􏰙􏲟􏰥􏰦􏲟􏰢􏰛􏲟􏰖􏰗􏲟􏰠􏰡􏲟􏰞􏰫􏰚􏰛􏲟􏰭􏰮􏲟􏰧􏰨􏲟􏰜􏰝􏰌􏰣􏰤􏲟􏰬􏰙􏲟􏰩􏰪􏲟􏰘􏰙􏲟􏰥􏰦􏲟􏰢􏰛􏲟􏰖􏰗􏲟

### 比赛测评标准
比赛的测评标准采用了F1 score。 那么问题来了，什么是F1 score呢？ 
F1 score 是一种测量准确度的常用指标。F1 score 被定义为精准率和查全率的调和平均数，取值范围在0到1之间。 公式定义如下：


<img width="288" alt="f1" src="https://user-images.githubusercontent.com/61811515/87957811-c51e2a80-ca7e-11ea-8fe0-30eb0e167440.png"> 【1】

精准率(Precision)指的是在所有数据中里有多少内容被正确分类。具体公式如下图： 


<img width="484" alt="precision" src="https://user-images.githubusercontent.com/61811515/87958176-33fb8380-ca7f-11ea-8dc6-bee085189349.png"> 【2】

查全率(Recall Ratio)指的是分类正确的内容占那个类别的总量比。 举个具体例子Group A 总共有a个数据，其中b个被正确分类，那么 b/a 就是我们所说的查全率。 具体公式如下图：


<img width="446" alt="recall" src="https://user-images.githubusercontent.com/61811515/87958121-247c3a80-ca7f-11ea-8614-bccc804cd39d.png"> 【2】

因为我们希望model 可以在精准率和查全率越高越好。这也告诉我们高F1 score 会是我们测试model的标准。 

### 数据初探
简单的将数据用pandas load 了一遍，以下是数据的例子。 


![data preview](https://user-images.githubusercontent.com/61811515/87950927-ee868880-ca75-11ea-8a1f-96acc17c3bca.png)

此次比赛为防止手动分类，将文本进行了匿名处理。 

可以看到这个数据是属于非结构化数据（Unstructured Data)。那么这时候一个小问号跳了出来，什么是非结构化数据呢？
#### 结构化数据（Structured Data)和非结构化数据（Unstructured Data)
 最简单的方法去区分就在于是否存储在一个传统的数据库里且很容易找出检索规律。 结构化数据通常有清晰的data type 也会有很明显的规律，例如手机号码。手机号码的长度和data type 都是很容易可以找出来的。 非结构化数据则包含了所有没有predefined 过的data models 或者schema。 常见的非结构化数据有： 
* 文本文件
* email （Usually referred as semi-structured, since it contains internal structure) 
* 社交媒体数据 （如微信朋友圈数据）
* 网页数据
* 手机短信和定位
* 媒体（如MP3，照片等） 
* 视频
【3】

对训练集的分布如下图：


![distribution](https://user-images.githubusercontent.com/61811515/87968103-e6d2de00-ca8d-11ea-959f-82e4da7536f8.JPG)

可以看到每个组的数据量分布不均匀。数据分布不均会对model的影响之后在学习笔记中会继续讨论。  

### 解题思路
由于本次赛题对文本进行了匿名处理，不能使用中文分词等操作，这也成为了这次的一个难点。 

在介绍思路之前，我们需要了解一个概念，词向量。 在自然语言学习中，我们需要尽可能地让我们的计算机能明白词的意思，以此来提高任务的准确性。 因此如何在计算机中表示每个词成为了一个很重要的问题。 而词向量技术则是将词转化为向量，对于相似的词其对应的词向量也会相近。 由此让我们的计算机可以理解文字的意思。 下图为其中一些例子：


<img width="910" alt="words" src="https://user-images.githubusercontent.com/61811515/87974394-0242e680-ca98-11ea-902d-3dbafba76013.png"> 【5】

接下来让我们一起看看解题思路吧。 
对于非结构化数据，可能涉及到特征提取和分类模型两个部分。 以下是四种解题思路： 
* TF-IDF +机器学习分类器
直接使用TF-IDF 对文本提取特征，再使用分类器进行分类。 在分类器的选择上可以使用SVM，LR或者XG Boost。 这些方法也会在后续的学习笔记上进一步详细介绍。 
#### TF-IDF
TF-IDF, short for term frequency-Inverse document frequency 被用来衡量一个单词在一个文本里的重要性。 常被用来作为比重。 TF-IDF 与文字在文本中出现的次数成正比，同时与它在文本中的出现次数成反比。 计算公式如下图：


<img width="349" alt="tfidf" src="https://user-images.githubusercontent.com/61811515/87969753-ba6c9100-ca90-11ea-8fa8-640f6fd1bfa3.png">

以下几种方法都涉及到了分布式假设。换句话说就是相同上下文语境的词有相似的含义。 根据这个对词向量进行了优化。 
* FastText
FastText 是一种词向量，利用Facebook 提供的FastText 工具，可以快速构建出分类模型。 
相关的document 链接：https://fasttext.cc/docs/en/support.html

* Word2Vec + 深度学习分类器
Word2Vec 是进阶的一种词向量，再通过构建深度学习分类完成分类。深度学习分类的网络结构可使用TextCNN，TextRNN或者BiLSTM，详细介绍会在之后的学习笔记中进一步讨论。 

* BERT 词向量
BERT(Bidirectional Encoder Representation from Transformers) 是一款高配的词向量，具有强大的建模学习能力。进一步优化了以上两种词向量方法，引入了基于语言模型的动态表征，可用于解决一词多义。  详细介绍会在之后的学习笔记中讨论。

## 参考文献
【1】百度百科. https://baike.baidu.com/item/F1分数/13864979?fr=aladdin
【2】Wikipedia. https://en.wikipedia.org/wiki/Precision_and_recall
【3】Taylor, C. (2018, March 8). Structured vs. Unstructured Data. Retrieved July 20, 2020, from https://www.datamation.com/big-data/structured-vs-unstructured-data.html
【4】NLP 的词向量：https://www.infoq.cn/article/PFvZxgGDm27453BbS24W
【5】Machine Learning Crash Course from Google: https://developers.google.com/machine-learning/crash-course/embeddings/translating-to-a-lower-dimensional-space
![close-up-of-scrabble-tiles-forming-the-words-the-end-2889685](https://user-images.githubusercontent.com/61811515/87973752-028eb200-ca97-11ea-8a78-5ef5073f5aee.jpg)



