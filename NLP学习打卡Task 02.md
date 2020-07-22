# NLP 之 新闻分类 - 2 数据分析
![data](https://user-images.githubusercontent.com/61811515/88124178-8382a200-cb9a-11ea-8179-ff1fa0d119d8.jpg)

[image](https://healthitanalytics.com/news/machine-learning-uses-social-determinants-data-to-predict-utilization)

这次blog让我们对数据进行一波分析，观察一下数据。 

为方便将类别作为further reference，将每个类别与编号的对应放在下面：
{‘科技’：0，‘股票’：1，‘体育’：2，‘娱乐’：3，‘时政’：4，‘社会’：5，‘教育’：6，‘财经’：7，‘家居’：8，‘游戏’：9，‘房产’：10，’时尚‘：11，’彩票‘：12，’星座‘：13}

## 数据读取 
```
import pandas as pd
# read_csv 内分别对应了文件路径和分隔符
train_df = pd.read_csv("train_set.csv",sep="\t")
# 给出数据前5行的内容
train_df.head() 
```
下图为前5行的数据。 


![sample](https://user-images.githubusercontent.com/61811515/88192839-004d6480-cc0b-11ea-8c98-a4c730782150.png)

本次赛题数据共有20万行。在数据读取的过程中，可能会遇到的问题就是内存不够。 其中一种解决办法是直接告诉pandas 你所需要的数据类型是什么。因为当pandas在猜测你的csv每列的数据类型的时候，系统需要将你所有的数据都存在内存里，这样就会造成内存不足，无法读取全部数据。 以下是一个如何告诉pandas 数据类型的例子。
```
train = pd.read_csv("train_set.csv", dtype=dict(label=np.int32, text='string'))
```
由于我们的label的范围在0到13之间，所以32 bit 的整数就足够使用，进一步节省空间。 

## 数据分析
### 全文本数据分析 
#### 文本长度
```
%pylab inline 
# 因为text 都是由一个空格分开每个字符，可以用来分开句子
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())
```
%pylab 是一个魔法函数。实际上这个函数的内容包含了
```
import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot

from IPython.core.pylabtools import figsize, getfigs
from pylab import *
from numpy import *
```
它用一行代码导入了Numpy 以及Matplotlib 的library。 简化了你的代码。想要了解怎么用Pylab 中的function，可以从了解Numpy 和Matplotlib python library 入手。

Numpy 介绍链接：https://numpy.org/doc/1.19/ 

Matplotlib 介绍链接：https://matplotlib.org

在介绍完Pylab 之后，我们来看一下句子的平均字符长度吧。 


![sen_avg](https://user-images.githubusercontent.com/61811515/88196058-c0887c00-cc0e-11ea-943d-89267072b0d1.png)

由以上数据可以看出文本平均有907个字符。文章中位数在676个字符。这次赛题文本普遍较长。 

那文本长度的分布又是怎么样呢？ 
```
hist(train_df['text_len'],bins=200)
plt.xlabel('Text Char Count')
plt.title('Char Count Distribution')
```


![distribution](https://user-images.githubusercontent.com/61811515/88196347-12310680-cc0f-11ea-901e-e3aa88b2d2db.png)

图中可以看到超过8000个字符的占很少数。 

让我们Zoom in 到0 到8000个字符看看。 
```
hist(train_df['text_len'],bins=200)
plt.xlabel('Text Char Count')
plt.xlim([0,8000])
plt.title('Char Count Distribution')
```
这里的xlim 可以让你把x轴的数限制在一个范围内。 出来的结果如下图：
![distribution2](https://user-images.githubusercontent.com/61811515/88196742-89ff3100-cc0f-11ea-9778-b3b1c6441c0d.png)

拉近看，大部分的数据都是处于2000个字符以下。 

在看完字符长度的数据后，接下来来看一下所有文本中经常出现的字符吧。
```
from collections import Counter
# 将所有的text 整合在一起
all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
# 方便观察字符出现次数，用字符的数量从多到少进行排序
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

# Unique的字符
print(len(word_count))
# 6869

# 最高频字符
print(word_count[0])
# ('3750', 7482224)
```

从数据结果看出，在训练集中总共有6869个字符，其中编号3750 字符是所有文本中最常出现的字符。

我们还可以通过每个字符在不同文本的出现次数推测出标点符号的编号。 
```
# 先将text 变成set成为unique 字符，再将所有字符用‘ ’连起来放入dataframe 新的column 里
train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines= ' '.join(list(train_df['text_unique']))
word_count= Counter(all_lines.split(' '))

word_counts[0]
#('3750', 197997)

word_counts[1]
#('900', 197653)

word_counts[2]
#('648', 191975)
``` 
通过以上代码结果发现字符编号3750，900，648 出现字符覆盖95%以上的文本。这些字符为标点符号的概率较大。
而通常对于低频词我们在NLP中会选择忽略。因为过多的字符会使我们的training 时间加长但却没有获得等值的信息量。 至于用于筛选的频率则是一个需要测试的参数。 在之后实际写数据模型的时候会对方法进行探讨。 

### 文章平均句子数量（作业一）
上面我们发现 3750，900，648 是经常出现的字符。假设这些字符为标点符号，一篇文本会有多少个句子组成呢？ 

以 3750 作为句号， 我们对文本句子进行了计算。 
```
train_df['sent_len'] = train_df['text'].apply(lambda x: len(x.split(' 3750 ')))
train_df['sent_len'].describe()
```
得出下结果


![len](https://user-images.githubusercontent.com/61811515/88211554-db192000-cc23-11ea-9211-6610c82cb00f.png)

以 900 作为句号， 我们对文本句子进行了计算。 
```
train_df['sent_len'] = train_df['text'].apply(lambda x: len(x.split(' 900 ')))
train_df['sent_len'].describe()
```
得出下结果


![900](https://user-images.githubusercontent.com/61811515/88211814-3b0fc680-cc24-11ea-96b1-7192f62acf4f.png)

以 648 作为句号， 我们对文本句子进行了计算。 
```
train_df['sent_len'] = train_df['text'].apply(lambda x: len(x.split(' 648 ')))
train_df['sent_len'].describe()
```
得出下结果


![648](https://user-images.githubusercontent.com/61811515/88211826-3f3be400-cc24-11ea-8bcb-00b437da87f5.png)

句子数量在20到30之间浮动。 

若我们使用三个字符，算断句数量。 
```
import re
# re.split 可以帮助你使用多个字符分句
train_df['sent_len'] = train_df['text'].apply(lambda x: len(re.split(' 648 | 900 | 3750 ',x)))
train_df['sent_len'].describe()
```
结果为下图：


![split](https://user-images.githubusercontent.com/61811515/88212328-1405c480-cc25-11ea-8a25-82ebbd86383f.png)

平均会有78个断句。 75%的文本断句都在100 及以下。 

## 类别分析
在看完整体的文本后，我们针对每个类别的新闻文本进行分析。 
```
train_df['label'].value_counts().plot(kind='bar')
```

![type](https://user-images.githubusercontent.com/61811515/88212753-bb82f700-cc25-11ea-90ae-480689d0a242.png)

上图为每个类别的柱状图，从统计结果可以看出，赛题的数据类别分布存在较为不均匀的情况。 在训练集中科技类新闻最多，其次为股票类新闻，最少为星座新闻。在Task 1 中也提及这类情况。 这对于数据模型的精准度会有一定的影响。 一些可能的处理方法有：
* 选择合适的衡量标准。
    * 我们采用了F1 score 作为一个数据模型准确度的标准。 查全率也会作为一个优秀数据模型的一个考量。 
* 重新sample 训练集。对此我们有两种方式：
  * Under-sampling 
对于采量不足的sample，我们可以在数据充足的情况下，将其他多的sample 通过随机抽样的方式减少。以此达到均匀的数据集。 
  * Over-sampling
换个角度，我们可以从采量不足的sample 中入手，使用repetition，bootstrapping 或者 SMOTE 的方法增加数据量。

  我们也可以将以上两种方式结合起来用。 

* 正确的 K-folder Cross-Validation
当我们采用了over-sampling 的方法去解决数据不均匀的问题时，由于我们对于不常见的sample 产生新的数据集，会有over-fitting 的问题。 因此这个cross-validation 需要从sample之前的数据入手。 在sample 数据时加入random 的机制也可尽量减少overfitting的可能。 

* 用数据比例去重新抽取训练集
* 用Cluster 将每个类别进行组合，每个类别只留中心的数据。 
【2】

### 分类别文本字符分析 （作业二）
接下来我们将文本分开，对每个类别的文本字符进行分析。

开始对文本未进行删减处理， 发现每类文本新闻中出现次数多的字符前三名落入编号3750,900,648 字符。 从之前的分析中我们知道字符编号3750,900,648 有很大概率是标点符号。 因为对这些字符进行删减,进一步探究字符分布。 

```
train_df['text'] = train_df['text'].apply(lambda x: x.replace(' 3750 ',' ').replace(' 900 ',' ').replace(' 648 ',' '))
train_df.groupby('label').text.apply(lambda x: Counter(' '.join(list(x)).split(' ')).most_common(1))
```
得出如下结果：


![chars](https://user-images.githubusercontent.com/61811515/88216650-60ec9980-cc2b-11ea-8e6b-a768ba1788fc.png)


希望你能从这blog中获取新知。如有发现任何问题，欢迎纠错讨论。一起进步。


![images](https://user-images.githubusercontent.com/61811515/88217028-ee2fee00-cc2b-11ea-9d02-a82360bb326e.jpg)


# 参考文献
【1】读数据内存不足https://stackoverflow.com/questions/17557074/memory-error-when-using-pandas-read-csv

【2】不均衡数据处理方法：https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html

