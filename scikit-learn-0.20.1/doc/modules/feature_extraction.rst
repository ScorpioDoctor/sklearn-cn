.. _feature_extraction:

==================
特征提取(Feature extraction)
==================

.. currentmodule:: sklearn.feature_extraction

:mod:`sklearn.feature_extraction` 模块可以被用于以机器学习算法支持的格式从原始数据集(如文本和图像)提取特征。

.. note::

   特征提取与特征选择 :ref:`feature_selection` 有很大的不同 ：前者是将任何形式的数据如文本，图像转换成可用于机器学习的数值型特征；
   后者是一种应用在这些特征上的机器学习技术。

.. topic:: 译者注

  本章节视频：
  `SKLearn特征抽取之特征字典向量化和特征哈希变换 
  <http://www.studyai.com/course/play/d26830535a874da2b49fb1ebc362cbbf>`_ ;
  `SKLearn特征抽取之文本特征抽取(词袋表示法) 
  <http://www.studyai.com/course/play/c554382d9ee54745a5c4b5f0ecc6ccd4>`_ 。

.. _dict_feature_extraction:

从字典加载特征
===========================

类 :class:`DictVectorizer` 可用于将 以标准Python ``dict`` 对象的列表形式表示的特征数组转换为 
scikit-learn 估计器使用的 NumPy/SciPy 表示形式。

虽然处理速度不是特别快，但Python的 ``dict`` 优点是使用方便，稀疏（缺失的特征不需要存储），
并且除了值之外还存储特征名称。


:class:`DictVectorizer` 类实现了对 标称型特征(categorical or  nominal or discrete features)的 one-of-K 或 "one-hot" 编码。
标称型特征是 "attribute-value" 对，其中 value的取值被限制在一个不排序的可能性的离散列表中。 (e.g. 话题标识符，对象类型，标签，名称)。

在下面, "city" 是一个 标称型属性(特征)，而 "temperature" 是一个传统的 数值型特征(numerical feature)::

  >>> measurements = [
  ...     {'city': 'Dubai', 'temperature': 33.},
  ...     {'city': 'London', 'temperature': 12.},
  ...     {'city': 'San Francisco', 'temperature': 18.},
  ... ]

  >>> from sklearn.feature_extraction import DictVectorizer
  >>> vec = DictVectorizer()

  >>> vec.fit_transform(measurements).toarray()
  array([[ 1.,  0.,  0., 33.],
         [ 0.,  1.,  0., 12.],
         [ 0.,  0.,  1., 18.]])

  >>> vec.get_feature_names()
  ['city=Dubai', 'city=London', 'city=San Francisco', 'temperature']

:class:`DictVectorizer` 类也是在自然语言处理模型中训练序列分类器的有用的表达变换(representation transformation)，
通常通过提取围绕特定兴趣词的特征窗口来工作。

例如，假设我们具有提取我们想要用作训练序列分类器（例如：块）的互补标签的部分语音（PoS）标签的一个算法。
以下 dict 可以是在 “The cat sat on the mat.” 的句子，围绕 “sat” 一词提取的这样一个特征窗口::

  >>> pos_window = [
  ...     {
  ...         'word-2': 'the',
  ...         'pos-2': 'DT',
  ...         'word-1': 'cat',
  ...         'pos-1': 'NN',
  ...         'word+1': 'on',
  ...         'pos+1': 'PP',
  ...     },
  ...     # in a real application one would extract many such dictionaries
  ... ]

上述描述可以被矢量化为适合于传递给分类器的稀疏二维矩阵（可能要在pipe之后进行 :class:`text.TfidfTransformer` 归一化）::

  >>> vec = DictVectorizer()
  >>> pos_vectorized = vec.fit_transform(pos_window)
  >>> pos_vectorized                # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
  <1x6 sparse matrix of type '<... 'numpy.float64'>'
      with 6 stored elements in Compressed Sparse ... format>
  >>> pos_vectorized.toarray()
  array([[1., 1., 1., 1., 1., 1.]])
  >>> vec.get_feature_names()
  ['pos+1=PP', 'pos-1=NN', 'pos-2=DT', 'word+1=on', 'word-1=cat', 'word-2=the']

你可以想象，如果一个文本语料库的每一个单词都提取了这样一个上下文，那么所得的矩阵将会非常宽（许多 one-hot-features），其中大部分通常将会是0。 
为了使产生的数据结构能够适应内存，该类 ``DictVectorizer`` 默认使用 ``scipy.sparse`` 矩阵而不是 ``numpy.ndarray``。



.. _feature_hashing:

特征哈希(散列)化
===============

.. currentmodule:: sklearn.feature_extraction

类 :class:`FeatureHasher` 是一种高速，低内存消耗的 向量化方法，它使用了特征散列化
(`feature hashing <https://en.wikipedia.org/wiki/Feature_hashing>`_) 技术 ，或可称为 “散列法”(hashing trick)的技术。 
该类的做法不是去构建 训练中遇到的特征 的哈希表，如向量化所做的那样, :class:`FeatureHasher` 类实例 将哈希函数应用于特征，
以便直接在样本矩阵中确定它们的列索引。 
结果是以牺牲可检测性(inspectability)为代价，带来速度的提高和内存使用的减少; 
hasher 不记得输入特征是什么样的，也没有 ``inverse_transform`` 办法。

由于散列函数可能导致（不相关）特征之间的冲突，因此使用带符号散列函数，并且散列值的符号确定存储在特征的输出矩阵中的值的符号。 
这样，碰撞可能会抵消而不是累积错误，并且任何输出特征的值的预期平均值为零。默认情况下，此机制将使用 ``alternate_sign=True`` 启用，
尤其对小型哈希表的大小（ ``n_features < 10000`` ）特别有用。 对于大哈希表的大小，可以禁用它，以便将输出传递给估计器，
如 :class:`sklearn.naive_bayes.MultinomialNB` 或 :class:`sklearn.feature_selection.chi2` 特征选择器，这些特征选项器希望输入是非负的。

:class:`FeatureHasher` 类接受三种类型的输入：mappings ，``(feature, value)`` pairs，或 strings。
其中 mappings 就像是python的 ``dict`` 或在 ``collections`` 模块中的字典的变体。
到底使用哪种参数依赖于构造器的 ``input_type`` 参数。
Mapping 被当作是由 ``(feature, value)`` 组成的列表(list), 而 单个字符串有一个内在的值 1 ，因此 ``['feat1', 'feat2', 'feat3']`` 
被解释成 ``[('feat1', 1), ('feat2', 1), ('feat3', 1)]``。
如果一个特征在一个样本中多次出现，那么该特征关联的值就会被累加起来，比如像这样 (``('feat', 2)`` 和 ``('feat', 3.5)`` 就变成了 ``('feat', 5.5)``)。
类 :class:`FeatureHasher` 的输出总是 CSR 格式的 一个 ``scipy.sparse`` 矩阵。

特征散列(Feature hashing)可以被用于文档分类，但是它不像 :class:`text.CountVectorizer` 类,
:class:`FeatureHasher` 类不进行单词分割 或其他预处理除了 Unicode-to-UTF-8 编码;
请看下面 :ref:`hashing_vectorizer` , 是一个 combined tokenizer/hasher 。

作为一个例子，考虑一个单词级的自然语言处理任务，它需要从 ``(token, part_of_speech)`` pairs 中抽取特征。
我们可以使用一个 Python 生成器函数 来提取特征 ::

  def token_features(token, part_of_speech):
      if token.isdigit():
          yield "numeric"
      else:
          yield "token={}".format(token.lower())
          yield "token,pos={},{}".format(token, part_of_speech)
      if token[0].isupper():
          yield "uppercase_initial"
      if token.isupper():
          yield "all_uppercase"
      yield "pos={}".format(part_of_speech)

然后, 要被传递到 ``FeatureHasher.transform`` 里面去的 ``raw_X`` 可以使用下面的方法构建::

  raw_X = (token_features(tok, pos_tagger(tok)) for tok in corpus)

然后使用下面的方法把它喂给 FeatureHasher 类的一个对象实例 (hasher) ::

  hasher = FeatureHasher(input_type='string')
  X = hasher.transform(raw_X)

得到的输出是一个 ``scipy.sparse`` 类型的矩阵 ``X``。

这里需要注意的是 由于我们使用了Python的生成器，导致在特征抽取过程中引入了懒惰性:
只有在hasher有需求的时候tokens才会被处理(tokens are only processed on demand from the hasher)。 

实现细节
----------------------

:class:`FeatureHasher` 类使用带符号的 32-bit 变体的 MurmurHash3。 作为其结果(也因为 ``scipy.sparse`` 里面的限制)，
当前支持的特征的最大数量 :math:`2^{31} - 1` 。

散列技巧(hashing trick)的原始形式源于Weinberger et al。 使用两个分开的哈希函数，:math:`h` 和 :math:`\xi` 分别确定特征的列索引和符号。 
现有的实现是基于假设：MurmurHash3的符号位与其他位独立(the sign bit of MurmurHash3 is independent of its other bits)。

由于使用简单的模数将哈希函数转换为列索引，建议使用2次幂作为 ``n_features`` 参数; 否则特征不会被均匀的分布到列中。


.. topic:: 参考文献:

 * Kilian Weinberger, Anirban Dasgupta, John Langford, Alex Smola and
   Josh Attenberg (2009). `Feature hashing for large scale multitask learning
   <http://alex.smola.org/papers/2009/Weinbergeretal09.pdf>`_. Proc. ICML.

 * `MurmurHash3 <https://github.com/aappleby/smhasher>`_.


.. _text_feature_extraction:

文本特征提取
=======================

.. currentmodule:: sklearn.feature_extraction.text


词袋表示法
-------------------------------

文本分析是机器学习算法的主要应用领域。 然而，原始数据，符号文字序列不能直接传递给算法，因为它们大多数要求具有固定长度的数字矩阵特征向量，
而不是具有可变长度的原始文本文档。

为解决这个问题，scikit-learn 提供了从文本内容中提取数字特征的最常见方法，即：

- **tokenizing** 令牌化 即对每个可能的 词令牌(token) 分成字符串并赋予整型id，例如通过使用空格和标点符号作为令牌分隔符(token separators)。

- **counting** 统计计数 即数出每个文档中令牌的出现次数。

- **normalizing** 标准化 即 对大多数样本/文档中出现的重要性递减的token进行归一化和加权

在这个机制中, 特征和样本是如下定义的：

- 每个单独的令牌发生频率（归一化或不归一化）被视为一个特征 (each **individual token occurrence frequency** (normalized or not)
  is treated as a **feature**.)。

- 给定文档中所有的令牌频率向量被看做一个多元样本(the vector of all the token frequencies for a given **document** is
  considered a multivariate **sample**.)。

因此，文档的集合(文集：corpus of documents)可被表示为矩阵形式，每行对应一个文本文档，每列对应文集中出现的词令牌(如单个词)。

我们称 向量化(**vectorization**) 是将文本文档集合转换为数字集合特征向量的通用方法。 这种特别的策略（令牌化，计数和归一化）被称为
**Bag of Words** 或 "Bag of n-grams" 表示法。文档由单词的出现与否和出现频率来描述，同时完全忽略文档中单词的相对位置信息。


稀疏性
--------

由于大多数文本文档通常只使用文集的词向量全集中的一个小子集，所以得到的矩阵将具有许多特征值为零（通常大于99％）。

例如，10,000 个短文本文档（如电子邮件）的集合将使用总共100,000个独特词的大小的词汇，而每个文档将单独使用100到1000个独特的单词。

为了能够将这样的矩阵存储在存储器中，并且还可以加速代数的矩阵/向量运算，实现通常将使用诸如 ``scipy.sparse`` 包中的稀疏实现


常见 Vectorizer 的用法
-----------------------

:class:`CountVectorizer` 在单个类中实现了 词语切分(tokenization) 和 出现频数统计(occurrence counting) ::

  >>> from sklearn.feature_extraction.text import CountVectorizer

这个模型有很多参数，但参数的默认初始值是相当合理的（请参阅 :ref:`参考文档 <text_feature_extraction_ref>` 了解详细信息）::

  >>> vectorizer = CountVectorizer()
  >>> vectorizer                     # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  CountVectorizer(analyzer=...'word', binary=False, decode_error=...'strict',
          dtype=<... 'numpy.int64'>, encoding=...'utf-8', input=...'content',
          lowercase=True, max_df=1.0, max_features=None, min_df=1,
          ngram_range=(1, 1), preprocessor=None, stop_words=None,
          strip_accents=None, token_pattern=...'(?u)\\b\\w\\w+\\b',
          tokenizer=None, vocabulary=None)

我们用该类对一个简约的文本语料库进行 分词(tokenize)和 统计单词出现频数 ::

  >>> corpus = [
  ...     'This is the first document.',
  ...     'This is the second second document.',
  ...     'And the third one.',
  ...     'Is this the first document?',
  ... ]
  >>> X = vectorizer.fit_transform(corpus)
  >>> X                              # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <4x9 sparse matrix of type '<... 'numpy.int64'>'
      with 19 stored elements in Compressed Sparse ... format>

默认配置是 通过提取至少包含2个字母的单词来对 string 进行分词。做这一步的函数可以显式地被调用 ::

  >>> analyze = vectorizer.build_analyzer()
  >>> analyze("This is a text document to analyze.") == (
  ...     ['this', 'is', 'text', 'document', 'to', 'analyze'])
  True

analyzer 在拟合过程中找到的每个 term（项）都会被分配一个唯一的整数索引，对应于 resulting matrix 中的一列。
此列的一些说明可以被检索如下 ::

  >>> vectorizer.get_feature_names() == (
  ...     ['and', 'document', 'first', 'is', 'one',
  ...      'second', 'the', 'third', 'this'])
  True

  >>> X.toarray()           # doctest: +ELLIPSIS
  array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 2, 1, 0, 1],
         [1, 0, 0, 0, 1, 0, 1, 1, 0],
         [0, 1, 1, 1, 0, 0, 1, 0, 1]]...)

从 feature名称 到 列索引(column index) 的逆映射存储在 ``vocabulary_`` 属性中::

  >>> vectorizer.vocabulary_.get('document')
  1

因此，在未来对 transform 方法的调用中，在 训练语料库(training corpus) 中没有看到的单词将被完全忽略：::

  >>> vectorizer.transform(['Something completely new.']).toarray()
  ...                           # doctest: +ELLIPSIS
  array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]...)

请注意，在前面的语料库中，第一个和最后一个文档具有完全相同的词，因此被编码成相同的向量。 
特别是我们丢失了 最后一个文件是一个疑问的形式 的信息。
为了保留局部的词组顺序信息，除了提取一元模型 1-grams（个别词）之外，我们还可以提取 2-grams 的单词 ::

  >>> bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
  ...                                     token_pattern=r'\b\w+\b', min_df=1)
  >>> analyze = bigram_vectorizer.build_analyzer()
  >>> analyze('Bi-grams are cool!') == (
  ...     ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])
  True

由上述 向量化器(vectorizer) 提取的 vocabulary 因此会变得更大，同时可以在局部定位模式时消除歧义 ::

  >>> X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
  >>> X_2
  ...                           # doctest: +ELLIPSIS
  array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]]...)


特别是 “Is this” 的疑问形式只出现在最后一个文档中::

  >>> feature_index = bigram_vectorizer.vocabulary_.get('is this')
  >>> X_2[:, feature_index]     # doctest: +ELLIPSIS
  array([0, 0, 0, 1]...)

.. _stop_words:

使用 stop words
................

Stop words are words like "and", "the", "him", which are presumed to be
uninformative in representing the content of a text, and which may be
removed to avoid them being construed as signal for prediction.  Sometimes,
however, similar words are useful for prediction, such as in classifying
writing style or personality.

There are several known issues in our provided 'english' stop word list. See
[NQY18]_.

Please take care in choosing a stop word list.
Popular stop word lists may include words that are highly informative to
some tasks, such as *computer*.

You should also make sure that the stop word list has had the same
preprocessing and tokenization applied as the one used in the vectorizer.
The word *we've* is split into *we* and *ve* by CountVectorizer's default
tokenizer, so if *we've* is in ``stop_words``, but *ve* is not, *ve* will
be retained from *we've* in transformed text.  Our vectorizers will try to
identify and warn about some kinds of inconsistencies.

.. topic:: References

    .. [NQY18] J. Nothman, H. Qin and R. Yurchak (2018).
               `"Stop Word Lists in Free Open-source Software Packages"
               <http://aclweb.org/anthology/W18-2502>`__.
               In *Proc. Workshop for NLP Open Source Software*.

.. _tfidf:

Tf–idf term weighting
---------------------

在一个大的文本语料库中，一些单词将出现很多次（例如 “the”, “a”, “is” 是英文），因此对文档的实际内容没有什么有意义的信息。 
如果我们直接将直接计数数据提供给分类器，那么这些非常频繁的词组(very frequent terms)会掩盖住那些我们感兴趣但却很少出现的词。

为了重新计算特征权重，并将其转化为适合分类器使用的浮点值，因此使用 tf-idf 变换(tf–idf transform)是非常常见的。

Tf 表示 **term-frequency** 而 tf–idf 表示 term-frequency 乘以 **inverse document-frequency**:
:math:`\text{tf-idf(t,d)}=\text{tf(t,d)} \times \text{idf(t)}`.

使用 ``TfidfTransformer`` 的默认设置, ``TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)``
term 的频率( the term frequency), 一个term出现在给定文档的次数，被乘以 idf component, 计算如下：

:math:`\text{idf}(t) = log{\frac{1 + n_d}{1+\text{df}(d,t)}} + 1`,

其中 :math:`n_d` 是文档的总数量，:math:`\text{df}(d,t)` 包含某个 term :math:`t` 的文档的数量。然后计算出的 tf-idf
vectors 用欧式范数归一化，如下所示：

:math:`v_{norm} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v{_1}^2 +
v{_2}^2 + \dots + v{_n}^2}}`.

上面所介绍的就是用于信息检索领域的原始的 term加权机制。该 term加权机制 在文档分类和聚类中的表现也比较好。

接下来的小节包含了对 tf-idfs 的进一步解释以及实验案例来说明 tf-idfs 是如何准确计算出来的，以及 scikit-learn 的类 
:class:`TfidfTransformer` 是如何计算 tf-idfs 的。还有 :class:`TfidfVectorizer` 类与标准的 idf 的定义的细微差别。
标准的 idf 的定义 如下所示：

:math:`\text{idf}(t) = log{\frac{n_d}{1+\text{df}(d,t)}}.`


在 :class:`TfidfTransformer` 类和 :class:`TfidfVectorizer` 类中，如果设置了 ``smooth_idf=False`` ,那么
数量 "1" 就被加到 idf 上而不是 idf 的分母上:

:math:`\text{idf}(t) = log{\frac{n_d}{\text{df}(d,t)}} + 1`

归一化是被 :class:`TfidfTransformer` 类实现的 ::

  >>> from sklearn.feature_extraction.text import TfidfTransformer
  >>> transformer = TfidfTransformer(smooth_idf=False)
  >>> transformer   # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  TfidfTransformer(norm=...'l2', smooth_idf=False, sublinear_tf=False,
                   use_idf=True)

请再次查看 :ref:`参考文档 <text_feature_extraction_ref>` 来获得所有参数的细节信息。

让我们以下面的数量来举个栗子。 第一个 term 出现次数 100%，因此可能不是很感兴趣。另外两个特征出现的次数仅仅比50%小点儿，
因此有可能是更加能够代表文档内容的表示::

  >>> counts = [[3, 0, 1],
  ...           [2, 0, 0],
  ...           [3, 0, 0],
  ...           [4, 0, 0],
  ...           [3, 2, 0],
  ...           [3, 0, 2]]
  ...
  >>> tfidf = transformer.fit_transform(counts)
  >>> tfidf                         # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
  <6x3 sparse matrix of type '<... 'numpy.float64'>'
      with 9 stored elements in Compressed Sparse ... format>

  >>> tfidf.toarray()                        # doctest: +ELLIPSIS
  array([[0.81940995, 0.        , 0.57320793],
         [1.        , 0.        , 0.        ],
         [1.        , 0.        , 0.        ],
         [1.        , 0.        , 0.        ],
         [0.47330339, 0.88089948, 0.        ],
         [0.58149261, 0.        , 0.81355169]])

每一行都被归一化到单位欧式范数:

:math:`v_{norm} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v{_1}^2 +
v{_2}^2 + \dots + v{_n}^2}}`

比如, 我们可以计算在 `counts` 数组中的第一个文档中第一个 term 的 tf-idf :

:math:`n_{d} = 6`

:math:`\text{df}(d, t)_{\text{term1}} = 6`

:math:`\text{idf}(d, t)_{\text{term1}} =
log \frac{n_d}{\text{df}(d, t)} + 1 = log(1)+1 = 1`

:math:`\text{tf-idf}_{\text{term1}} = \text{tf} \times \text{idf} = 3 \times 1 = 3`

现在, 如果我们重复上述计算过程去计算文档中剩余的 2 个 terms, 我们可以得到：

:math:`\text{tf-idf}_{\text{term2}} = 0 \times (log(6/1)+1) = 0`

:math:`\text{tf-idf}_{\text{term3}} = 1 \times (log(6/2)+1) \approx 2.0986`

原始 tf-idfs 的向量:

:math:`\text{tf-idf}_{\text{raw}} = [3, 0, 2.0986].`


然后, 应用 Euclidean (L2) norm, 我们可以在文档1 上得到以下 tf-idfs :

:math:`\frac{[3, 0, 2.0986]}{\sqrt{\big(3^2 + 0^2 + 2.0986^2\big)}}
= [ 0.819,  0,  0.573].`

更进一步, 默认参数 ``smooth_idf=True`` 会添加 "1" 到分子和分母上，就好像又看到了另一篇文档而这篇文档恰好包含了所有的term仅仅一次，这么做就可以避免
除零的异常发生了:

:math:`\text{idf}(t) = log{\frac{1 + n_d}{1+\text{df}(d,t)}} + 1`

使用这个修改版本, 文档1 中 第三个 term 的 tf-idf 变为 1.8473:

:math:`\text{tf-idf}_{\text{term3}} = 1 \times log(7/3)+1 \approx 1.8473`

并且 L2-normalized tf-idf 变为

:math:`\frac{[3, 0, 1.8473]}{\sqrt{\big(3^2 + 0^2 + 1.8473^2\big)}}
= [0.8515, 0, 0.5243]`::

  >>> transformer = TfidfTransformer()
  >>> transformer.fit_transform(counts).toarray()
  array([[0.85151335, 0.        , 0.52433293],
         [1.        , 0.        , 0.        ],
         [1.        , 0.        , 0.        ],
         [1.        , 0.        , 0.        ],
         [0.55422893, 0.83236428, 0.        ],
         [0.63035731, 0.        , 0.77630514]])

``fit`` 方法计算出的每个feature的权重被保存在模型属性 ``idf_`` ::

  >>> transformer.idf_                       # doctest: +ELLIPSIS
  array([1. ..., 2.25..., 1.84...])




由于 tf–idf 在文本特征提取中被经常使用，我们还提供了另一个类 :class:`TfidfVectorizer` 来组合 :class:`CountVectorizer` 和 :class:`TfidfTransformer`
的所有的选项到一个单一模型中去 ::

  >>> from sklearn.feature_extraction.text import TfidfVectorizer
  >>> vectorizer = TfidfVectorizer()
  >>> vectorizer.fit_transform(corpus)
  ...                                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <4x9 sparse matrix of type '<... 'numpy.float64'>'
      with 19 stored elements in Compressed Sparse ... format>

虽然 tf-idf normalization 通常非常有用，但是可能有一种情况是二元出现标记( binary occurrence markers)会提供更好的特征。 
这可以使用类 :class:`CountVectorizer` 的 ``binary`` 参数来实现。 特别地，一些估计器，诸如 :ref:`bernoulli_naive_bayes`
显式的使用离散的布尔随机变量。 而且，非常短的文本很可能影响 tf-idf 值，而 二进制出现信息(binary occurrence info) 更稳定。

通常情况下，调整特征提取参数的最佳方法是使用基于网格搜索的交叉验证，例如通过将特征提取器与分类器进行流水线化:

 * :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py`


解码文本文件
-------------------
文本由字符组成，但文件由字节组成。字节转化成字符依照一定的编码(encoding)方式。 
为了在Python中的使用文本文档，这些字节必须被 解码 为 Unicode 的字符集。 
常用的编码方式有 ASCII，Latin-1（西欧），KOI8-R（俄语）和通用编码 UTF-8 和 UTF-16。
还有许多其他的编码存在

.. note::
    编码也可以称为 ‘字符集’, 但是这个术语不太准确: 单个字符集可能存在多个编码。

scikit-learn 中的文本提取器知道如何解码文本文件， 但只有当您告诉他们文件的编码的情况下才行， 
:class:`CountVectorizer` 才需要一个 ``encoding`` 参数。 
对于现代文本文件，正确的编码可能是 UTF-8，因此它也是默认解码方式 (``encoding="utf-8"``)。

如果正在加载的文本不是使用UTF-8进行编码，则会得到 ``UnicodeDecodeError``. 
矢量化的方式可以通过设定 ``decode_error`` 参数设置为 ``"ignore"`` 或 ``"replace"`` 来避免抛出解码错误。 
有关详细信息，请参阅Python函数 ``bytes.decode`` 的文档（在Python提示符下键入 ``help(bytes.decode)`` ）。

如果您在解码文本时遇到问题，请尝试以下操作::

- 了解文本的实际编码方式。该文件可能带有标题或 README，告诉您编码，或者可能有一些标准编码，您可以根据文本的来源来推断编码方式。

- 您可能可以使用 UNIX 命令 ``file`` 找出它一般使用什么样的编码。 Python ``chardet`` 模块附带一个名为 ``chardetect.py`` 的脚本，
  它会猜测具体的编码，尽管你不能依靠它的猜测是正确的。

- 你可以尝试 UTF-8 并忽略错误。您可以使用 ``bytes.decode(errors='replace')`` 对字节字符串进行解码，
  用无意义字符替换所有解码错误，或在向量化器中设置 ``decode_error='replace'``. 这可能会损坏您的功能的有用性。

- 真实文本可能来自各种使用不同编码的来源，或者甚至以与编码的编码不同的编码进行粗略解码。这在从 Web 检索的文本中是常见的。
  Python 包 ``ftfy`` 可以自动排序一些解码错误类，所以您可以尝试将未知文本解码为 ``latin-1`` ，然后使用 ``ftfy`` 修复错误。

- 如果文本的编码的混合，那么它很难整理分类（20个新闻组数据集的情况），您可以把它们回到简单的单字节编码，如 ``latin-1`` 。
  某些文本可能显示不正确，但至少相同的字节序列将始终代表相同的功能。.

例如，以下代码段使用 ``chardet`` （没有附带在scikit-learn中，必须单独安装）来计算出编码方式。
然后，它将文本向量化并打印学习的词汇（特征）。输出在下方给出。

  >>> import chardet    # doctest: +SKIP
  >>> text1 = b"Sei mir gegr\xc3\xbc\xc3\x9ft mein Sauerkraut"
  >>> text2 = b"holdselig sind deine Ger\xfcche"
  >>> text3 = b"\xff\xfeA\x00u\x00f\x00 \x00F\x00l\x00\xfc\x00g\x00e\x00l\x00n\x00 \x00d\x00e\x00s\x00 \x00G\x00e\x00s\x00a\x00n\x00g\x00e\x00s\x00,\x00 \x00H\x00e\x00r\x00z\x00l\x00i\x00e\x00b\x00c\x00h\x00e\x00n\x00,\x00 \x00t\x00r\x00a\x00g\x00 \x00i\x00c\x00h\x00 \x00d\x00i\x00c\x00h\x00 \x00f\x00o\x00r\x00t\x00"
  >>> decoded = [x.decode(chardet.detect(x)['encoding'])
  ...            for x in (text1, text2, text3)]        # doctest: +SKIP
  >>> v = CountVectorizer().fit(decoded).vocabulary_    # doctest: +SKIP
  >>> for term in v: print(v)                           # doctest: +SKIP

(根据 ``chardet`` 的版本，可能会返回第一个值错误的结果。)

有关 Unicode 和字符编码的一般介绍，请参阅 Joel Spolsky's `Absolute Minimum Every Software Developer Must Know
About Unicode <http://www.joelonsoftware.com/articles/Unicode.html>`_.

.. _`ftfy`: https://github.com/LuminosoInsight/python-ftfy


应用和案例
-------------------------

词汇表达方式相当简单，但在实践中却非常有用。

特别是在 **supervised setting** 中，它能够把快速和可扩展的线性模型组合来训练 **document classifiers** , 例如:

 * :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`

在 **unsupervised setting** 中，可以通过应用诸如 :ref:`k_means` 的聚类算法来将相似文档分组在一起：

  * :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`

最后，通过松弛聚类的约束条件，可以通过使用非负矩阵分解( :ref:`NMF` )来发现语料库的主要主题：

  * :ref:`sphx_glr_auto_examples_applications_plot_topics_extraction_with_nmf_lda.py`


词袋表示法的局限性
----------------------------------------------

单个单词(unigrams)的集合无法捕获短语和多字表达，有效地忽略了任何单词顺序依赖。
另外，这个单词模型不包含潜在的拼写错误或词汇推倒。

N-grams 来救场！不去构建一个简单的unigrams集合 (n=1)，而是使用bigrams集合 (n=2)，其中计算连续字对。

还可以考虑 n-gram 的集合，这是一种对拼写错误和派生有弹性的表示。

例如，假设我们正在处理两个文档的语料库： ``['words', 'wprds']`` . 第二个文件包含 `words` 一词的拼写错误。 
一个简单的单词表示将把这两个视为非常不同的文档，两个可能的特征都是不同的。 
然而，一个 2-gram 的表示可以找到匹配的文档中的8个特征中的4个，这可能有助于优选的分类器更好地决定::

  >>> ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
  >>> counts = ngram_vectorizer.fit_transform(['words', 'wprds'])
  >>> ngram_vectorizer.get_feature_names() == (
  ...     [' w', 'ds', 'or', 'pr', 'rd', 's ', 'wo', 'wp'])
  True
  >>> counts.toarray().astype(int)
  array([[1, 1, 1, 0, 1, 1, 1, 0],
         [1, 1, 0, 1, 1, 1, 0, 1]])

在上面的例子中，使用 ``char_wb`` 分析器，它只能从字边界内的字符（每侧填充空格）创建 n-gram。 ``char`` 分析器可以创建跨越单词的 n-gram ::

  >>> ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5))
  >>> ngram_vectorizer.fit_transform(['jumpy fox'])
  ...                                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <1x4 sparse matrix of type '<... 'numpy.int64'>'
     with 4 stored elements in Compressed Sparse ... format>
  >>> ngram_vectorizer.get_feature_names() == (
  ...     [' fox ', ' jump', 'jumpy', 'umpy '])
  True

  >>> ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5))
  >>> ngram_vectorizer.fit_transform(['jumpy fox'])
  ...                                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <1x5 sparse matrix of type '<... 'numpy.int64'>'
      with 5 stored elements in Compressed Sparse ... format>
  >>> ngram_vectorizer.get_feature_names() == (
  ...     ['jumpy', 'mpy f', 'py fo', 'umpy ', 'y fox'])
  True

对于使用白色空格进行单词分离的语言，对于语言边界感知变体 ``char_wb`` 尤其有趣，
因为在这种情况下，它会产生比原始 ``char`` 变体显着更少的噪音特征。 
对于这样的语言，它可以增加使用这些特征训练的分类器的预测精度和收敛速度，
同时保持关于拼写错误和词导出的稳健性。

虽然可以通过提取 n-gram 而不是单独的单词来保存一些局部定位信息，
但是包含 n-gram 的单词和袋子可以破坏文档的大部分内部结构，因此破坏了该内部结构的大部分含义。

为了处理自然语言理解的更广泛的任务，因此应考虑到句子和段落的局部结构。因此，许多这样的模型将被称为 “结构化输出” 问题，
这些问题目前不在 scikit-learn 的范围之内。


.. _hashing_vectorizer:

用散列技巧矢量化大型语料库
------------------------------------------------------

上述向量化方案是简单的，但是它存在 从字符串令牌到整数特征索引的内存映射 （ ``vocabulary_`` 属性），在处理 大型数据集时会引起几个问题 :

- 语料库越大，词汇量越大，使用的内存也越大.

- 拟合（fitting）需要根据原始数据集的大小等比例分配中间数据结构的大小.

- 构建词映射需要完整的传递数据集，因此不可能以严格在线的方式拟合文本分类器. 

- pickling和un-pickling ``vocabulary`` 很大的向量器会非常慢（通常比pickling/un-pickling flat数据结构，比如同等大小的Numpy数组还要慢）.

- 将向量化任务分隔成并行的子任务很不容易实现，因为 ``vocabulary_`` 属性要共享状态有一个细颗粒度的同步障碍：
  从标记字符串中映射特征索引与每个标记的首次出现顺序是独立的，
  因此应该被共享，在这点上并行worker的性能受到了损害，使他们比串行更慢。

通过组合由 ``sklearn.feature_extraction.FeatureHasher`` 类实现的 “散列技巧” (:ref:`Feature_hashing`) 
和 :class:`CountVectorizer` 的文本预处理和标记化功能，可以克服这些限制。

这种组合是在 :class:`HashingVectorizer` 中实现的，该类是与 :class:`CountVectorizer` 大部分 API 兼容的变换器(transformer)类。 
:class:`HashingVectorizer` 是无状态的，这意味着您不需要 ``fit`` 它::

  >>> from sklearn.feature_extraction.text import HashingVectorizer
  >>> hv = HashingVectorizer(n_features=10)
  >>> hv.transform(corpus)
  ...                                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <4x10 sparse matrix of type '<... 'numpy.float64'>'
      with 16 stored elements in Compressed Sparse ... format>

你可以看到从向量输出中抽取了16个非0特征标记：与之前由 :class:`CountVectorizer` 在同一个样本语料库抽取的19个非0特征要少。
差异来自哈希方法的冲突，因为较低的 ``n_features`` 参数的值。

在真实世界的环境下，``n_features`` 参数可以使用默认值 ``2 ** 20``（将近100万可能的特征）。
如果内存或者下游模型的大小是一个问题，那么选择一个较小的值比如 ``2 ** 18`` 可能有一些帮助，
而不需要为典型的文本分类任务引入太多额外的冲突。

注意 维度并不影响CPU的算法训练时间，训练是在操作CSR矩阵 
（``LinearSVC(dual=True)``, ``Perceptron``, ``SGDClassifier``, ``PassiveAggressive``），
但是，它对CSC matrices (``LinearSVC(dual=False)``, ``Lasso()``, etc)算法有效。

让我们再次尝试使用默认设置::

  >>> hv = HashingVectorizer()
  >>> hv.transform(corpus)
  ...                               # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
  <4x1048576 sparse matrix of type '<... 'numpy.float64'>'
      with 19 stored elements in Compressed Sparse ... format>

冲突没有再出现，但是，代价是输出空间的维度值非常大。当然，这里使用的19词以外的其他词之前仍会有冲突。

:class:`HashingVectorizer` 还具有以下限制 ::

- 不能反转模型（没有 ``inverse_transform`` 方法）。 因为进行mapping的哈希方法的单向本质，也无法访问原始的字符串表征。

- 没有提供 IDF 加权，因为这需要在模型中引入状态。如果需要的话，可以在管道中添加 :class:`TfidfTransformer` 。

使用 HashingVectorizer 执行核外scaling 
------------------------------------------------------

使用 :class:`HashingVectorizer` 的一个有趣的开发是执行核外 `out-of-core`_ 缩放的能力。 
这意味着我们可以从无法放入电脑主内存的数据中进行学习。

.. _out-of-core: https://en.wikipedia.org/wiki/Out-of-core_algorithm

实现核外扩展的一个策略是将数据以流的方式以一小批提交给估计器。每批的向量化都是用 :class:`HashingVectorizer` 。
这样来保证估计器的输入空间的维度是相等的。 因此任何时间使用的内存数都限定在小频次的大小。 尽管用这种方法可以处理的数据没有限制，
但是从实用角度学习时间受到想要在这个任务上花费的CPU时间的限制。

对于文本分类任务中的外核缩放的完整示例，请参阅  :ref:`sphx_glr_auto_examples_applications_plot_out_of_core_classification.py`.

自定义向量化器类
----------------------------------

通过将可调用对象传递给向量化程序构造函数可以定制行为:::

  >>> def my_tokenizer(s):
  ...     return s.split()
  ...
  >>> vectorizer = CountVectorizer(tokenizer=my_tokenizer)
  >>> vectorizer.build_analyzer()(u"Some... punctuation!") == (
  ...     ['some...', 'punctuation!'])
  True

特别的，我们命名:

  * ``preprocessor``: 可以将整个文档作为输入（作为单个字符串）的可调用对象，并返回文档的可能转换的版本，仍然是整个字符串。
    这可以用于删除HTML标签，小写整个文档等。

  * ``tokenizer``: 一个可从预处理器接收输出并将其分成标记的可调用对象，然后返回这些列表。

  * ``analyzer``: 一个可替代预处理程序和标记器的可调用程序。默认分析仪都会调用预处理器和刻录机，但是自定义分析仪将会跳过这个。 
    N-gram 提取和停止字过滤在分析器级进行，因此定制分析器可能必须重现这些步骤。

(Lucene 用户可能会识别这些名称，但请注意，scikit-learn 概念可能无法一对一映射到 Lucene 概念上。)

为了使预处理器，标记器和分析器了解模型参数，可以从类派生并覆盖 ``build_preprocessor``, ``build_tokenizer`` 和 ``build_analyzer`` 工厂方法，
而不是传递自定义函数。

一些经验和技巧:

  * 如果文档由外部包进行预先标记，则将它们存储在文件（或字符串）中，令牌由空格分隔，并传递  ``analyzer=str.split``  
  * Fancy 令牌级分析，如词干，词法，复合分割，基于词性的过滤等不包括在 scikit-learn 代码库中，但可以通过定制分词器或分析器来添加。
  
    这是一个 ``CountVectorizer``, 使用 NLTK 的 tokenizer 和 lemmatizer: `NLTK <http://www.nltk.org>`_::

        >>> from nltk import word_tokenize          # doctest: +SKIP
        >>> from nltk.stem import WordNetLemmatizer # doctest: +SKIP
        >>> class LemmaTokenizer(object):
        ...     def __init__(self):
        ...         self.wnl = WordNetLemmatizer()
        ...     def __call__(self, doc):
        ...         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        ...
        >>> vect = CountVectorizer(tokenizer=LemmaTokenizer())  # doctest: +SKIP

    (请注意，这不会过滤标点符号。)


    例如，以下例子将英国的一些拼写变成美国拼写::

        >>> import re
        >>> def to_british(tokens):
        ...     for t in tokens:
        ...         t = re.sub(r"(...)our$", r"\1or", t)
        ...         t = re.sub(r"([bt])re$", r"\1er", t)
        ...         t = re.sub(r"([iy])s(e$|ing|ation)", r"\1z\2", t)
        ...         t = re.sub(r"ogue$", "og", t)
        ...         yield t
        ...
        >>> class CustomVectorizer(CountVectorizer):
        ...     def build_tokenizer(self):
        ...         tokenize = super(CustomVectorizer, self).build_tokenizer()
        ...         return lambda doc: list(to_british(tokenize(doc)))
        ...
        >>> print(CustomVectorizer().build_analyzer()(u"color colour")) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [...'color', ...'color']

    用于其他样式的预处理; 例子包括 stemming, lemmatization, 或 normalizing numerical tokens, 后者说明如下:

     * :ref:`sphx_glr_auto_examples_bicluster_plot_bicluster_newsgroups.py`


在处理不使用显式字分隔符（例如空格）的亚洲语言时，自定义向量化器也是有用的。

.. _image_feature_extraction:

图像特征提取
========================

.. currentmodule:: sklearn.feature_extraction.image

图像块提取
----------------


:func:`extract_patches_2d` 函数从存储为二维数组的灰度图像或三维数组的彩色图像中提取图像块(patches)。
彩色图像的颜色信息在第三个纬度存放。如果要从所有的图像块(patches)中重建图像，请使用函数 :func:`reconstruct_from_patches_2d` 。
比如我们生成一个 4x4 像素的RGB格式三通道图像::

    >>> import numpy as np
    >>> from sklearn.feature_extraction import image

    >>> one_image = np.arange(4 * 4 * 3).reshape((4, 4, 3))
    >>> one_image[:, :, 0]  # R channel of a fake RGB picture
    array([[ 0,  3,  6,  9],
           [12, 15, 18, 21],
           [24, 27, 30, 33],
           [36, 39, 42, 45]])

    >>> patches = image.extract_patches_2d(one_image, (2, 2), max_patches=2,
    ...     random_state=0)
    >>> patches.shape
    (2, 2, 2, 3)
    >>> patches[:, :, :, 0]
    array([[[ 0,  3],
            [12, 15]],
    <BLANKLINE>
           [[15, 18],
            [27, 30]]])
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> patches.shape
    (9, 2, 2, 3)
    >>> patches[4, :, :, 0]
    array([[15, 18],
           [27, 30]])

现在让我们尝试通过在重叠区域进行平均来从图像块重建原始图像::

    >>> reconstructed = image.reconstruct_from_patches_2d(patches, (4, 4, 3))
    >>> np.testing.assert_array_equal(one_image, reconstructed)

:class:`PatchExtractor` 类的工作方式与 :func:`extract_patches_2d` 函数相同, 只是它支持多幅图像作为输入。
它被实现为一个estimator，因此它可以在 pipelines 中使用。请看::

    >>> five_images = np.arange(5 * 4 * 4 * 3).reshape(5, 4, 4, 3)
    >>> patches = image.PatchExtractor((2, 2)).transform(five_images)
    >>> patches.shape
    (45, 2, 2, 3)

图像的连接图
-------------------------------

scikit-learn 中的好几个estimators可以使用特征或样本之间的连接信息(connectivity information)。 
例如，Ward clustering (:ref:`hierarchical_clustering`) 可以只把相邻像素(neighboring pixels)聚集在一起，从而形成连续的斑块:

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_coin_ward_segmentation_001.png
   :target: ../auto_examples/cluster/plot_coin_ward_segmentation.html
   :align: center
   :scale: 40

出于这个目的, 这些estimators使用一个连接性矩阵('connectivity' matrix), 给出哪些样本是连接着的。

函数 :func:`img_to_graph` 从2D或3D图像返回这样一个矩阵('connectivity' matrix)。
类似地，函数 :func:`grid_to_graph` 为给定shape的图像构建连接矩阵。

这些矩阵可用于在使用连接信息的估计器中强加连接，如 (:ref:`hierarchical_clustering`)，而且还要构建预计算的内核或相似矩阵。

.. note:: **案例**

   * :ref:`sphx_glr_auto_examples_cluster_plot_coin_ward_segmentation.py`

   * :ref:`sphx_glr_auto_examples_cluster_plot_segmentation_toy.py`

   * :ref:`sphx_glr_auto_examples_cluster_plot_feature_agglomeration_vs_univariate_selection.py`
