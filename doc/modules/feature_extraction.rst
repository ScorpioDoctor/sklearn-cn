.. _feature_extraction:

==================
特征提取(Feature extraction)
==================

.. currentmodule:: sklearn.feature_extraction

:mod:`sklearn.feature_extraction` 模块可以被用于以机器学习算法支持的格式从原始数据集(如文本和图像)提取特征。

.. note::

   特征提取与特征选择 :ref:`feature_selection` 有很大的不同 ：前者是将任何形式的数据如文本，图像转换成可用于机器学习的数值型特征；
   后者是一种应用在这些特征上的机器学习技术。


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
以下 dict 可以是在 “坐在垫子上的猫” 的句子，围绕 “sat” 一词提取的这样一个特征窗口:

For example, suppose that we have a first algorithm that extracts Part of
Speech (PoS) tags that we want to use as complementary tags for training
a sequence classifier (e.g. a chunker). The following dict could be
such a window of features extracted around the word 'sat' in the sentence
'The cat sat on the mat.'::

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

Tf means **term-frequency** while tf–idf means term-frequency times
**inverse document-frequency**:
:math:`\text{tf-idf(t,d)}=\text{tf(t,d)} \times \text{idf(t)}`.

Using the ``TfidfTransformer``'s default settings,
``TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)``
the term frequency, the number of times a term occurs in a given document,
is multiplied with idf component, which is computed as

:math:`\text{idf}(t) = log{\frac{1 + n_d}{1+\text{df}(d,t)}} + 1`,

where :math:`n_d` is the total number of documents, and :math:`\text{df}(d,t)`
is the number of documents that contain term :math:`t`. The resulting tf-idf
vectors are then normalized by the Euclidean norm:

:math:`v_{norm} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v{_1}^2 +
v{_2}^2 + \dots + v{_n}^2}}`.

This was originally a term weighting scheme developed for information retrieval
(as a ranking function for search engines results) that has also found good
use in document classification and clustering.

The following sections contain further explanations and examples that
illustrate how the tf-idfs are computed exactly and how the tf-idfs
computed in scikit-learn's :class:`TfidfTransformer`
and :class:`TfidfVectorizer` differ slightly from the standard textbook
notation that defines the idf as

:math:`\text{idf}(t) = log{\frac{n_d}{1+\text{df}(d,t)}}.`


In the :class:`TfidfTransformer` and :class:`TfidfVectorizer`
with ``smooth_idf=False``, the
"1" count is added to the idf instead of the idf's denominator:

:math:`\text{idf}(t) = log{\frac{n_d}{\text{df}(d,t)}} + 1`

This normalization is implemented by the :class:`TfidfTransformer`
class::

  >>> from sklearn.feature_extraction.text import TfidfTransformer
  >>> transformer = TfidfTransformer(smooth_idf=False)
  >>> transformer   # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  TfidfTransformer(norm=...'l2', smooth_idf=False, sublinear_tf=False,
                   use_idf=True)

Again please see the :ref:`reference documentation
<text_feature_extraction_ref>` for the details on all the parameters.

Let's take an example with the following counts. The first term is present
100% of the time hence not very interesting. The two other features only
in less than 50% of the time hence probably more representative of the
content of the documents::

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

Each row is normalized to have unit Euclidean norm:

:math:`v_{norm} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v{_1}^2 +
v{_2}^2 + \dots + v{_n}^2}}`

For example, we can compute the tf-idf of the first term in the first
document in the `counts` array as follows:

:math:`n_{d} = 6`

:math:`\text{df}(d, t)_{\text{term1}} = 6`

:math:`\text{idf}(d, t)_{\text{term1}} =
log \frac{n_d}{\text{df}(d, t)} + 1 = log(1)+1 = 1`

:math:`\text{tf-idf}_{\text{term1}} = \text{tf} \times \text{idf} = 3 \times 1 = 3`

Now, if we repeat this computation for the remaining 2 terms in the document,
we get

:math:`\text{tf-idf}_{\text{term2}} = 0 \times (log(6/1)+1) = 0`

:math:`\text{tf-idf}_{\text{term3}} = 1 \times (log(6/2)+1) \approx 2.0986`

and the vector of raw tf-idfs:

:math:`\text{tf-idf}_{\text{raw}} = [3, 0, 2.0986].`


Then, applying the Euclidean (L2) norm, we obtain the following tf-idfs
for document 1:

:math:`\frac{[3, 0, 2.0986]}{\sqrt{\big(3^2 + 0^2 + 2.0986^2\big)}}
= [ 0.819,  0,  0.573].`

Furthermore, the default parameter ``smooth_idf=True`` adds "1" to the numerator
and  denominator as if an extra document was seen containing every term in the
collection exactly once, which prevents zero divisions:

:math:`\text{idf}(t) = log{\frac{1 + n_d}{1+\text{df}(d,t)}} + 1`

Using this modification, the tf-idf of the third term in document 1 changes to
1.8473:

:math:`\text{tf-idf}_{\text{term3}} = 1 \times log(7/3)+1 \approx 1.8473`

And the L2-normalized tf-idf changes to

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

The weights of each
feature computed by the ``fit`` method call are stored in a model
attribute::

  >>> transformer.idf_                       # doctest: +ELLIPSIS
  array([1. ..., 2.25..., 1.84...])




As tf–idf is very often used for text features, there is also another
class called :class:`TfidfVectorizer` that combines all the options of
:class:`CountVectorizer` and :class:`TfidfTransformer` in a single model::

  >>> from sklearn.feature_extraction.text import TfidfVectorizer
  >>> vectorizer = TfidfVectorizer()
  >>> vectorizer.fit_transform(corpus)
  ...                                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <4x9 sparse matrix of type '<... 'numpy.float64'>'
      with 19 stored elements in Compressed Sparse ... format>

While the tf–idf normalization is often very useful, there might
be cases where the binary occurrence markers might offer better
features. This can be achieved by using the ``binary`` parameter
of :class:`CountVectorizer`. In particular, some estimators such as
:ref:`bernoulli_naive_bayes` explicitly model discrete boolean random
variables. Also, very short texts are likely to have noisy tf–idf values
while the binary occurrence info is more stable.

As usual the best way to adjust the feature extraction parameters
is to use a cross-validated grid search, for instance by pipelining the
feature extractor with a classifier:

 * :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py`


解码文本文件
-------------------
Text is made of characters, but files are made of bytes. These bytes represent
characters according to some *encoding*. To work with text files in Python,
their bytes must be *decoded* to a character set called Unicode.
Common encodings are ASCII, Latin-1 (Western Europe), KOI8-R (Russian)
and the universal encodings UTF-8 and UTF-16. Many others exist.

.. note::
    An encoding can also be called a 'character set',
    but this term is less accurate: several encodings can exist
    for a single character set.

The text feature extractors in scikit-learn know how to decode text files,
but only if you tell them what encoding the files are in.
The :class:`CountVectorizer` takes an ``encoding`` parameter for this purpose.
For modern text files, the correct encoding is probably UTF-8,
which is therefore the default (``encoding="utf-8"``).

If the text you are loading is not actually encoded with UTF-8, however,
you will get a ``UnicodeDecodeError``.
The vectorizers can be told to be silent about decoding errors
by setting the ``decode_error`` parameter to either ``"ignore"``
or ``"replace"``. See the documentation for the Python function
``bytes.decode`` for more details
(type ``help(bytes.decode)`` at the Python prompt).

If you are having trouble decoding text, here are some things to try:

- Find out what the actual encoding of the text is. The file might come
  with a header or README that tells you the encoding, or there might be some
  standard encoding you can assume based on where the text comes from.

- You may be able to find out what kind of encoding it is in general
  using the UNIX command ``file``. The Python ``chardet`` module comes with
  a script called ``chardetect.py`` that will guess the specific encoding,
  though you cannot rely on its guess being correct.

- You could try UTF-8 and disregard the errors. You can decode byte
  strings with ``bytes.decode(errors='replace')`` to replace all
  decoding errors with a meaningless character, or set
  ``decode_error='replace'`` in the vectorizer. This may damage the
  usefulness of your features.

- Real text may come from a variety of sources that may have used different
  encodings, or even be sloppily decoded in a different encoding than the
  one it was encoded with. This is common in text retrieved from the Web.
  The Python package `ftfy`_ can automatically sort out some classes of
  decoding errors, so you could try decoding the unknown text as ``latin-1``
  and then using ``ftfy`` to fix errors.

- If the text is in a mish-mash of encodings that is simply too hard to sort
  out (which is the case for the 20 Newsgroups dataset), you can fall back on
  a simple single-byte encoding such as ``latin-1``. Some text may display
  incorrectly, but at least the same sequence of bytes will always represent
  the same feature.

For example, the following snippet uses ``chardet``
(not shipped with scikit-learn, must be installed separately)
to figure out the encoding of three texts.
It then vectorizes the texts and prints the learned vocabulary.
The output is not shown here.

  >>> import chardet    # doctest: +SKIP
  >>> text1 = b"Sei mir gegr\xc3\xbc\xc3\x9ft mein Sauerkraut"
  >>> text2 = b"holdselig sind deine Ger\xfcche"
  >>> text3 = b"\xff\xfeA\x00u\x00f\x00 \x00F\x00l\x00\xfc\x00g\x00e\x00l\x00n\x00 \x00d\x00e\x00s\x00 \x00G\x00e\x00s\x00a\x00n\x00g\x00e\x00s\x00,\x00 \x00H\x00e\x00r\x00z\x00l\x00i\x00e\x00b\x00c\x00h\x00e\x00n\x00,\x00 \x00t\x00r\x00a\x00g\x00 \x00i\x00c\x00h\x00 \x00d\x00i\x00c\x00h\x00 \x00f\x00o\x00r\x00t\x00"
  >>> decoded = [x.decode(chardet.detect(x)['encoding'])
  ...            for x in (text1, text2, text3)]        # doctest: +SKIP
  >>> v = CountVectorizer().fit(decoded).vocabulary_    # doctest: +SKIP
  >>> for term in v: print(v)                           # doctest: +SKIP

(Depending on the version of ``chardet``, it might get the first one wrong.)

For an introduction to Unicode and character encodings in general,
see Joel Spolsky's `Absolute Minimum Every Software Developer Must Know
About Unicode <http://www.joelonsoftware.com/articles/Unicode.html>`_.

.. _`ftfy`: https://github.com/LuminosoInsight/python-ftfy


应用和案例
-------------------------

The bag of words representation is quite simplistic but surprisingly
useful in practice.

In particular in a **supervised setting** it can be successfully combined
with fast and scalable linear models to train **document classifiers**,
for instance:

 * :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`

In an **unsupervised setting** it can be used to group similar documents
together by applying clustering algorithms such as :ref:`k_means`:

  * :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`

Finally it is possible to discover the main topics of a corpus by
relaxing the hard assignment constraint of clustering, for instance by
using :ref:`NMF`:

  * :ref:`sphx_glr_auto_examples_applications_plot_topics_extraction_with_nmf_lda.py`


词袋表示法的局限性
----------------------------------------------

A collection of unigrams (what bag of words is) cannot capture phrases
and multi-word expressions, effectively disregarding any word order
dependence. Additionally, the bag of words model doesn't account for potential
misspellings or word derivations.

N-grams to the rescue! Instead of building a simple collection of
unigrams (n=1), one might prefer a collection of bigrams (n=2), where
occurrences of pairs of consecutive words are counted.

One might alternatively consider a collection of character n-grams, a
representation resilient against misspellings and derivations.

For example, let's say we're dealing with a corpus of two documents:
``['words', 'wprds']``. The second document contains a misspelling
of the word 'words'.
A simple bag of words representation would consider these two as
very distinct documents, differing in both of the two possible features.
A character 2-gram representation, however, would find the documents
matching in 4 out of 8 features, which may help the preferred classifier
decide better::

  >>> ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
  >>> counts = ngram_vectorizer.fit_transform(['words', 'wprds'])
  >>> ngram_vectorizer.get_feature_names() == (
  ...     [' w', 'ds', 'or', 'pr', 'rd', 's ', 'wo', 'wp'])
  True
  >>> counts.toarray().astype(int)
  array([[1, 1, 1, 0, 1, 1, 1, 0],
         [1, 1, 0, 1, 1, 1, 0, 1]])

In the above example, ``char_wb`` analyzer is used, which creates n-grams
only from characters inside word boundaries (padded with space on each
side). The ``char`` analyzer, alternatively, creates n-grams that
span across words::

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

The word boundaries-aware variant ``char_wb`` is especially interesting
for languages that use white-spaces for word separation as it generates
significantly less noisy features than the raw ``char`` variant in
that case. For such languages it can increase both the predictive
accuracy and convergence speed of classifiers trained using such
features while retaining the robustness with regards to misspellings and
word derivations.

While some local positioning information can be preserved by extracting
n-grams instead of individual words, bag of words and bag of n-grams
destroy most of the inner structure of the document and hence most of
the meaning carried by that internal structure.

In order to address the wider task of Natural Language Understanding,
the local structure of sentences and paragraphs should thus be taken
into account. Many such models will thus be casted as "Structured output"
problems which are currently outside of the scope of scikit-learn.


.. _hashing_vectorizer:

用散列技巧矢量化大型语料库
------------------------------------------------------

The above vectorization scheme is simple but the fact that it holds an **in-
memory mapping from the string tokens to the integer feature indices** (the
``vocabulary_`` attribute) causes several **problems when dealing with large
datasets**:

- the larger the corpus, the larger the vocabulary will grow and hence the
  memory use too,

- fitting requires the allocation of intermediate data structures
  of size proportional to that of the original dataset.

- building the word-mapping requires a full pass over the dataset hence it is
  not possible to fit text classifiers in a strictly online manner.

- pickling and un-pickling vectorizers with a large ``vocabulary_`` can be very
  slow (typically much slower than pickling / un-pickling flat data structures
  such as a NumPy array of the same size),

- it is not easily possible to split the vectorization work into concurrent sub
  tasks as the ``vocabulary_`` attribute would have to be a shared state with a
  fine grained synchronization barrier: the mapping from token string to
  feature index is dependent on ordering of the first occurrence of each token
  hence would have to be shared, potentially harming the concurrent workers'
  performance to the point of making them slower than the sequential variant.

It is possible to overcome those limitations by combining the "hashing trick"
(:ref:`Feature_hashing`) implemented by the
:class:`sklearn.feature_extraction.FeatureHasher` class and the text
preprocessing and tokenization features of the :class:`CountVectorizer`.

This combination is implementing in :class:`HashingVectorizer`,
a transformer class that is mostly API compatible with :class:`CountVectorizer`.
:class:`HashingVectorizer` is stateless,
meaning that you don't have to call ``fit`` on it::

  >>> from sklearn.feature_extraction.text import HashingVectorizer
  >>> hv = HashingVectorizer(n_features=10)
  >>> hv.transform(corpus)
  ...                                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <4x10 sparse matrix of type '<... 'numpy.float64'>'
      with 16 stored elements in Compressed Sparse ... format>

You can see that 16 non-zero feature tokens were extracted in the vector
output: this is less than the 19 non-zeros extracted previously by the
:class:`CountVectorizer` on the same toy corpus. The discrepancy comes from
hash function collisions because of the low value of the ``n_features`` parameter.

In a real world setting, the ``n_features`` parameter can be left to its
default value of ``2 ** 20`` (roughly one million possible features). If memory
or downstream models size is an issue selecting a lower value such as ``2 **
18`` might help without introducing too many additional collisions on typical
text classification tasks.

Note that the dimensionality does not affect the CPU training time of
algorithms which operate on CSR matrices (``LinearSVC(dual=True)``,
``Perceptron``, ``SGDClassifier``, ``PassiveAggressive``) but it does for
algorithms that work with CSC matrices (``LinearSVC(dual=False)``, ``Lasso()``,
etc).

Let's try again with the default setting::

  >>> hv = HashingVectorizer()
  >>> hv.transform(corpus)
  ...                               # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
  <4x1048576 sparse matrix of type '<... 'numpy.float64'>'
      with 19 stored elements in Compressed Sparse ... format>

We no longer get the collisions, but this comes at the expense of a much larger
dimensionality of the output space.
Of course, other terms than the 19 used here
might still collide with each other.

The :class:`HashingVectorizer` also comes with the following limitations:

- it is not possible to invert the model (no ``inverse_transform`` method),
  nor to access the original string representation of the features,
  because of the one-way nature of the hash function that performs the mapping.

- it does not provide IDF weighting as that would introduce statefulness in the
  model. A :class:`TfidfTransformer` can be appended to it in a pipeline if
  required.

使用 HashingVectorizer 执行核外scaling 
------------------------------------------------------

An interesting development of using a :class:`HashingVectorizer` is the ability
to perform `out-of-core`_ scaling. This means that we can learn from data that
does not fit into the computer's main memory.

.. _out-of-core: https://en.wikipedia.org/wiki/Out-of-core_algorithm

A strategy to implement out-of-core scaling is to stream data to the estimator
in mini-batches. Each mini-batch is vectorized using :class:`HashingVectorizer`
so as to guarantee that the input space of the estimator has always the same
dimensionality. The amount of memory used at any time is thus bounded by the
size of a mini-batch. Although there is no limit to the amount of data that can
be ingested using such an approach, from a practical point of view the learning
time is often limited by the CPU time one wants to spend on the task.

For a full-fledged example of out-of-core scaling in a text classification
task see :ref:`sphx_glr_auto_examples_applications_plot_out_of_core_classification.py`.

Customizing the vectorizer classes
----------------------------------

It is possible to customize the behavior by passing a callable
to the vectorizer constructor::

  >>> def my_tokenizer(s):
  ...     return s.split()
  ...
  >>> vectorizer = CountVectorizer(tokenizer=my_tokenizer)
  >>> vectorizer.build_analyzer()(u"Some... punctuation!") == (
  ...     ['some...', 'punctuation!'])
  True

In particular we name:

  * ``preprocessor``: a callable that takes an entire document as input (as a
    single string), and returns a possibly transformed version of the document,
    still as an entire string. This can be used to remove HTML tags, lowercase
    the entire document, etc.

  * ``tokenizer``: a callable that takes the output from the preprocessor
    and splits it into tokens, then returns a list of these.

  * ``analyzer``: a callable that replaces the preprocessor and tokenizer.
    The default analyzers all call the preprocessor and tokenizer, but custom
    analyzers will skip this. N-gram extraction and stop word filtering take
    place at the analyzer level, so a custom analyzer may have to reproduce
    these steps.

(Lucene users might recognize these names, but be aware that scikit-learn
concepts may not map one-to-one onto Lucene concepts.)

To make the preprocessor, tokenizer and analyzers aware of the model
parameters it is possible to derive from the class and override the
``build_preprocessor``, ``build_tokenizer`` and ``build_analyzer``
factory methods instead of passing custom functions.

一些经验和技巧:

  * If documents are pre-tokenized by an external package, then store them in
    files (or strings) with the tokens separated by whitespace and pass
    ``analyzer=str.split``
  * Fancy token-level analysis such as stemming, lemmatizing, compound
    splitting, filtering based on part-of-speech, etc. are not included in the
    scikit-learn codebase, but can be added by customizing either the
    tokenizer or the analyzer.
    Here's a ``CountVectorizer`` with a tokenizer and lemmatizer using
    `NLTK <http://www.nltk.org>`_::

        >>> from nltk import word_tokenize          # doctest: +SKIP
        >>> from nltk.stem import WordNetLemmatizer # doctest: +SKIP
        >>> class LemmaTokenizer(object):
        ...     def __init__(self):
        ...         self.wnl = WordNetLemmatizer()
        ...     def __call__(self, doc):
        ...         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        ...
        >>> vect = CountVectorizer(tokenizer=LemmaTokenizer())  # doctest: +SKIP

    (Note that this will not filter out punctuation.)


    The following example will, for instance, transform some British spelling
    to American spelling::

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

    for other styles of preprocessing; examples include stemming, lemmatization,
    or normalizing numerical tokens, with the latter illustrated in:

     * :ref:`sphx_glr_auto_examples_bicluster_plot_bicluster_newsgroups.py`


Customizing the vectorizer can also be useful when handling Asian languages
that do not use an explicit word separator such as whitespace.

.. _image_feature_extraction:

图像特征提取
========================

.. currentmodule:: sklearn.feature_extraction.image

图像块提取
----------------

The :func:`extract_patches_2d` function extracts patches from an image stored
as a two-dimensional array, or three-dimensional with color information along
the third axis. For rebuilding an image from all its patches, use
:func:`reconstruct_from_patches_2d`. For example let use generate a 4x4 pixel
picture with 3 color channels (e.g. in RGB format)::

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

Let us now try to reconstruct the original image from the patches by averaging
on overlapping areas::

    >>> reconstructed = image.reconstruct_from_patches_2d(patches, (4, 4, 3))
    >>> np.testing.assert_array_equal(one_image, reconstructed)

The :class:`PatchExtractor` class works in the same way as
:func:`extract_patches_2d`, only it supports multiple images as input. It is
implemented as an estimator, so it can be used in pipelines. See::

    >>> five_images = np.arange(5 * 4 * 4 * 3).reshape(5, 4, 4, 3)
    >>> patches = image.PatchExtractor((2, 2)).transform(five_images)
    >>> patches.shape
    (45, 2, 2, 3)

图像的连接图
-------------------------------

Several estimators in the scikit-learn can use connectivity information between
features or samples. For instance Ward clustering
(:ref:`hierarchical_clustering`) can cluster together only neighboring pixels
of an image, thus forming contiguous patches:

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_coin_ward_segmentation_001.png
   :target: ../auto_examples/cluster/plot_coin_ward_segmentation.html
   :align: center
   :scale: 40

For this purpose, the estimators use a 'connectivity' matrix, giving
which samples are connected.

The function :func:`img_to_graph` returns such a matrix from a 2D or 3D
image. Similarly, :func:`grid_to_graph` build a connectivity matrix for
images given the shape of these image.

These matrices can be used to impose connectivity in estimators that use
connectivity information, such as Ward clustering
(:ref:`hierarchical_clustering`), but also to build precomputed kernels,
or similarity matrices.

.. note:: **案例**

   * :ref:`sphx_glr_auto_examples_cluster_plot_coin_ward_segmentation.py`

   * :ref:`sphx_glr_auto_examples_cluster_plot_segmentation_toy.py`

   * :ref:`sphx_glr_auto_examples_cluster_plot_feature_agglomeration_vs_univariate_selection.py`
