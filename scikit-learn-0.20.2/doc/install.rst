.. _installation-instructions:

=======================
安装 scikit-learn
=======================

.. note::

    如果你想对这个工程做出自己的贡献, 推荐你阅读:
    :ref:`install the latest development version<install_bleeding_edge>`.


.. _install_official_release:

安装最新版本
=============================

Scikit-learn 需要以下依赖项:

- Python (>= 2.7 or >= 3.4),
- NumPy (>= 1.8.2),
- SciPy (>= 0.13.3).


.. warning::

    Scikit-learn 0.20 是支持 Python 2.7 和 Python 3.4 的最后一个版本。
    Scikit-learn 0.21 将需要 Python 3.5 或者 以上.

如果你已经安装了 numpy 和 scipy,
安装 scikit-learn 最简单的方法就是使用 ``pip``::

    pip install -U scikit-learn

或者 ``conda``::

    conda install scikit-learn

如果尚未安装NumPy或SciPy，也可以使用conda或pip安装这些文件。使用pip时，
请确保使用二进制的wheel，NumPy和SciPy不从源代码重新编译，这可能发生在使用特定的操作系统和硬件配置(如Raspberry PI上的Linux)时。
从源构建numpy和spy可能很复杂(特别是在Windows上)，并且需要进行仔细的配置，
以确保它们与线性代数例程的优化实现相关联。相反，使用第三方发行版，如下所述。

如果您必须使用pip安装Scikit-Learn及其依赖项，则可以将其安装为 ``scikit-learn[alldeps]`` 。
这方面最常见的用例是一个 ``requments.txt`` 文件，作为PaaS应用程序或Docker映像的自动构建过程的一部分。
此选项不适合从命令行手动安装。

.. note::

   For installing on PyPy, PyPy3-v5.10+, Numpy 1.14.0+, and scipy 1.1.0+
   are required.


有关更多发行版的安装说明，请参见 :ref:`其他发行版 <install_by_distribution>` 。
有关从源代码编译开发版本，或者如果您的体系结构没有可用的发行版，请参见“高级安装说明(:ref:`advanced-installation`)”


第三方发布版
==========================
如果您还没有安装带有numpy和sciy的python，我们建议您通过package manager或python bundle进行安装。
这些都有numpy，ciply，sckit-Learn，matplotlib和许多其他有用的科学和数据处理库。

可用的选项有:

Canopy和Anaconda可用于所有被支持的平台
-----------------------------------------------

`Canopy <https://www.enthought.com/products/canopy>`_ 和 `Anaconda <https://www.anaconda.com/download>`_ 
都自带了最新版的 scikit-learn, 另外还有大量的适用于Windows, Mac OSX and Linux的python科学计算库。

Anaconda 将 scikit-learn 作为其免费发行的一部分。


.. warning::

    要更新或卸载通过 Anaconda 或 ``conda`` 安装的 scikit-learn，你  **不能使用 pip uninstall xxx 命令** 。 
    应该用下面的方法:

    去更新 ``scikit-learn``::

        conda update scikit-learn

    去卸载 ``scikit-learn``::

        conda remove scikit-learn

    Upgrading with ``pip install -U scikit-learn`` or uninstalling
    ``pip uninstall scikit-learn`` is likely fail to properly remove files
    installed by the ``conda`` command.

    pip 更新或卸载操作只能用来管理那些使用 ``pip install`` 命令安装的package。


WinPython可在Windows平台上使用
-----------------------

`WinPython <https://winpython.github.io/>`_ 工程将 scikit-learn 作为一个额外的插件。

