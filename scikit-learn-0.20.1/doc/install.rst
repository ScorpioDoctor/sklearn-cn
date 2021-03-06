.. _installation-instructions:

=======================
安装 scikit-learn
=======================

.. note::

    If you wish to contribute to the project, it's recommended you
    :ref:`install the latest development version<install_bleeding_edge>`.


.. _install_official_release:

安装最新版
=============================

Scikit-learn requires:

- Python (>= 2.7 or >= 3.4),
- NumPy (>= 1.8.2),
- SciPy (>= 0.13.3).


.. warning::

    Scikit-learn 0.20 is the last version to support Python 2.7 and Python 3.4.
    Scikit-learn 0.21 will require Python 3.5 or newer.

If you already have a working installation of numpy and scipy,
the easiest way to install scikit-learn is using ``pip`` ::

    pip install -U scikit-learn

or ``conda``::

    conda install scikit-learn

If you have not installed NumPy or SciPy yet, you can also install these using
conda or pip. When using pip, please ensure that *binary wheels* are used,
and NumPy and SciPy are not recompiled from source, which can happen when using
particular configurations of operating system and hardware (such as Linux on
a Raspberry Pi). 
Building numpy and scipy from source can be complex (especially on Windows) and
requires careful configuration to ensure that they link against an optimized
implementation of linear algebra routines.
Instead, use a third-party distribution as described below.

If you must install scikit-learn and its dependencies with pip, you can install
it as ``scikit-learn[alldeps]``. The most common use case for this is in a
``requirements.txt`` file used as part of an automated build process for a PaaS
application or a Docker image. This option is not intended for manual
installation from the command line.

.. note::

   For installing on PyPy, PyPy3-v5.10+, Numpy 1.14.0+, and scipy 1.1.0+
   are required.


For installation instructions for more distributions see
:ref:`other distributions <install_by_distribution>`.
For compiling the development version from source, or building the package
if no distribution is available for your architecture, see the
:ref:`advanced-installation`.

安装第三方发布版
==========================
If you don't already have a python installation with numpy and scipy, we
recommend to install either via your package manager or via a python bundle.
These come with numpy, scipy, scikit-learn, matplotlib and many other helpful
scientific and data processing libraries.

Available options are:

Canopy 和 Anaconda 可以用于所有平台
-----------------------------------------------

`Canopy
<https://www.enthought.com/products/canopy>`_ and `Anaconda
<https://www.anaconda.com/download>`_ both ship a recent
version of scikit-learn, in addition to a large set of scientific python
library for Windows, Mac OSX and Linux.

Anaconda offers scikit-learn as part of its free distribution.


.. warning::

    To upgrade or uninstall scikit-learn installed with Anaconda
    or ``conda`` you **should not use the pip command**. Instead:

    To upgrade ``scikit-learn``::

        conda update scikit-learn

    To uninstall ``scikit-learn``::

        conda remove scikit-learn

    Upgrading with ``pip install -U scikit-learn`` or uninstalling
    ``pip uninstall scikit-learn`` is likely fail to properly remove files
    installed by the ``conda`` command.

    pip upgrade and uninstall operations only work on packages installed
    via ``pip install``.


WinPython 可以在Windows平台上使用
-----------------------

The `WinPython <https://winpython.github.io/>`_ project distributes
scikit-learn as an additional plugin.

