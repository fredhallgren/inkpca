
# Incremental kernel PCA

Incremental kernel PCA based on rank-one updates to the eigendecomposition of
the kernel matrix, which takes into account the changing mean of the
covariance matrix for additional data points.

This is the most efficient algorithm for incremental kernel PCA currently
available. See our paper for further details, available at [this link](https://arxiv.org/abs/1802.00043).

We also create the first incremental algorithm for the Nyström approximation
to the kernel matrix.

The algorithm is located in the file ``incremental_kpca.py``. The rank-one
update algorithm we apply is located in ``eigen_update.py``. Experiments
on two datasets from the UCI Machine Learning repository are included.



Requirements
------------

* Python 3.6

* Numpy

* Scipy

* nose (for tests)


Running
-------

To run the experiments, do from this folder

    cd inkpca
    python experiments.py

Tested on Ubuntu 20.04
