---
layout: post
comments: true
title: "From scikit-learn to Faiss: Migrating PCA for Scalable Vector Search"
excerpt: "Keep sklearn for training and validation, while leveraging Faiss for high-performance production inference."
date: 2025-07-19
category: "mlops"
tags: machine-learning, mlops, deployment
---

## Why using Faiss ##

Faiss is a high‑performance library for vector similarity search and related primitives (clustering, compression, linear transforms like PCA). t scales to millions–billions of vectors on CPU and GPU and it is a much faster implementation of PCA. In practice this reduces memory, latency, and Python overhead.

### Why migrate PCA to Faiss? ###

If you’re already using scikit-learn for training, why switch to Faiss for deployment?

* Training PCA in sklearn is convenient, but for the deployment implementation is slow.
* Faiss offers faster, more efficient kernels for applying PCA at scale.
* You can migrate a trained sklearn.PCA to a faiss.PCAMatrix without retraining.

## PCA refresher ##

PCA (Principal Component Analysis) is a linear dimensionality reduction technique. It projects data into a lower-dimensional space using the eigenvectors of the covariance matrix. You can check [this class](https://youtu.be/dhK8nbtii6I?si=rEa2z5YDaGERLTfy) for a detail exaplanation.

### Sklearn ###

We’ll focus on the essential operation of PCA: projecting vectors using `transform()`.

Given :
* X as the input data
* skl_pca as the trained PCA object from sklearn

You can project X into the PCA-transformed space like this:

    X_transformed = X @ skl_pca.components_.T

If whitening was applied during PCA fitting, you’ll also need to scale the output:

    scale = xp.sqrt(skl_pca.explained_variance_)
    min_scale = xp.finfo(scale.dtype).eps
    scale[scale < min_scale] = min_scale
    X_transformed /= scale

For reference, see the [official implementation](https://github.com/scikit-learn/scikit-learn/blob/c5497b7f7/sklearn/decomposition/_base.py#L116).

### Faiss ###

In Faiss, after training a `PCAMatrix`, the transformation looks slightly different:

     X_transformed = X @ faiss_pca.A.T + faiss_pca.b

Here, `A` is the components matrix, and `b` is a bias vector (equivalent to applying the mean shift from `sklearn`).

## Migrating from `sklearn.PCA` to `Faiss` ##

To migrate from a trained `sklearn.PCA` model to a `faiss.PCAMatrix`, you need to extract:
* `A`: the transformed components matrix 
* `b`: the bias vector to match sklearn’s behavior

Depending on whether whitening is used:

Without whitening:
    A = W
    b = -mean @ A.T   # == -mean @ W.T

With `whiten=True`:
    A = W / sqrt(λ)[:, None]   # escala cada fila de W
    b = -mean @ A.T

After these definitions we can get:
    X @ A.T + b  ==  sklearn.PCA.transform(X)

### Code ###

First, lets create a small PCA using the  USPS digits datasets.

```
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=3_000)

skl_pca = PCA(n_components=32, random_state=42)
skl_pca.fit(X_train)
```

Great, we have our PCA from sklearn trained. Now, we are going to try to migrate to faiss:

```
import sklearn
import faiss
import numpy as np


def sklearn_pca_to_faiss(skl_pca) -> faiss.PCAMatrix:
    d_in = skl_pca.components_.shape[1]
    d_out = skl_pca.n_components_

    # Build A: rows are components; include whitening if requested
    if getattr(skl_pca, "whiten", False):
        scale = np.sqrt(skl_pca.explained_variance_)[:, None]
        A = (skl_pca.components_ / scale).astype(np.float32)
    else:
        A = skl_pca.components_.astype(np.float32)

    faiss_pca = faiss.PCAMatrix(d_in, d_out, 0.0, False)  # eigen_power handled manually
    faiss.copy_array_to_vector(A.reshape(-1), faiss_pca.A)

    mean = skl_pca.mean_.astype(np.float32)
    faiss.copy_array_to_vector(mean.reshape(-1), faiss_pca.mean)

    # Choose bias so that X @ A^T + b == (X - mean) @ A^T
    b = -mean @ A.T  # shape (d_out,)
    faiss.copy_array_to_vector(b.reshape(-1), faiss_pca.b)

    faiss_pca.is_trained = True
    return faiss_pca

faiss_pca = sklearn_pca_to_faiss(skl_pca)
```

Important: Use Faiss’s `copy_array_to_vector` utility to load arrays into Faiss structures. See [this file](https://github.com/facebookresearch/faiss/blob/514b44fca8542bafe8640adcbf1cccce1900f74c/faiss/python/array_conversions.py#L128) for implementation details.

### Validation ###

Always validate that the conversion is correct:

```
import numpy as np
import faiss
import sklearn
import time

X = np.random.randn(1_000_000, faiss_pca.d_in).astype(np.float32)
np.testing.assert_allclose(skl_pca.transform(X), faiss_pca.apply_py(X), atol=1e-5)
np.testing.assert_allclose(skl_pca.transform(X_train), faiss_pca.apply_py(X_train), atol=1e-5)
np.testing.assert_allclose(skl_pca.transform(X_test), faiss_pca.apply_py(X_test), atol=1e-5)
print("OK: sklearn == faiss")

# sklearn
t0 = time.perf_counter(); _ = skl_pca.transform(X); t1 = time.perf_counter()
# faiss
t2 = time.perf_counter(); _ = faiss_pca.apply_py(X); t3 = time.perf_counter()

print(f"sklearn.transform: {(t1-t0):.3f}s  | {(X.shape[0]/(t1-t0)):.0f} vec/s")
print(f"faiss.apply_py  : {(t3-t2):.3f}s  | {(X.shape[0]/(t3-t2)):.0f} vec/s")
print(f"Speedup: {((t1-t0)/(t3-t2)):.1f}x")
```

Here we can see that a 1.2x speedup was achieve. See the complete code [here]().

## Conclusion ##

Migrating from `scikit-learn` to `Faiss` for PCA application is a straightforward optimization with real-world impact. You can keep sklearn for training and validation, then deploy the exact same projection using Faiss—boosting inference performance without retraining.

This method is simple, deterministic, and production-ready. And with just a few lines of code, you bridge the gap between experimentation and scalable deployment.
