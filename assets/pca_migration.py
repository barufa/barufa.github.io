import time
import faiss
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


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

X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=3_000)

skl_pca = PCA(n_components=32, random_state=42)
skl_pca.fit(X_train)

faiss_pca = sklearn_pca_to_faiss(skl_pca)

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
print(f"faiss.apply_py   : {(t3-t2):.3f}s  | {(X.shape[0]/(t3-t2)):.0f} vec/s")
print(f"Speedup: {((t1-t0)/(t3-t2)):.1f}x")
