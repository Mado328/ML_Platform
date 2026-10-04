import pytest

from ml_platform.models.classifiers.knn import KNN

def test_invalid_initialization():
    with pytest.raises(ValueError):
        knn = KNN(n_neighbors=2.5)
    with pytest.raises(ValueError):
        knn = KNN(n_neighbors=0)
    with pytest.raises(ValueError):
        knn = KNN(n_neighbors=-1)
    with pytest.raises(ValueError):
        knn = KNN(n_neighbors=True)

def test_set_params_invalid():
    knn = KNN(n_neighbors=3)
    with pytest.raises(ValueError):
        knn.set_params(unknown_param=4)
    with pytest.raises(ValueError):
        knn.set_params(n_neighbors=0)
    with pytest.raises(ValueError):
        knn.set_params(n_neighbors=-1)
    with pytest.raises(ValueError):
        knn.set_params(n_neighbors=True)
    with pytest.raises(ValueError):
        knn.set_params(n_neighbors=5.0)

def test_get_params():
    knn = KNN(n_neighbors=3)
    params = knn.get_params()
    assert params == {"n_neighbors": 3}

def test_fit_predict():
    knn = KNN()
    # Load a small sample dataset for testing
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    
    knn.fit(X, y)
    assert knn.predict(X) == [0, 0]

    knn.set_params(n_neighbors=1)
    knn.fit(X, y)
    predictions_proba = knn.predict_proba(X)
    assert predictions_proba == [[1.0, 0.0], [0.0, 1.0]]
    assert round(sum(p for l in predictions_proba for p in l) / len(predictions_proba)) == 1

def test_fit_with_X_and_y_length_mismatch():
    knn = KNN()
    X = [[0, 0], [1, 1]]
    y = [0]  # Mismatched length
    with pytest.raises(ValueError):
        knn.fit(X, y)

def test_fit_with_empty_Xy():
    knn = KNN()
    X = []
    y = []
    with pytest.raises(ValueError):
        knn.fit(X, y)