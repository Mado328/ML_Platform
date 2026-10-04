import pytest

import ml_platform.models.classifiers.nb as nb

def test_invalid_initialization():
    with pytest.raises(ValueError):
        classifier = nb.MultinomialNB(alpha=-1.0)
    with pytest.raises(ValueError):
        classifier = nb.MultinomialNB(alpha=0.0)
    with pytest.raises(ValueError):
        classifier = nb.MultinomialNB(alpha="invalid")
    with pytest.raises(ValueError):
        classifier = nb.MultinomialNB(alpha=True)
    with pytest.raises(ValueError):
        classifier = nb.MultinomialNB(alpha=float("inf"))
    with pytest.raises(ValueError):
        classifier = nb.MultinomialNB(alpha=float("nan"))

def test_set_params_invalid():
    classifier = nb.MultinomialNB(alpha=1.0)
    with pytest.raises(ValueError):
        classifier.set_params(unknown_param=4)
    with pytest.raises(ValueError):
        classifier.set_params(alpha=-1.0)
    with pytest.raises(ValueError):
        classifier.set_params(alpha=0.0)
    with pytest.raises(ValueError):
        classifier.set_params(alpha="invalid")
    with pytest.raises(ValueError):
        classifier.set_params(alpha=True)
    with pytest.raises(ValueError):
        classifier.set_params(alpha=float("inf"))
    with pytest.raises(ValueError):
        classifier.set_params(alpha=float("nan"))

def test_get_params():
    classifier = nb.MultinomialNB(alpha=1.0)
    params = classifier.get_params()
    assert params == {"alpha": 1.0}

def test_fit_predict():
    classifier = nb.MultinomialNB(alpha=1.0)
    # Load a small sample dataset for testing
    X = ["spam message", "ham message"]
    y = ["spam", "ham"]

    classifier.fit(X, y)
    predictions = []
    for message in ["spam message", "ham message"]:
        predictions.append(classifier.predict(message))

    assert predictions == ["spam", "ham"]

    classifier.set_params(alpha=0.5)
    classifier.fit(X, y)
    predictions_proba = []
    for message in ["spam message", "ham message"]:
        predictions_proba.append(classifier.predict_proba(message))

    assert round(sum(p for d in predictions_proba for p in d.values()) / len(predictions_proba)) == 1

def test_set_params_updates_alpha_and_invalidates_fitted_model():
    classifier = nb.MultinomialNB(alpha=1.0)
    classifier.fit(["spam message", "ham message"], ["spam", "ham"])

    result = classifier.set_params(alpha=0.5)

    assert result is classifier
    assert classifier.get_params() == {"alpha": 0.5}
    assert classifier.is_fitted is False

    with pytest.raises(RuntimeError):
        classifier.predict("spam message")