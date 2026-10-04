import math
import re
from collections import Counter, defaultdict
from typing import Any

from .base_classifiers import BaseClassifier


def tokenize(x):
    """Разбивает строку на слова и приводит их к нижнему регистру."""
    # \w+ находит последовательности букв и цифр, в том числе кириллицу.
    # lower() позволяет считать «Приз» и «приз» одним словом.

    return re.findall(r"\w+", x.lower())
    

class MultinomialNB(BaseClassifier):
    def __init__(self, alpha=1.0):
        self._is_fitted = False
        self.alpha = 1.0
        self.set_params(alpha=alpha)

    def fit(self, X, y):
        """Запоминает статистику слов и классов по обучающим примерам."""

        if self.is_fitted:
            raise RuntimeError("MultinomialNB classifier has already been fitted")
        if not X:
            raise ValueError("X must contain at least one training sample")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")
        # Список классов, например ["обычное", "спам"].
        self.classes = sorted(set(y))

        # Сколько обучающих сообщений относится к каждому классу.
        # Нужно для оценки априорной вероятности класса P(класс).
        self.doc_counts = Counter(y)

        # Для каждого класса храним счётчик слов.
        # Например: word_counts["спам"]["приз"] — сколько раз
        # слово «приз» встретилось в сообщениях класса «спам».
        self.word_counts = defaultdict(Counter)

        # Общее количество слов в сообщениях каждого класса.
        # Используется при оценке P(слово | класс).
        self.total_words = Counter()

        # Словарь всех слов, встретившихся в обучающих сообщениях.
        self.vocabulary = set()

        # Обрабатываем каждое сообщение вместе с его правильным классом.
        for x, label in zip(X, y):
            words = tokenize(x)

            # Увеличиваем счётчики слов для соответствующего класса.
            self.word_counts[label].update(words)

            # Запоминаем общее число слов в сообщениях этого класса.
            self.total_words[label] += len(words)

            # Добавляем слова сообщения в общий словарь.
            self.vocabulary.update(words)

        # Специальная категория для слов, которых не было при обучении.
        self.unknown_token = "<UNK>"

        # Добавляем её в словарь, чтобы неизвестные слова тоже
        # могли получить ненулевую вероятность благодаря сглаживанию.
        self.vocabulary.add(self.unknown_token)

        # Размер словаря нужен в формуле сглаживания Лапласа.
        self.vocab_size = len(self.vocabulary)

        # Общее число обучающих сообщений — для априорной вероятности класса.
        self.total_docs = len(X)

        self._is_fitted = True

        return self

    def predict_proba(self, X):
        """Возвращает оценку вероятности каждого класса для сообщения."""

        if not self.is_fitted:
            raise RuntimeError("MultinomialNB classifier has not been fitted")
        # Разбиваем сообщение на слова.
        words = tokenize(X)

        # Заменяем каждое незнакомое слово на специальный токен.
        words = [
            word if word in self.vocabulary else self.unknown_token
            for word in words
        ]

        # Здесь будут оценки логарифма вероятности для каждого класса.
        # Используем логарифмы, чтобы произведение множества маленьких
        # вероятностей не округлилось компьютером до нуля.
        log_scores = {}

        for label in self.classes:
            # Априорная вероятность класса:
            # доля обучающих сообщений, принадлежащих этому классу.
            prior = self.doc_counts[label] / self.total_docs

            # Начинаем оценку с логарифма априорной вероятности.
            score = math.log(prior)

            # Добавляем вклад каждого слова сообщения.
            for word in words:
                # Сколько раз слово встречалось в обучающих сообщениях класса.
                count = self.word_counts[label][word]

                # Оцениваем вероятность слова при условии класса:
                # (частота слова + alpha) /
                # (все слова класса + alpha * размер словаря).
                word_probability = (
                    (count + self.alpha)
                    / (self.total_words[label] + self.alpha * self.vocab_size)
                )

                # По наивному Байесу считаем слова независимыми
                # и складываем логарифмы их вероятностей.
                score += math.log(word_probability)

            # Сохраняем итоговую оценку для класса.
            log_scores[label] = score

        # Переводим логарифмические оценки в обычные вероятности.
        # Вычитание максимума помогает избежать переполнения при exp().
        max_score = max(log_scores.values())
        exp_scores = {
            label: math.exp(score - max_score)
            for label, score in log_scores.items()
        }

        # Нормализуем значения, чтобы сумма вероятностей стала равна 1.
        total = sum(exp_scores.values())

        return {
            label: score / total
            for label, score in exp_scores.items()
        }

    def predict(self, X):
        """Возвращает класс с наибольшей оценённой вероятностью."""

        # Получаем оценки вероятностей для всех классов.
        probabilities = self.predict_proba(X)

        # Выбираем класс, у которого оценка максимальна.
        return max(probabilities, key=lambda label: probabilities[label])

    def get_params(self) -> dict[str, Any]:

        return {'alpha': self.alpha}

    def set_params(self, **params: Any) -> "MultinomialNB":
        # get_params содержит только допустимые гиперпараметры.

        allowed_params = self.get_params()
        unknown_params = set(params) - allowed_params.keys()
    
        # Неизвестные имена — ошибка API, а не повод создать новый атрибут.
        if unknown_params:
            names = ", ".join(sorted(unknown_params))
            raise ValueError(f"Unknown parameter(s): {names}")
    
        # Пустой вызов — без изменений.
        if not params:
            return self
    
        # Проверяем значение до изменения объекта.
        alpha = params.get("alpha", self.alpha)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a finite number greater than 0")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite number greater than 0")
    
        # Меняем состояние только после успешной проверки.
        if alpha != self.alpha:
            self.alpha = float(alpha)
            self._is_fitted = False
    
        return self

    @property
    def is_fitted(self) -> bool:
        
        return self._is_fitted

    @property
    def classes_(self) -> Any:
        if not self.is_fitted:
            raise RuntimeError("MultinomialNB classifier has not been fitted")
        return self.classes
