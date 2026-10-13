from .models.classifiers.nb import MultinomialNB


def main() -> None:
    """Run a small Naive Bayes classifier demonstration."""
    xs = [
        "вы выиграли приз нажмите сюда",
        "получите бесплатный подарок",
        "срочное предложение выигрыш приз",
        "давай встретимся сегодня вечером",
        "напомни купить молоко",
        "спасибо за помощь с проектом",
    ]
    labels = ["спам", "спам", "спам", "обычное", "обычное", "обычное"]

    model = MultinomialNB()
    model.fit(xs, labels)
    print(model.get_params())

    for message in (
        "вы получили бесплатный приз",
        "давай сегодня обсудим проект",
    ):
        print(f"Сообщение: {message}")
        print(f"Класс: {model.predict(message)}")
        # These are model scores, not necessarily calibrated confidence values.
        print(f"Вероятности: {model.predict_proba(message)}")
        print()
