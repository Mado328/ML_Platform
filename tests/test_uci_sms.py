from pathlib import Path

import pandas as pd
import pytest

from ml_platform.data.uci_sms import load_uci_sms


def write_tsv(tmp_path: Path, content: str) -> Path:
    """Создаёт небольшой временный TSV-файл для теста загрузчика."""
    # Фикстура tmp_path предоставляет отдельную временную папку для каждого теста.
    path = tmp_path / "sample.tsv"

    # Кодировка UTF-8 соответствует загруженному набору UCI.
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_headerless_tsv_and_preserves_quotes(tmp_path: Path) -> None:
    # В файле UCI нет заголовка. Несбалансированная кавычка проверяет, что
    # кавычки в тексте сообщений воспринимаются как обычные символы, а не синтаксис CSV.
    source = write_tsv(
        tmp_path,
        'ham\tHe said "hello\nspam\t"Claim your prize\n',
    )

    data = load_uci_sms(source)

    # Проверяем, что первая строка прочитана как данные и загрузчик вернул
    # DataFrame с ожидаемыми двумя столбцами.
    assert list(data.columns) == ["label", "text"]
    assert len(data) == 2
    assert data.iloc[0]["label"] == "ham"
    assert data.iloc[0]["text"] == 'He said "hello'
    assert data.iloc[1]["text"] == '"Claim your prize'
    assert isinstance(data["label"].dtype, pd.StringDtype)


def test_preserves_duplicate_messages(tmp_path: Path) -> None:
    # Загрузчик сохраняет дубликаты: политика их обработки будет определена
    # отдельно при подготовке данных для обучения и оценки модели.
    source = write_tsv(
        tmp_path,
        "ham\tSame message\nham\tSame message\nspam\tDifferent message\n",
    )

    data = load_uci_sms(source)

    assert len(data) == 3
    assert data["text"].duplicated().sum() == 1


def test_rejects_unexpected_label(tmp_path: Path) -> None:
    # Обе допустимые метки присутствуют, поэтому тест проверяет именно
    # обработку неожиданной метки.
    source = write_tsv(
        tmp_path,
        "ham\tNormal message\nspam\tPrize message\nother\tUnknown class\n",
    )

    with pytest.raises(ValueError, match="Invalid labels"):
        load_uci_sms(source)


def test_rejects_missing_class(tmp_path: Path) -> None:
    # Набор только с сообщениями ham неполон для бинарной классификации UCI,
    # поэтому загрузчик должен отклонить его.
    source = write_tsv(tmp_path, "ham\tFirst message\nham\tSecond message\n")

    with pytest.raises(ValueError, match="Invalid labels"):
        load_uci_sms(source)


def test_rejects_empty_message_text(tmp_path: Path) -> None:
    # Обе метки присутствуют, поэтому проверяется именно обработка пустого текста.
    source = write_tsv(tmp_path, "ham\tNormal message\nspam\t   \n")

    with pytest.raises(ValueError, match="Empty message text"):
        load_uci_sms(source)


def test_rejects_empty_file(tmp_path: Path) -> None:
    # Для файла нулевого размера pandas выбрасывает EmptyDataError.
    # Загрузчик преобразует её в стабильный ValueError, не раскрывая
    # специфичную для pandas ошибку вызывающему коду.
    source = write_tsv(tmp_path, "")

    with pytest.raises(ValueError, match="The dataset is empty"):
        load_uci_sms(source)
