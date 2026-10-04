import pandas as pd

def load_uci_sms(path: str) -> pd.DataFrame:
    data = pd.read_csv(
                path,
                sep="\t",
                header=None,
                names=["label", "text"],
                dtype={"label": "string", "text": "string"},
                encoding="utf-8-sig",
                keep_default_na=False,
            )
    return data