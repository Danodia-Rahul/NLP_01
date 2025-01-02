import os
from pathlib import Path
import pandas as pd
from datasets import Dataset

def prepare_data(kaggle_credentials: str, is_kaggle: bool):
    # Set up Kaggle credentials
    cred_path = Path('~/.kaggle/kaggle.json').expanduser()
    if not cred_path.exists():
        cred_path.parent.mkdir(exist_ok=True)
        cred_path.write_text(kaggle_credentials)
        cred_path.chmod(0o600)

    # Dataset path
    path = Path('us-patent-phrase-to-phrase-matching')
    if not is_kaggle and not path.exists():
        import zipfile, kaggle
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)

    if is_kaggle:
        path = Path('../input/us-patent-phrase-to-phrase-matching')

    # Load data
    df = pd.read_csv(path / 'train.csv')
    df['input'] = 'Text1: ' + df.context + '; Text2: ' + df.target + '; Anc1: ' + df.anchor
    ds = Dataset.from_pandas(df)
    return ds
