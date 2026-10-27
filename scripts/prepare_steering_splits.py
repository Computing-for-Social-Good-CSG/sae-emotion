"""Rebuild the shared steering splits from the combined source (seed 42)."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def build_splits(source):
    source = source.copy()
    source['emotion'] = source['emotion'].str.lower().str.strip()
    pool, test = train_test_split(source[['input_text', 'emotion']], test_size=.2,
                                  random_state=42, stratify=source.emotion)
    pool = pd.concat([g.sample(n=400, random_state=42)
                      for _, g in pool.groupby('emotion')], ignore_index=True)
    pool = pool.sample(frac=1, random_state=42).reset_index(drop=True)
    pool = pool[~pool.input_text.isin(test.input_text)]
    groups = pool.drop_duplicates('input_text')
    # First label defines the stratum; every occurrence stays with its group.
    train_keys, val_keys = train_test_split(groups.input_text, test_size=.2,
                                            random_state=42, stratify=groups.emotion)
    train = pool[pool.input_text.isin(train_keys)].sample(frac=1, random_state=42).reset_index(drop=True)
    val = pool[pool.input_text.isin(val_keys)].sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.reset_index(drop=True)
    features = source[source.input_text.isin(train.input_text)].drop_duplicates(['input_text', 'emotion'])
    return {'train_df': train, 'val_df': val, 'test_df': test,
            'feature_discovery_train': features.reset_index(drop=True)}


if __name__ == '__main__':
    data = Path(__file__).resolve().parents[1] / 'data'
    for name, frame in build_splits(pd.read_csv(data / 'combined_emotions.csv')).items():
        frame.to_csv(data / f'{name}.csv', index=False)
        print(f'{name}: {len(frame)} rows')
