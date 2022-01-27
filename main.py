import pandas as pd
from os import path, system

columns_to_load = 100


def load_dataset(dataset_path, sample=False):
    if not path.exists(dataset_path):
        system('cat data/USCensus1990.data.txt.* > data/USCensus1990.data.txt')
    if sample:
        a_dataframe = pd.read_csv(dataset_path)[:columns_to_load]
    else:
        a_dataframe = pd.read_csv(dataset_path)
    # need to ignore caseid
    a_dataframe = a_dataframe.drop(columns=['caseid'])
    return a_dataframe


if __name__ == '__main__':
    data = load_dataset('data/USCensus1990.data.txt', sample=True)
    print(data)
    print()
    print()