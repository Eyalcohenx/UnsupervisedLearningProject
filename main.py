import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from os import path, system
from sklearn.decomposition import PCA
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.sos import SOS
from pyod.models.abod import ABOD
from pyod.models.lmdd import LMDD
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.cof import COF
from pyod.models.copod import COPOD


def OutlierFilter(X, y=None, alg=0, top_num_to_remove=20):

    # Probabilistic - Copula-Based Outlier Detection
    if alg == 0 or alg == 'copod' or alg == 'COPOD':
        clf = ABOD()

    # Linear Model - One-Class Support Vector Machines
    elif alg == 1 or alg == 'ocsvm' or alg == 'OCSVM':
        clf = OCSVM()

    # Proximity-Based - Connectivity-Based Outlier Factor
    elif alg == 2 or alg == 'cof' or alg == 'COF':
        clf = COF()

    # Neural Network - Deep One-Class Classification
    elif alg == 3 or alg == 'auto_encoder' or alg == 'AutoEncoder':
        clf = DeepSVDD()

    else:
        print("Unknown outliers screening algorithm")

    X = X.astype(float)

    if y is not None:
        clf.fit(X, y)
    else:
        clf.fit(X)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)

    X["score"] = (clf.decision_scores_ / sum(clf.decision_scores_)).tolist()

    drop_rows = X.sort_values("score", ascending=False).head(top_num_to_remove)
    print("removing the following examples:\n")
    print(drop_rows)

    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
    red_points = principalDf.iloc[drop_rows.index.values.tolist()]
    plt.scatter(red_points['principal component 1'], red_points['principal component 2'], c='red')
    print(red_points)
    plt.show()

    X.drop(["score"], axis=1)

    return drop_rows.index.values.tolist()


rows_to_load = 1000


def load_dataset(dataset_path, sample=False):
    if not path.exists(dataset_path):
        system('cat data/USCensus1990.data.txt.* > data/USCensus1990.data.txt')
    if sample:
        a_dataframe = pd.read_csv(dataset_path)[:rows_to_load]
    else:
        a_dataframe = pd.read_csv(dataset_path)
    # need to ignore caseid
    a_dataframe = a_dataframe.drop(columns=['caseid'])

    return a_dataframe


def main():
    # Loading the data
    data = load_dataset('data/USCensus1990.data.txt', sample=True)

    # Filter outliers
    outlier_indexes = OutlierFilter(data, alg=0, top_num_to_remove=20)
    filtered_data = data.drop(outlier_indexes, axis=0, inplace=True)

    print(filtered_data)


if __name__ == '__main__':
    main()
