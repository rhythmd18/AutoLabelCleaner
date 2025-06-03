import random
from math import *
from collections import Counter
import pandas as pd
from pandas.api.types import is_float_dtype
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import *
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import warnings

random.seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')


def is_numeric(D, col):
    '''
    Check if a column in the DataFrame is numeric.
    Input:
        D: DataFrame, the dataset to check.
        col: str, the column name to check.
    Output:
        bool, True if the column is numeric, False otherwise.
    '''
    return not (D[col].dtype == 'O' or (D[col].dtype == 'int' and D[col].nunique() < 0.05 * len(D)))


def introduce_noise(df, cols, noise_percentage=0.1, threshold=0.49):
    '''
    Introduce noise into a DataFrame by randomly changing values in specified columns.
    Input:
        df: DataFrame, the dataset to introduce noise into.
        cols: list of str, the columns to introduce noise in.
        noise_percentage: float, the percentage of rows to introduce noise into (default is 0.1).
        threshold: float, the minimum difference required to consider a value as noisy (default is 0.49).
    Output:
        D: DataFrame, the dataset with noise introduced.
        noisy_cells: list of tuples, each tuple contains the index and column name of the noisy cell.
    '''
    D = df.copy()
    num_rows = len(df)
    num_noisy_rows = int(noise_percentage * num_rows)
    noisy_indices = random.sample(range(num_rows), num_noisy_rows)
    noisy_cells = []

    for idx in noisy_indices:
        col = random.choice(cols)
        original_value = df.loc[idx, col]

        if is_numeric(df, col):
            min_val = D[col].min()
            max_val = D[col].max()
            # Generate noise until it's significantly different
            noise_value = original_value
            while abs(noise_value - original_value) < threshold:
                noise_value = min_val + (max_val - min_val) * random.random()
            D.loc[idx, col] = noise_value
        else:
            possible_values = [v for v in df[col].unique().tolist() if v != original_value]
            if possible_values:
                noise_value = random.choice(possible_values)
                D.loc[idx, col] = noise_value
            else:
                # If there's only one unique value, skip adding noise
                continue

        noisy_cells.append((idx, col))
    return D, noisy_cells


def get_n_clusters(D):
    '''
    Determine the optimal number of clusters using silhouette score.
    Input:
        D: DataFrame or 2D array-like, the data to cluster.
    Output:
        k: int, the optimal number of clusters.
    '''
    range_n_clusters = [2, 3, 4, 5, 6]
    k = 2
    max_avg = 0
    for n_clusters in range_n_clusters:
        if len(D) < n_clusters:
            continue
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(D)
        silhouette_avg = silhouette_score(D, cluster_labels)
        # print(f'For n_clusters = {n_clusters}, The average silhouette score is {silhouette_avg}')
        if silhouette_avg > max_avg:
            max_avg = silhouette_avg
            k = n_clusters
    return k


def detect_and_reduce_errors(D, cols, k_prime, phi, n):
    '''
    Detect and reduce errors in the dataset using a clustering-based approach.
    Input:
        D: DataFrame, the dataset to process.
        cols: list of str, the columns to consider for error detection.
        k_prime: int, the number of nearest neighbors to consider.
        phi: float, the threshold for numeric attribute distance.
        n: int, the number of iterations to run for error detection.
    Output:
        final_flags: dict, a dictionary where keys are column names and values are lists of indices of detected errors.
    '''

    # -----------------------PHASES 3.1 - 3.3----------------------------------
    def single_detection_run(D, k_prime, phi):
        flags = {}
        D_prime_full = D.copy()

        for col in cols:
            D_prime = D.drop(col, axis=1)
            clusterer = KMeans(n_clusters=get_n_clusters(D))
            preds = clusterer.fit_predict(D_prime)
            D_prime['cluster'] = preds

            # Prepare NearestNeighbors across the entire data (excluding cluster col)
            all_features = D_prime.drop('cluster', axis=1).values
            nn = NearestNeighbors(n_neighbors=k_prime + 1, metric='euclidean')
            nn.fit(all_features)
            distances_all, neighbors_all = nn.kneighbors(all_features)

            # Precompute once: maps each index to its neighbors and distances
            neighbors_dict = {
                idx: (distances_all[idx][1:], neighbors_all[idx][1:])
                for idx in range(len(D))
            }

            for idx in D.index:
                cluster_indices = D_prime[D_prime['cluster'] == D_prime.loc[idx, 'cluster']].index.tolist()
                neighbor_indices = neighbors_dict[idx][1]
                neighbor_indices = [i for i in neighbor_indices if i in cluster_indices]

                if not neighbor_indices:
                    continue

                if is_numeric(D, col):
                    nearest = D.iloc[neighbor_indices]
                    if distances_all[idx][1] < 0.21 and dist(nearest.mean(), D.iloc[idx]) > phi:
                        if col not in flags:
                            flags[col] = []
                        flags[col].append(idx)
                else:
                    votes = D.iloc[neighbor_indices][col]
                    majority = Counter(votes).most_common(1)[0][0]
                    if majority != D.iloc[idx][col]:
                        if col not in flags:
                            flags[col] = []
                        flags[col].append(idx)

        return flags


    # ---------------------------PHASE 3.4------------------------------------
    num_attributes = [col for col in cols if is_numeric(D, col)]
    cat_attributes = [col for col in cols if not is_numeric(D, col)]

    max_errors_num, max_errors_cat = 0, 0
    n1_num, n1_cat = 0, 0

    final_flags = {}

    flags_for_each_iteration = {}
    for i in range(n):
        flags = single_detection_run(D, k_prime, phi) # Steps 3.1 - 3.3
        flags_for_each_iteration[i] = flags

        total_errors_num = sum(len(flags[A]) for A in num_attributes if A in flags)
        total_errors_cat = sum(len(flags[A]) for A in cat_attributes if A in flags)

        if total_errors_num > max_errors_num:
            max_errors_num = total_errors_num
            n1_num = i

        if total_errors_cat > max_errors_cat:
            max_errors_cat = total_errors_cat
            n1_cat = i


    # For numeric attributes
    if len(num_attributes):
        for attr in num_attributes:
            for flag in flags_for_each_iteration[n1_num][attr]:
                count = 0
                for iter in range(n):
                    if iter != n1_num and flag in flags_for_each_iteration[iter][attr]:
                        count += 1
                if count >= n-1:
                    if attr not in final_flags:
                        final_flags[attr] = []
                    final_flags[attr].append(flag)


    # For categorical attributes
    if len(cat_attributes) > 0:
        temp_flags = {}
        for attr in cat_attributes:
            for flag in flags_for_each_iteration[n1_cat][attr]:
                count = 0
                for iter in range(n):
                    if iter != n1_cat and flag in flags_for_each_iteration[iter][attr]:
                        count += 1
                if count >= n-1:
                    if attr not in temp_flags:
                        temp_flags[attr] = []
                    temp_flags[attr].append(flag)


        new_flags_for_each_iteration = {}
        for i in range(n):
            new_flags = single_detection_run(D, k_prime-1, phi) # Steps 3.1 - 3.3
            new_flags_for_each_iteration[i] = new_flags

        for attr in temp_flags:
            for flag in temp_flags[attr]:
                count = 0
                for i in new_flags_for_each_iteration:
                    if flag in new_flags_for_each_iteration[i][attr]:
                        count += 1
                if count == n:
                    if attr not in final_flags:
                        final_flags[attr] = []
                    final_flags[attr].append(flag)

    return final_flags


def get_classification_metrics(N, P, TN, TP, FP, FN):
    '''
    Calculate classification metrics based on the confusion matrix values.
    Input:
        N: int, number of negative instances.
        P: int, number of positive instances.
        TN: int, true negatives.
        TP: int, true positives.
        FP: int, false positives.
        FN: int, false negatives.
    Output:
        tuple, containing false alarm rate, error rate, true negative rate, precision, recall, and F1 score.
    '''
    false_alarm_rate = FP / (FP + TN)
    error_rate = (FP + FN) / (N + P)
    true_negative_rate = TN / (TN + FP)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = (2 * precision * recall) / (precision + recall)

    return (false_alarm_rate, error_rate, true_negative_rate, precision, recall, f1_score)


def score(f, e, t, p, r):
    '''
    Calculate a score based on the classification metrics. 
    The aim is to balance the metrics in a way that higher values of true negative rate, precision, and recall are favored 
    while penalizing false alarm and error rates.
    Input:
        f: float, false alarm rate.
        e: float, error rate.
        t: float, true negative rate.
        p: float, precision.
        r: float, recall.
    Output:
        float, the calculated score.
    '''
    return t + p + r - f - e


def get_detection_results(iteration, phi, k_prime, verbose=False):
        '''
        Run the error detection and reduction process, and calculate classification metrics.
        Input:
            iteration: int, the number of iterations to run for error detection.
            phi: float, the threshold for numeric attribute distance.
            k_prime: int, the number of nearest neighbors to consider.
        Output:
            tuple, containing false alarm rate, error rate, true negative rate, precision, recall, and F1 score.
        '''
        flags = detect_and_reduce_errors(D, cols, k_prime=k_prime, phi=phi, n=iteration)
        flagged_noises = set()
        for rows in flags.values():
            for row in rows:
                flagged_noises.add(row)

        true_noises = [cell[0] for cell in sorted(noisy_cells)]
        true_correct = [i for i in D.index if i not in true_noises]
        flagged_correct = [i for i in D.index if i not in flagged_noises]

        N = len(true_noises)
        P = len(D) - N
        TN = len(set(flagged_noises) & set(true_noises))
        TP = len(set(flagged_correct) & set(true_correct))
        FP = len(set(true_noises) & set(flagged_correct))
        FN = len(set(true_correct) & set(flagged_noises))

        f, e, t, p, r, f1 = get_classification_metrics(N, P, TN, TP, FP, FN)

        if verbose:
            print(f'\n-----------------------------------------------')
            print(f'For n = {iteration}, phi = {phi}, k\' = {k_prime}')
            print(f'False Alarm Rate:   {f}')
            print(f'Error Rate:         {e}')
            print(f'True Negative Rate: {t}')
            print(f'Precision:          {p}')
            print(f'Recall:             {r}')
            print(f'F1-Score:           {f1}')

        return (f, e, t, p, r, f1)