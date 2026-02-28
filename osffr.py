import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from aif360.metrics import utils
from scipy.sparse import issparse
import random
from gower import gower_matrix
from sklearn.cluster import DBSCAN
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sympy import Symbol
from sympy.solvers import solve
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from aif360.algorithms.preprocessing import *
from aif360.algorithms.preprocessing.optim_preproc_helpers import distortion_functions, opt_tools
from aif360.algorithms.inprocessing import *
from aif360.algorithms.postprocessing import *
import math
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from DataPreparation import DataPreparation


def preprocess_dataset(dataset_path, dataset_type, model):
    if dataset_type == 'German':
        df = pd.read_csv(dataset_path)
        df['age'] = df['age'].apply(lambda age: 1 if age >= 25 else 0)
        df['personal_status'] = df['personal_status'].apply(lambda sex: 1 if sex == 'male' else 0)
        print("German dataset:")
        print(df.head())
        sensitive_attributes = ['personal_status','age']
        label = 'credit'
        privileged = [1, 1]
        unprivileged = [0, 0]
        favorable_label = 1
        unfavorable_label = 2
        groups = [
                  {'name': 'Male Adult', 'attributes': {'personal_status': 1, 'age': 1}},
                  {'name': 'Female Adult', 'attributes': {'personal_status': 0, 'age': 1}},
                  {'name': 'Male Young', 'attributes': {'personal_status': 1, 'age': 0}},
                  {'name': 'Female Young', 'attributes': {'personal_status': 0, 'age': 0}}
              ]
        model = model

    elif dataset_type == 'COMPAS':
        df = pd.read_csv(dataset_path)
        selected_columns = ['sex', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count',
                            'juv_other_count', 'priors_count', 'c_charge_degree',
                            'c_charge_desc', 'two_year_recid']
        df = df[selected_columns]
        df = df[(df['race'] == 'Caucasian') | (df['race'] == 'African-American')].reset_index(drop=True)
        print("COMPAS dataset:")
        print(df.head())
        sensitive_attributes = ['race','sex']
        label = 'two_year_recid'
        privileged = ['Caucasian', 'Female']
        unprivileged = ['African-American', 'Male']
        favorable_label = 0
        unfavorable_label = 1
        groups = [
                  {'name': 'Caucasian Female', 'attributes': {'race': 1, 'sex': 1}},
                  {'name': 'Black Female', 'attributes': {'race': 0, 'sex': 1}},
                  {'name': 'Causasian Male', 'attributes': {'race': 1, 'sex': 0}},
                  {'name': 'Black Male', 'attributes': {'race': 0, 'sex': 0}}
              ]
        model = model

    elif dataset_type == 'Adult':
        df = pd.read_csv(dataset_path, delimiter=';')
        df['income'] = df['income'].str.strip().replace({'>50K.': '>50K', '<=50K.': '<=50K'})
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        df.replace('?', np.nan, inplace=True)
        df = df.drop(columns=['fnlwgt', 'education-num'])
        df = df[(df['race'] == 'White') | (df['race'] == 'Black')].reset_index(drop=True)
        print("Adult dataset:")
        print(df.head())
        sensitive_attributes = ['race','sex']
        label = 'income'
        privileged = ['White', 'Male']
        unprivileged = ['Black', 'Female']
        favorable_label = '>50K'
        unfavorable_label = '<=50K'
        groups = [
                  {'name': 'White Male', 'attributes': {'race': 1, 'sex': 1}},
                  {'name': 'Black Male', 'attributes': {'race': 0, 'sex': 1}},
                  {'name': 'White Female', 'attributes': {'race': 1, 'sex': 0}},
                  {'name': 'Black Female', 'attributes': {'race': 0, 'sex': 0}}
              ]
        model = model
    elif dataset_type == 'Hospital':
        df = pd.read_csv(dataset_path, delimiter=',')
        print("Hospital dataset:")
        print(df.head())
        sensitive_attributes = ['gender', 'race']
        label = 'disposition'
        privileged = ['Male', 1]
        unprivileged = ['Female', 0]
        favorable_label = 'Admit'
        unfavorable_label = 'Discharge'
        groups = [
                  {'name': 'White Male', 'attributes': {'race': 1, 'gender': 1}},
                  {'name': 'Others Male', 'attributes': {'race': 0, 'gender': 1}},
                  {'name': 'White Female', 'attributes': {'race': 1, 'gender': 0}},
                  {'name': 'Others Female', 'attributes': {'race': 0, 'gender': 0}}
              ]
        model = model

    return df, sensitive_attributes, label, privileged, unprivileged, \
    		favorable_label, unfavorable_label, groups, model

#====================================================

def find_optimal_epsilon(filtered_data, cat_features, min_samples, distance_matrix, eps_step=0.001, eps_min=0.01, eps_max=1.1):

    def cluster_count(eps):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)  # - (1 if -1 in unique_labels else 0)
        return n_clusters, unique_labels

    # Binary search for optimal epsilon
    while eps_max - eps_min > eps_step:
        eps_mid = (eps_min + eps_max) / 2
        n_clusters_mid, labels_mid = cluster_count(eps_mid)
        print(eps_mid, n_clusters_mid, labels_mid, 'mids')

        if n_clusters_mid == 1:
            if -1 in labels_mid:
                eps_min = eps_mid  # Only noise points, increase epsilon
            else:
                eps_max = eps_mid  # Only core points, decrease epsilon
        elif n_clusters_mid > 2:
            eps_min = eps_mid  # More than two clusters, increase epsilon
        else:
            eps_max = eps_mid  # Exactly two clusters, continue search to fine-tune
        print(eps_min, eps_max, 'min, max')
    return eps_max

# Custom SMOTE-DBSCAN function
def custom_smote_dbscan(X_train, cat_features, pu_ix, nu_ix, group_column_train, total_ratio):
    """
    X_train is the training dataset preprocessed, group_column_train is a column containing the group of each
    instance in X_train
    """
    cat_attr_ix = [i for i, value in enumerate(cat_features) if value]

    X2_df = X_train[group_column_train == pu_ix]
    X2 = X2_df.values
    X3_df = X_train[group_column_train == nu_ix]
    X3 = X3_df.values

    PU = len(X2)
    NU = len(X3)

    # Determine the oversampling target based on a given total_ratio
    if (PU / NU) > total_ratio:
        oversampling_target = (PU / total_ratio) - NU
        os_df = X3_df
        os_ix = nu_ix
    elif (PU / NU) == total_ratio:
        print("The ratio of PU to NU is within the acceptable range of total_ratio.")
        return [], 0, pu_ix
    else:
        oversampling_target = (total_ratio * NU) - PU
        os_df = X2_df
        os_ix = pu_ix
    os_df = os_df.reset_index(drop=True)

    # Calculate min_samples
    min_samples = round(math.log(len(os_df)))
    distance_matrix = gower_matrix(os_df, cat_features=cat_features)

    # Find the optimal epsilon for os_df
    optimal_eps = find_optimal_epsilon(os_df, cat_features, min_samples, distance_matrix)

    # DBSCAN clustering with the optimal epsilon
    dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples, metric='precomputed')
    clusters = dbscan.fit_predict(distance_matrix)

    # Get cluster labels
    labels = dbscan.labels_

    # Get core samples
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    # Identify core, border, and noise points
    core_points = os_df[core_samples_mask]
    border_points = os_df[~core_samples_mask & (labels != -1)]
    noise_points = os_df[labels == -1]

    if len(border_points) == 0:
        border_points = core_points

    # Initialize synthetic samples list
    synthetic_samples = []

    border_indices = border_points.index.tolist()
    random.shuffle(border_indices)
    current_index = 0

    while len(synthetic_samples) < oversampling_target:
        idx_A = border_indices[current_index % len(border_indices)]
        current_index += 1
        point_A = os_df.loc[idx_A]

        # Ensure point B is not a noise point
        distances_to_A = distance_matrix[idx_A]
        neighbors = np.argsort(distances_to_A)[1:min_samples+1]  # Exclude the point itself
        valid_neighbors = [idx for idx in neighbors if labels[idx] != -1]  # Exclude noise points

        if not valid_neighbors:
            continue  # Skip if no valid neighbors are found

        idx_B = np.random.choice(valid_neighbors)
        point_B = os_df.loc[idx_B]

        synthetic_point = {}
        for i, col in enumerate(os_df.columns):
            if cat_features[i]:
                neighbor_values = os_df.iloc[valid_neighbors][col].tolist()
                synthetic_point[col] = max(set(neighbor_values), key=neighbor_values.count)
            else:
                alpha = np.random.rand()
                synthetic_point[col] = point_A[col] + alpha * (point_B[col] - point_A[col])

        synthetic_samples.append(synthetic_point)

    return pd.DataFrame(synthetic_samples), len(synthetic_samples), os_ix

# Example usage (assuming you have defined X_train, cat_features, etc.):
# synthetic_samples, num_samples, oversampled_index = custom_smote_dbscan(X_train, cat_features, pu_ix, nu_ix, group_column_train, total_ratio)

#====================================================

def oversample_groups(X_train, cat_features, custom_smote, group_column_train, total_ratio, reverse_group_mapping):
    """
    Function to oversample multiple groups automatically based on group labels.

    Parameters:
    - X_train: Preprocessed training dataset.
    - cat_features: List indicating categorical features.
    - custom_smote: Custom SMOTE function to be used.
    - group_column_train: Column containing the group label for each instance.
    - total_ratio: Desired ratio of positive to negative labels.
    - reverse_group_mapping: Mapping of groups to sensitive attributes and labels.

    Returns:
    - synthetic_samples_matrix: Matrix containing all generated synthetic samples.
    - synthetic_samples_group: Array of group labels for the synthetic samples.
    """

    synthetic_samples = []
    synthetic_samples_group = []

    groups = sorted(group_column_train.unique())
    paired_groups = [(groups[i], groups[i+1]) for i in range(0, len(groups), 2)]

    for group1, group2 in paired_groups:
        ########## Determine pu_ix and nu_ix using reverse_group_mapping ##########
        if reverse_group_mapping[group1][2] == 1:
            pu_ix = group1
            nu_ix = group2
        else:
            pu_ix = group2
            nu_ix = group1
        ##########################################################################

        group_df_pu = X_train[group_column_train == pu_ix]
        group_df_nu = X_train[group_column_train == nu_ix]
        positive_count = group_df_pu[group_df_pu[label] == 1].shape[0]
        negative_count = group_df_nu[group_df_nu[label] == 0].shape[0]

        if positive_count == 0 or negative_count == 0:
            continue

        current_ratio = positive_count / negative_count

        if current_ratio == total_ratio:
            continue  # Skip the most privileged group

        synthetic_points, synthetic_count, os_ix = custom_smote(X_train, cat_features, pu_ix, nu_ix, group_column_train, total_ratio=total_ratio)
        pu_column = np.full((len(synthetic_points), 1), os_ix)
        synthetic_samples.append(synthetic_points)
        synthetic_samples_group.append(pu_column)
        print(f"Oversampling for group pair ({pu_ix}, {nu_ix}): Added {synthetic_count} synthetic samples in {os_ix}.")

    synthetic_samples_matrix = pd.concat(synthetic_samples, ignore_index=True)
    synthetic_samples_group = np.concatenate(synthetic_samples_group)

    return synthetic_samples_matrix, synthetic_samples_group


#====================================================

def evaluate_model_performance(X_train, X_test, protected_attributes, label_name, groups, model, weights=None):
    favorable_label = 1.0
    unfavorable_label = 0.0
    X_train[label_name] = X_train[label_name].astype(float)
    X_test[label_name] = X_test[label_name].astype(float)
    # If weights is not provided, create an array of ones with the same length as X_train
    if weights is None:
        weights = np.ones(len(X_train))

    # Create BinaryLabelDatasets
    binary_ds_train = BinaryLabelDataset(df=X_train, label_names=[label_name],
                                         protected_attribute_names=protected_attributes,
                                         favorable_label=favorable_label, unfavorable_label=unfavorable_label)
    binary_ds_test = BinaryLabelDataset(df=X_test, label_names=[label_name],
                                        protected_attribute_names=protected_attributes,
                                        favorable_label=favorable_label, unfavorable_label=unfavorable_label)
    if model == 'Logistic Regression':
        classifier = LogisticRegression(max_iter = 2000)
    elif model == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model == 'Gradient Boosting':
        classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model == 'Neural Network':
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), max_iter=1000, random_state=1)
    else:
        raise ValueError('Choose one classification algorithm between Logistic Regression, Random Forest, Gradient Boosting')
    if model == 'Neural Network':
        classifier.fit(X_train.drop(columns=[label_name]), X_train[label_name])
    else:
        classifier.fit(X_train.drop(columns=[label_name]), X_train[label_name], sample_weight=weights)
    predicted_labels = classifier.predict(X_test.drop(columns=[label_name]))

    X_test_with_predictions = pd.concat([X_test.drop(columns=[label_name]), pd.Series(predicted_labels, name=label_name, index=X_test.index)], axis=1)

    binary_ds_test_pred = BinaryLabelDataset(df=X_test_with_predictions, label_names=[label_name],
                                             protected_attribute_names=protected_attributes,
                                             favorable_label=favorable_label, unfavorable_label=unfavorable_label)

    all_results = {}
    for (group1, group2) in itertools.combinations(groups, 2):
        print(group1, group2, 'gruppi')
        pair_key = f"{group1['name']} vs {group2['name']}"
        all_results[pair_key] = evaluate(
            binary_ds_test, binary_ds_test_pred,
            [group1['attributes']], [group2['attributes']])
        #print([group1['attributes']], [group2['attributes']])


    return all_results, predicted_labels

#====================================================

def evaluate(test_data, pred, priv_group, unpriv_group):
    cm = ClassificationMetric(test_data, pred,
                              unprivileged_groups=unpriv_group,
                              privileged_groups=priv_group)
    dm = BinaryLabelDatasetMetric(pred,
                                  unprivileged_groups=unpriv_group,
                                  privileged_groups=priv_group)

    measure_scores = {
        'Balanced Accuracy': balanced_accuracy_score(test_data.labels, pred.labels),
        'Accuracy': cm.accuracy(),
        'F1 Score': f1_score(test_data.labels.ravel(), pred.labels.ravel()),  # Ensure labels are flat
        'Disparate Impact Ratio': dm.disparate_impact(),
        #'Demographic Parity Difference': cm.statistical_parity_difference(),
        #'Predictive Parity Difference': cm.positive_predictive_value(privileged=True) - cm.positive_predictive_value(privileged=False),
        'Average Odds Difference': cm.average_odds_difference(),
        'Equal Opportunity Difference': cm.equal_opportunity_difference(),
        #'Equalized Odds Difference': cm.average_abs_odds_difference(),
        'Consistency': dm.consistency(),
        #'TPR Difference': cm.true_positive_rate_difference(),
        #'FPR Difference': cm.false_positive_rate_difference(),
        #'TNR Difference': cm.true_negative_rate(privileged=True) - cm.true_negative_rate(privileged=False),
        #'FNR Difference': cm.false_negative_rate_difference(),
    }

    return measure_scores

#====================================================

def compute_metrics(df, actual_labels, predicted_labels):
    """Compute fairness and performance metrics."""
    cm = confusion_matrix(actual_labels, predicted_labels)
    TN, FP, FN, TP = cm.ravel()
    metrics = {
        'Accuracy': accuracy_score(actual_labels, predicted_labels),
        'Precision': precision_score(actual_labels, predicted_labels),
        'Recall': recall_score(actual_labels, predicted_labels),
        'F1 Score': f1_score(actual_labels, predicted_labels),
        'TPR': TP / (TP + FN),
        'FPR': FP / (FP + TN),
        'TNR': TN / (TN + FP),
        'FNR': FN / (FN + TP),
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    }

    return metrics

#====================================================

def compute_IR(train_df, s_attr, label):
    # receives the training dataframe and the list of sensitive attributes
    # Define sensitive attribute and label columns
    s_attr_0 = s_attr[0]
    s_attr_1 = s_attr[1]

    # Calculate the number of ones and zeros for each group
    num_group_11_ones = train_df[(train_df[s_attr_0] == 1) &
                                (train_df[s_attr_1] == 1) &
                                (train_df[label] == 1)].shape[0]

    num_group_11_zeros = train_df[(train_df[s_attr_0] == 1) &
                                 (train_df[s_attr_1] == 1) &
                                 (train_df[label] == 0)].shape[0]

    num_group_10_ones = train_df[(train_df[s_attr_0] == 1) &
                                (train_df[s_attr_1] == 0) &
                                (train_df[label] == 1)].shape[0]

    num_group_10_zeros = train_df[(train_df[s_attr_0] == 1) &
                                 (train_df[s_attr_1] == 0) &
                                 (train_df[label] == 0)].shape[0]

    num_group_01_ones = train_df[(train_df[s_attr_0] == 0) &
                                (train_df[s_attr_1] == 1) &
                                (train_df[label] == 1)].shape[0]

    num_group_01_zeros = train_df[(train_df[s_attr_0] == 0) &
                                 (train_df[s_attr_1] == 1) &
                                 (train_df[label] == 0)].shape[0]

    num_group_00_ones = train_df[(train_df[s_attr_0] == 0) &
                                (train_df[s_attr_1] == 0) &
                                (train_df[label] == 1)].shape[0]

    num_group_00_zeros = train_df[(train_df[s_attr_0] == 0) &
                                 (train_df[s_attr_1] == 0) &
                                 (train_df[label] == 0)].shape[0]

    # Calculate imbalance ratios
    imbalance_ratio_11 = num_group_11_ones / num_group_11_zeros if num_group_11_zeros != 0 else float('inf')
    imbalance_ratio_10 = num_group_10_ones / num_group_10_zeros if num_group_10_zeros != 0 else float('inf')
    imbalance_ratio_01 = num_group_01_ones / num_group_01_zeros if num_group_01_zeros != 0 else float('inf')
    imbalance_ratio_00 = num_group_00_ones / num_group_00_zeros if num_group_00_zeros != 0 else float('inf')

    # Print imbalance ratios
    print(f"Imbalance ratio for group (1, 1): {imbalance_ratio_11}")
    print(f"Imbalance ratio for group (1, 0): {imbalance_ratio_10}")
    print(f"Imbalance ratio for group (0, 1): {imbalance_ratio_01}")


#====================================================

def test(df):  
    race = [0, 1]
    gender = ['Male', 'Female']
    disposition = ['Admit', 'Discharge']
    s = 0
    for d in disposition:
        for g in gender:
            for r in race:
                L1 = len(df.loc[(df['race'] == r ) & (df['gender'] == g ) & df['disposition'].eq(d)])
                L2 = len(df.loc[(df['race'] == r ) & (df['gender'] == g )])
                print(" r = ", r, "\tg = ", g, "\t d = ", d, "( ", L1, L2, " )")
                # print(" r = ", r, "\tg = ", g, "\t d = ", d, "( ", L1, L2, " )", "( ", round(L1 / L2, 3), " )")
                s += L1
    print(s, len(df))

#====================================================

def main():
    ##################################################################################
    # df, sensitive_attributes, label, privileged, unprivileged, favorable_label, unfavorable_label, groups, model = preprocess_dataset('/content/raw_german_dataset.csv', 'Adult', 'Gradient Boosting')
    data_path = "data/"
    german_data_path = data_path + 'raw_german_dataset.csv'
    compas_data_path = data_path + 'raw_compas_dataset.csv'
    adult_data_path = data_path + 'raw_adult_dataset.csv'
    hospital_data_path = data_path + 'raw_hospital_dataset.csv'

    current_dataset = [hospital_data_path, 'Hospital']
    # current_dataset = [german_data_path, 'German']
    # current_dataset = [compas_data_path, 'COMPAS']
    # current_dataset = [adult_data_path, 'Adult']

    df, sensitive_attributes, label, privileged, unprivileged, favorable_label, unfavorable_label, \
        groups, model = preprocess_dataset(current_dataset[0], current_dataset[1], 'Logistic Regression')
    # = '/content/raw_german_dataset.csv', 'German'
    # = preprocess_dataset('/content/raw_compas_dataset.csv', 'COMPAS')
    # = preprocess_dataset('/content/raw_adult_dataset.csv', 'Adult')

    #model == 'Logistic Regression':
    #model == 'Random Forest':
    #model == 'Gradient Boosting':
    #model == 'Neural Network':

    #use the DataPreparation class to preprocess the dataframe
    data_prep = DataPreparation(df, sensitive_attributes, label, privileged, unprivileged, favorable_label, unfavorable_label)
    data_prep.prepare()
    data_prep.df = data_prep.df.reset_index(drop=True)
    X_train, X_test = data_prep.X_train, data_prep.X_test
    attribute_types = data_prep.attribute_types
    cat_features = data_prep.cat_features
    numerical_features = data_prep.numerical_features
    reverse_group_mapping = data_prep.create_group_column()
    #theoretical_num_groups = len(reverse_group_mapping)
    X_train = X_train.reset_index(drop=True)


    compute_IR(X_train, sensitive_attributes, label)
    # test(X_train)
    # print (len(data_prep.X_train))
    print(len(X_train))
    print('\n===================================\n\n')
    results_orig, pred_labels_orig = evaluate_model_performance(X_train, X_test, sensitive_attributes, label,
                                                                groups, model=model)
    #model = 'Random Forest'
    #model = 'Gradient Boosting'

    # Initialize a list to hold DataFrames
    data_frames = []

    # Populate the list with DataFrames, each having a unique row index
    for key, values in results_orig.items():
        df_part = pd.DataFrame([values], index=[key])
        data_frames.append(df_part)

    # Concatenate all DataFrames into a single DataFrame
    results_orig_df = pd.concat(data_frames)
    results_orig_df.index.name = 'Comparison'

    # Print the results DataFrame
    print(results_orig_df)






if __name__ == "__main__":
    main()



