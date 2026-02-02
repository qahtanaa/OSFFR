import sys
# Define the new list of directories
new_path = ['./Fair-SMOTE']  # Adjust the path as needed

# Replace sys.path with the new list
sys.path = new_path
print(sys.path)

from Generate_Samples import generate_samples
from Measure import measure_final_score, calculate_recall, calculate_far, calculate_precision, calculate_accuracy
from SMOTE import smote
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but NearestNeighbors was fitted with feature names")


def oversample_fair_smote(X_train, sensitive_attributes, label):
    # Extracting group counts
    zero_zero_zero = len(X_train[(X_train[label] == 0) & (X_train[sensitive_attributes[0]] == 0)
                                & (X_train[sensitive_attributes[1]] == 0)])
    zero_zero_one = len(X_train[(X_train[label] == 0) & (X_train[sensitive_attributes[0]] == 0)
                                & (X_train[sensitive_attributes[1]] == 1)])
    zero_one_zero = len(X_train[(X_train[label] == 0) & (X_train[sensitive_attributes[0]] == 1)
                                & (X_train[sensitive_attributes[1]] == 0)])
    zero_one_one = len(X_train[(X_train[label] == 0) & (X_train[sensitive_attributes[0]] == 1)
                                & (X_train[sensitive_attributes[1]] == 1)])
    one_zero_zero = len(X_train[(X_train[label] == 1) & (X_train[sensitive_attributes[0]] == 0)
                                & (X_train[sensitive_attributes[1]] == 0)])
    one_zero_one = len(X_train[(X_train[label] == 1) & (X_train[sensitive_attributes[0]] == 0)
                                & (X_train[sensitive_attributes[1]] == 1)])
    one_one_zero = len(X_train[(X_train[label] == 1) & (X_train[sensitive_attributes[0]] == 1)
                                & (X_train[sensitive_attributes[1]] == 0)])
    one_one_one = len(X_train[(X_train[label] == 1) & (X_train[sensitive_attributes[0]] == 1)
                                & (X_train[sensitive_attributes[1]] == 1)])

    # Finding maximum
    maximum = max(zero_zero_zero, zero_zero_one, zero_one_zero, zero_one_one, one_zero_zero, one_zero_one, one_one_zero, one_one_one)
    print(f"Maximum count: {maximum}")

    # Printing which group has maximum count
    if maximum == zero_zero_zero:
        print("zero_zero_zero is maximum")
    elif maximum == zero_zero_one:
        print("zero_zero_one is maximum")
    elif maximum == zero_one_zero:
        print("zero_one_zero is maximum")
    elif maximum == zero_one_one:
        print("zero_one_one is maximum")
    elif maximum == one_zero_zero:
        print("one_zero_zero is maximum")
    elif maximum == one_zero_one:
        print("one_zero_one is maximum")
    elif maximum == one_one_zero:
        print("one_one_zero is maximum")
    elif maximum == one_one_one:
        print("one_one_one is maximum")

    # Calculating number of samples to be increased for each group
    zero_zero_zero_to_be_increased = maximum - zero_zero_zero
    zero_zero_one_to_be_increased = maximum - zero_zero_one
    zero_one_zero_to_be_increased = maximum - zero_one_zero
    zero_one_one_to_be_increased = maximum - zero_one_one
    one_zero_zero_to_be_increased = maximum - one_zero_zero
    one_zero_one_to_be_increased = maximum - one_zero_one
    one_one_zero_to_be_increased = maximum - one_one_zero
    one_one_one_to_be_increased = maximum - one_one_one

    print(f"Counts to be increased for each group:")
    print(f"zero_zero_zero: {zero_zero_zero_to_be_increased}")
    print(f"zero_zero_one: {zero_zero_one_to_be_increased}")
    print(f"zero_one_zero: {zero_one_zero_to_be_increased}")
    print(f"zero_one_one: {zero_one_one_to_be_increased}")
    print(f"one_zero_zero: {one_zero_zero_to_be_increased}")
    print(f"one_zero_one: {one_zero_one_to_be_increased}")
    print(f"one_one_zero: {one_one_zero_to_be_increased}")
    print(f"one_one_one: {one_one_one_to_be_increased}")

    df_zero_zero_zero = X_train[(X_train[label] == 0) & (X_train[sensitive_attributes[0]] == 0)
                                & (X_train[sensitive_attributes[1]] == 0)].copy()
    df_zero_zero_one = X_train[(X_train[label] == 0) & (X_train[sensitive_attributes[0]] == 0)
                                & (X_train[sensitive_attributes[1]] == 1)].copy()
    df_zero_one_zero = X_train[(X_train[label] == 0) & (X_train[sensitive_attributes[0]] == 1)
                                & (X_train[sensitive_attributes[1]] == 0)].copy()
    df_zero_one_one = X_train[(X_train[label] == 0) & (X_train[sensitive_attributes[0]] == 1)
                                & (X_train[sensitive_attributes[1]] == 1)].copy()
    df_one_zero_zero = X_train[(X_train[label] == 1) & (X_train[sensitive_attributes[0]] == 0)
                                & (X_train[sensitive_attributes[1]] == 0)].copy()
    df_one_zero_one = X_train[(X_train[label] == 1) & (X_train[sensitive_attributes[0]] == 0)
                                & (X_train[sensitive_attributes[1]] == 1)].copy()
    df_one_one_zero = X_train[(X_train[label] == 1) & (X_train[sensitive_attributes[0]] == 1)
                                & (X_train[sensitive_attributes[1]] == 0)].copy()
    df_one_one_one = X_train[(X_train[label] == 1) & (X_train[sensitive_attributes[0]] == 1)
                                & (X_train[sensitive_attributes[1]] == 1)].copy()


    df_zero_zero_zero.loc[:, sensitive_attributes[0]] = df_zero_zero_zero[sensitive_attributes[0]].astype(str)
    df_zero_zero_zero.loc[:, sensitive_attributes[1]] = df_zero_zero_zero[sensitive_attributes[1]].astype(str)

    df_zero_zero_one.loc[:, sensitive_attributes[0]] = df_zero_zero_one[sensitive_attributes[0]].astype(str)
    df_zero_zero_one.loc[:, sensitive_attributes[1]] = df_zero_zero_one[sensitive_attributes[1]].astype(str)

    df_zero_one_zero.loc[:, sensitive_attributes[0]] = df_zero_one_zero[sensitive_attributes[0]].astype(str)
    df_zero_one_zero.loc[:, sensitive_attributes[1]] = df_zero_one_zero[sensitive_attributes[1]].astype(str)

    df_zero_one_one.loc[:, sensitive_attributes[0]] = df_zero_one_one[sensitive_attributes[0]].astype(str)
    df_zero_one_one.loc[:, sensitive_attributes[1]] = df_zero_one_one[sensitive_attributes[1]].astype(str)

    df_one_zero_zero.loc[:, sensitive_attributes[0]] = df_one_zero_zero[sensitive_attributes[0]].astype(str)
    df_one_zero_zero.loc[:, sensitive_attributes[1]] = df_one_zero_zero[sensitive_attributes[1]].astype(str)

    df_one_zero_one.loc[:, sensitive_attributes[0]] = df_one_zero_one[sensitive_attributes[0]].astype(str)
    df_one_zero_one.loc[:, sensitive_attributes[1]] = df_one_zero_one[sensitive_attributes[1]].astype(str)

    df_one_one_zero.loc[:, sensitive_attributes[0]] = df_one_one_zero[sensitive_attributes[0]].astype(str)
    df_one_one_zero.loc[:, sensitive_attributes[1]] = df_one_one_zero[sensitive_attributes[1]].astype(str)

    df_one_one_one.loc[:, sensitive_attributes[0]] = df_one_one_one[sensitive_attributes[0]].astype(str)
    df_one_one_one.loc[:, sensitive_attributes[1]] = df_one_one_one[sensitive_attributes[1]].astype(str)

    # Generating samples for each group
    df_zero_zero_zero = generate_samples(zero_zero_zero_to_be_increased, df_zero_zero_zero, 'Germann')
    df_zero_zero_one = generate_samples(zero_zero_one_to_be_increased, df_zero_zero_one, 'Germann')
    df_zero_one_zero = generate_samples(zero_one_zero_to_be_increased, df_zero_one_zero, 'Germann')
    df_zero_one_one = generate_samples(zero_one_one_to_be_increased, df_zero_one_one, 'Germann')
    df_one_zero_zero = generate_samples(one_zero_zero_to_be_increased, df_one_zero_zero, 'Germann')
    df_one_zero_one = generate_samples(one_zero_one_to_be_increased, df_one_zero_one, 'Germann')
    df_one_one_zero = generate_samples(one_one_zero_to_be_increased, df_one_one_zero, 'Germann')
    df_one_one_one = generate_samples(one_one_one_to_be_increased, df_one_one_one, 'Germann')

    # Concatenating dataframes
    X_train_resampled_fair_smote = pd.concat([df_zero_zero_zero, df_zero_zero_one, df_zero_one_zero, df_zero_one_one,
                                              df_one_zero_zero, df_one_zero_one, df_one_one_zero, df_one_one_one])
    X_train_resampled_fair_smote.columns = X_train.columns

    return X_train_resampled_fair_smote

#====================================================

__main__
X_train_resampled_fair_smote = oversample_fair_smote(X_train, sensitive_attributes, label)
# Evaluate model performance
results_fair_smote, pred_labels_fair_smote = evaluate_model_performance(X_train_resampled_fair_smote, X_test, sensitive_attributes, label,
                                                            groups, model=model)
# Initialize a list to hold DataFrames
data_frames = []

# Populate the list with DataFrames, each having a unique row index
for key, values in results_fair_smote.items():
    df_part = pd.DataFrame([values], index=[key])
    data_frames.append(df_part)

# Concatenate all DataFrames into a single DataFrame
results_fair_smote_df = pd.concat(data_frames)
results_fair_smote_df.index.name = 'Comparison'

# Print the results DataFrame
print(results_fair_smote_df)