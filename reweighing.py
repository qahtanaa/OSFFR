new_path = ['./']
sys.path = new_path
from reweighing_cust import Reweighing

class CustomDataset:
    def __init__(self, data, sensitive_attributes, label):
        self.data = data
        self.protected_attribute_names = sensitive_attributes
        self.protected_attributes = np.column_stack([data[sensitive_attributes[0]], data[sensitive_attributes[1]]])
        self.labels = data[label].astype(float).values.reshape(-1, 1)
        self.favorable_label = 1.0
        self.unfavorable_label = 0.0
        self.instance_weights = np.ones(len(data))
        self.privileged_protected_attributes = [np.array([1.]), np.array([1.])]
        self.unprivileged_protected_attributes = [np.array([0.]), np.array([0.])]
        self.feature_names = data.columns[data.columns != label].tolist()

custom_data = CustomDataset(X_train, sensitive_attributes, label)

#====================================================

privileged_groups = [{sensitive_attributes[0]: 1, sensitive_attributes[1]: 1}]
unprivileged_groups = [{sensitive_attributes[0]: 0, sensitive_attributes[1]: 0}]
RW = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)

RW.fit(custom_data)
dataset_transf_train = RW.transform(custom_data)

#====================================================


results_reweighing, pred_labels_reweighing = evaluate_model_performance(X_train, X_test, sensitive_attributes, label,
                                                                        groups, model = model,
                                                                        weights=custom_data.instance_weights)
# Initialize a list to hold DataFrames
data_frames = []

# Populate the list with DataFrames, each having a unique row index
for key, values in results_reweighing.items():
    df_part = pd.DataFrame([values], index=[key])
    data_frames.append(df_part)

# Concatenate all DataFrames into a single DataFrame
results_reweighing_df = pd.concat(data_frames)
results_reweighing_df.index.name = 'Comparison'

# Print the results DataFrame
print(results_reweighing_df)




