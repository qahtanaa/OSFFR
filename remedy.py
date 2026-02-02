import sys
sys.path.append('./')  # Add the directory containing remedy_cust.py to the Python path
from remedy_cust import *

columns_all = X_train.drop(columns=[label])
label_y = label
columns_protected = sensitive_attributes

#====================================================

#names = ["Output of get_temp function, all of the attributes for given group"]
#temp2 = ["Output of get_temp function, sum count by group"]
temp2, names = get_temp(X_train, columns_protected, label_y)
print(temp2, names)
#temp2 counts the number of instances for each combination of sensitive attr and label (like my group count)
#names is the list of sensit attr
unfair_group, unfair_names, skew_candidates, unfair_dict = get_unfair_group(columns_protected, [])
#all empty beside skew candidates that is the list of sensitive attr
print(unfair_group, unfair_names, skew_candidates, unfair_dict)
#all_names is a dict that has 0:[], 1:sens attr1, 2:sens attr2, 3:sens attr1, sens attr2. So for each key (number) it
#associates all possible combinations of sens attr
all_names = candidate_groups(skew_candidates, unfair_dict, columns_protected, unfair_names)
#name values is a dict with sens attrib: possible values
names_values = name_val_dict(X_train, names)

all_names_lst = list(all_names.keys())[1:] # CHANGED HERE
all_names_lst.reverse()
#all_names_lst is a list of the keys of the dict all_names, so numbers [3,2,1]
all_names_lst

#====================================================

#get all of the candidate groups possible with the combos and names
filter_count = 30
#copy of the dataset
new_train_data = copy.deepcopy(X_train)
print(new_train_data.shape[0])

#iterate over all the names to get the temp2 df for each name
for a in all_names_lst:
  print("?????/////")
  print(a)
  #temp2 counts the number of instances for each combination of sensitive attr and label (like my group count)
  #names is the list of sensit attr
  temp2, names = get_temp(new_train_data, all_names[a], label_y)
  print(temp2, '\n', names, 'temp2,names \n')
  #temp is a df with the entire columns, where the columns are in the first iteration all sens attr and the label
  # in the second iteration are one sens attrib and the label
  # temp_g counts the instances considering only the sens attrib (first iteration both sens attr, then only one at the time)
  temp, temp_g = get_temp_g(new_train_data, names, label_y)
  print(temp,'\n', temp_g, 'temp,temp_g \n')
  temp_g = temp_g[temp_g['cnt'] > filter_count]
  #lst_of_counts is a list of df, the first df have first sens attr and the credit + count, the second is second sens attr
  #and the credit + count
  lst_of_counts = compute_lst_of_counts(temp, names, label_y)
  print(lst_of_counts, 'listof counts \n')

  need_pos, need_neg = compute_problematic_opt(temp2, temp_g, names, label_y, lst_of_counts)
  print("The sets of need pos and neg are")
  print(need_pos)
  print(need_neg)
  new_train_data['skewed'] = 0
  new_train_data["diff"] = 0
  print("started duplication")
  new_train_data = naive_duplicate(new_train_data, temp2, names, need_pos, need_neg, label_y)
  print(new_train_data.shape[0])
  print("label y ", new_train_data[label_y].value_counts())
#new_train_x = pd.DataFrame(new_train_data, columns = columns_all)
new_train_label = pd.DataFrame(new_train_data, columns = [label_y])
new_train_label = new_train_label[label_y]
new_train_label = new_train_label.astype('int')

#====================================================

X_train_resampled_remedy = new_train_data.drop(columns= ['skewed', 'diff'])
# Evaluate model performance
results_remedy, pred_labels_remedy = 
		evaluate_model_performance(X_train_resampled_remedy, 
				X_test, sensitive_attributes, label, groups, model= model)

#====================================================

# Initialize a list to hold DataFrames
data_frames = []

# Populate the list with DataFrames, each having a unique row index
for key, values in results_remedy.items():
    df_part = pd.DataFrame([values], index=[key])
    data_frames.append(df_part)

# Concatenate all DataFrames into a single DataFrame
results_remedy_df = pd.concat(data_frames)
results_remedy_df.index.name = 'Comparison'

# Print the results DataFrame
print(results_remedy_df)

#====================================================

