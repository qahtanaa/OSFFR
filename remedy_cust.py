from sympy import Symbol
from sympy.solvers import solve
import copy
import time
import itertools
import pandas as pd


def get_unfair_group(columns_protected, list_parse, entire = 1):
  unfair_group = []
  unfair_dict = {}
  names = []
  for col in columns_protected:
    found = False
    for item in list_parse:
      attr_given = item.split("=")[0]
      if col == attr_given:
        unfair_group.append(int(item.split("=")[1]))
        names.append(attr_given)
        unfair_dict[attr_given] = int(item.split("=")[1])
        found = True
  # if use the entire dataset
  if entire:
    return unfair_group, names, columns_protected, unfair_dict
        # break
    # if found == False:
    #   unfair_group.append(-1)
  return unfair_group, names, list(set(columns_protected).symmetric_difference(set(names))), unfair_dict


def candidate_groups(skew_candidates, unfair_dict, ordering, names):
  candidate_combos = []
  candidate_ind = {}
  num = 0
  for i in range(len(skew_candidates)+1):
    temp_candidate = list(itertools.combinations(skew_candidates, i))
    for tc in temp_candidate:
      candidate_ind[num] = list(tc)
      num += 1
  return candidate_ind


def name_val_dict(train_set,names):
  names_values = {}
  for n in names:
    names_values[n] = list(train_set[n].unique())
  return names_values


def get_temp(train_set, names, y_label):
  names2 = copy.deepcopy(names)
  names2.append(y_label)
  print(names2)
  temp = train_set[names2]
  temp['cnt'] = 0
  temp2 = temp.groupby(names2)['cnt'].count().reset_index()
  temp2['cnt'].sum()
  return temp2, names


def get_temp_g(train_set, names, y_label):
  names2 = copy.deepcopy(names)
  names2.append(y_label)
  temp = train_set[names2]
  temp['cnt'] = 0
  temp_g = temp.groupby(names)['cnt'].count().reset_index()
  return temp, temp_g


def compute_lst_of_counts(temp, names, label_y):
    # get the list of group-by attributes
    lst_of_counts = []
    for i in range(len(names)):
        grp_names = copy.copy(names)
        del grp_names[i]
        grp_names.append(label_y)
        temp_df = temp.groupby(grp_names)['cnt'].count()
        lst_of_counts.append(temp_df)
    return lst_of_counts


# helper function for optimized
def compute_neighbors_opt(group_lst,lst_of_counts, pos, neg):
    #start2 = time.time()
    times = len(group_lst)
    pos_cnt = 0
    neg_cnt = 0
    for i in range(times):
        df_groupby = lst_of_counts[i]
        temp_group_lst_pos = copy.copy(group_lst)
        temp_group_lst_neg = copy.copy(group_lst)
        del temp_group_lst_pos[i]
        del temp_group_lst_neg[i]
        # count positive
        temp_group_lst_pos.append(1)
        group_tuple_pos = tuple(temp_group_lst_pos)
        if group_tuple_pos in df_groupby.keys():
            pos_cnt += df_groupby[group_tuple_pos]
        else:
            pos_cnt += 0
        # count negative
        temp_group_lst_neg.append(0)
        group_tuple_neg = tuple(temp_group_lst_neg)
        if group_tuple_neg in df_groupby.keys():
            neg_cnt += df_groupby[group_tuple_neg]
        else:
            neg_cnt += 0
    pos_val = pos_cnt - times* pos
    neg_val = neg_cnt - times* neg
    #end2 = time.time()
    #print("The time to compute the neighbor counts for " +  str(group_lst) +" is " + str(end2-start2))
    if neg_val == -1 or (neg_val == 0 and pos_val == 0):
        return (pos_val, neg_val, -1)
    if pos_val == 0 or neg_val == 0:
        return (pos_val, neg_val, 0)
    # print("here", pos_val, neg_val, pos_val/neg_val)

    return (pos_val, neg_val, pos_val/neg_val)


#Function to determine based on the neighbors if the group is positive or negative
def determine_problematic_opt(group_lst, names, temp2, lst_of_counts, label_y, threshold= 0.3):
    #0: ok group, 1: need negative records, 2: need positive records
    d = copy.copy(temp2)
    for i in range(len(group_lst)):
        d = d[d[names[i]] == group_lst[i]]
    total =  d['cnt'].sum()
    d = d[d[label_y] == 1]
    pos = d['cnt'].sum()
    neg = total - pos
    neighbors = compute_neighbors_opt(group_lst,lst_of_counts, pos, neg)
    if(neighbors[2] == -1):
        # there is no neighbors
        return 0
    if(total > 30):
        # need to be large enough, need to adjust with different datasets.
        if neg == 0:
            if (pos > neighbors[2]):
                return 1
            if(pos <= neighbors[2]):
                return 0
        if (pos/(neg) - neighbors[2] > threshold):
            # too many positive records
            return 1
        if (neighbors[2] - pos/(neg) > threshold):
            return 2
    return 0


#Function to designate if a group is positive or negative
def compute_problematic_opt(temp2, temp_g, names, label, lst_of_counts):
    need_pos = []
    need_neg = []
    for index, row in temp_g.iterrows():
        group_lst = []
        for n in names:
            group_lst.append(row[n])
        problematic = determine_problematic_opt(group_lst, names, temp2, lst_of_counts,label)
        if(problematic == 1):
            if group_lst not in need_neg:
                need_neg.append(group_lst)
        if(problematic == 2):
            if group_lst not in need_pos:
                need_pos.append(group_lst)
    return need_pos, need_neg


def get_one_degree_neighbors(temp2, names, group_lst):
    result = []
    for i in range(len(group_lst)):
        d = copy.copy(temp2)
        for k in range(len(group_lst)):
            if k != i:
                d = d[d[names[k]] == group_lst[k]]
            else:
                d = d[d[names[k]] != group_lst[k]]
        result.append(d)
    return result


def compute_neighbors(group_lst, result, label_y):
    # compute the ratio of positive and negative records
    start2 = time.time()
    pos = 0
    neg = 0 
    for r in result:
        total  = r['cnt'].sum()
        r = r[r[label_y] == 1]
        pos += r['cnt'].sum()
        neg += total - r['cnt'].sum()
    if(neg == 0):
        return (pos, neg, -1)
    end2 = time.time()
    
    return(pos, neg, pos/neg)


def compute_diff_add(group_lst, temp2, names, label_y, need_positive_or_negative):

    d = copy.copy(temp2)

    for i in range(len(group_lst)):

        d = d[d[names[i]] == group_lst[i]]
    total =  d['cnt'].sum()
    d = d[d[label_y] == 1]
    pos = d['cnt'].sum()
    neg = total - pos
    result = get_one_degree_neighbors(temp2, names, group_lst)
    neighbors = compute_neighbors(group_lst, result, label_y)
    if(need_positive_or_negative == 1):
        # need pos

        x = Symbol('x')
        try:
          diff = solve((pos + x)/ neg -  neighbors[2])[0]
        except:
          return -1

        print(neighbors[2], pos, neg, diff)
    else:
        #need negative
        x = Symbol('x')
        try:
          diff = solve(pos/ (neg + x) -  neighbors[2])[0]
        except:
          return -1
    print(neighbors[2], pos, neg, diff)
    return diff


def round_int(x):
    if x in [float("-inf"),float("inf")]: return 0
    return int(round(x))


def make_duplicate(d, group_lst, diff, label_y, names, need_positive_or_negative):
    selected = copy.deepcopy(d)
    print("names ", names, group_lst)
    for i in range(len(group_lst)):
        att_name = names[i]
        selected = selected[(selected[att_name] == group_lst[i])]
    selected = selected[(selected[label_y] == need_positive_or_negative)]

    if len(selected) == 0:
        return pd.DataFrame()
    while(len(selected) < diff):
        # duplicate the dataframe
        select_copy = selected.copy(deep=True)
        selected = pd.concat([selected, select_copy])

        # the number needed is more than the not needed numbers.

    generated = selected.sample(n = diff, replace = False, axis = 0)

    return generated


def naive_duplicate(d, temp2, names, need_pos, need_neg, label_y):
    # add more records for all groups
    # The smote algorithm to boost the coverage
    for r in need_pos:
    # add more positive records
        # determine how many points to add
        diff = compute_diff_add(r, temp2, names, label_y, 1)
        if diff == -1:
          continue
        diff = round_int(diff)
        # add more records
        print("Adding " + str(diff) +" positive records")
        samples_to_add = make_duplicate(d, r, diff, label_y, names, need_positive_or_negative = 1)
        d = pd.concat([d, samples_to_add], ignore_index=True) 
    for k in need_neg:
        diff = compute_diff_add(k, temp2, names, label_y, need_positive_or_negative = 0)
        if diff == -1:
          continue
        diff = round_int(diff)
        print("Adding " + str(diff) +" negative records")
        samples_to_add = make_duplicate(d, k, diff, label_y, names, need_positive_or_negative = 0)
        d = pd.concat([d, samples_to_add], ignore_index=True)
    return d