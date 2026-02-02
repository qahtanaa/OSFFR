class ModelEvaluator:
    def __init__(self, df, protected_attributes, label_name, privileged, unprivileged, fav, unfav, groups, num_iterations=10, oversampling_methods=None):
        self.df = df
        self.protected_attributes = protected_attributes
        self.label_name = label_name
        self.privileged = privileged
        self.unprivileged = unprivileged
        self.fav = fav
        self.unfav = unfav
        self.groups = groups
        self.num_iterations = num_iterations
        self.oversampling_methods = oversampling_methods if oversampling_methods is not None else ['none']

    def evaluate_model_performance_mean(self):
        results_dict = {method: [] for method in self.oversampling_methods}

        for _ in range(self.num_iterations):
            data_prep = DataPreparation(self.df, self.protected_attributes, self.label_name,
                                        self.privileged, self.unprivileged, self.fav, self.unfav)
            data_prep.prepare()
            data_prep.df = data_prep.df.reset_index(drop=True)
            X_train, X_test = data_prep.X_train, data_prep.X_test
            X_train = X_train.reset_index(drop=True)
            cat_features = data_prep.cat_features
            numerical_features = data_prep.numerical_features
            reverse_group_mapping = data_prep.create_group_column()
            group_counts_train = X_train['Group'].value_counts().sort_index()
            subgroup_column_train = X_train['Group']
            subgroup_column_test = X_test['Group']
            X_train = X_train.drop(columns=['Group'])
            X_test = X_test.drop(columns=['Group'])

            num_privileged_ones = X_train[(X_train[self.protected_attributes[0]] == 1) &
                                          (X_train[self.protected_attributes[1]] == 1) &
                                          (X_train[self.label_name] == 1)].shape[0]
            num_privileged_zeros = X_train[(X_train[self.protected_attributes[0]] == 1) &
                                          (X_train[self.protected_attributes[1]] == 1) &
                                          (X_train[self.label_name] == 0)].shape[0]
            total_ratio = num_privileged_ones / num_privileged_zeros if num_privileged_zeros != 0 else float('inf')

            for method in self.oversampling_methods:
                X_train_method = X_train.copy()
                if method == 'none':
                    results, pred_labels = evaluate_model_performance(X_train_method, X_test, self.protected_attributes, self.label_name,
                                                                      self.groups, model=model)
                elif method == 'custom_smote_km':
                    X_train_no_sens = X_train_method.drop(columns=[self.protected_attributes[0], self.protected_attributes[1], self.label_name])
                    X_reduced = X_train_no_sens

                    kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=42)
                    clusters = kmeans.fit_predict(X_reduced)
                    X_train_method['Cluster_Labels'] = clusters

                    all_synthetic_samples_km = oversample_clusters(X_train_method, 'Cluster_Labels', self.protected_attributes,
                                                                   self.label_name, total_ratio, cat_features)
                    X_train_method = X_train_method.drop(columns=['Cluster_Labels'])
                    X_train_resampled_km = pd.concat([X_train_method, all_synthetic_samples_km], ignore_index=True)
                    results, pred_labels = evaluate_model_performance(X_train_resampled_km, X_test, self.protected_attributes,
                                                                      self.label_name, self.groups, model=model)
                elif method == 'custom_smote_dbscan':
                    synthetic_samples_matrix_dbscan, synthetic_samples_group_dbscan = oversample_groups(X_train_method, cat_features, custom_smote_dbscan, subgroup_column_train, total_ratio, reverse_group_mapping)
                    X_train_resampled_dbscan = pd.concat([X_train_method, pd.DataFrame(synthetic_samples_matrix_dbscan, columns=X_train.columns)], ignore_index=True)
                    results, pred_labels = evaluate_model_performance(X_train_resampled_dbscan, X_test, self.protected_attributes,
                                                                      self.label_name, self.groups, model=model)
                elif method == 'custom_smote_tax':
                    synthetic_samples_matrix_tax, synthetic_samples_group_tax = oversample_groups(X_train_method, cat_features, custom_smote_tax, subgroup_column_train, total_ratio, reverse_group_mapping)
                    X_train_resampled_tax = pd.concat([X_train_method, pd.DataFrame(synthetic_samples_matrix_tax, columns=X_train.columns)], ignore_index=True)
                    results, pred_labels = evaluate_model_performance(X_train_resampled_tax, X_test, self.protected_attributes,
                                                                      self.label_name, self.groups, model=model)
                elif method == 'Fair-SMOTE':
                    X_train_resampled_fair_smote = oversample_fair_smote(X_train_method, self.protected_attributes, self.label_name)
                    results, pred_labels = evaluate_model_performance(X_train_resampled_fair_smote, X_test, self.protected_attributes,
                                                                      self.label_name, self.groups, model=model)
                elif method == 'Reweighing':
                    custom_data = CustomDataset(X_train_method, self.protected_attributes, self.label_name)
                    privileged_groups = [{self.protected_attributes[0]: 1, self.protected_attributes[1]: 1}]
                    unprivileged_groups = [{self.protected_attributes[0]: 0, self.protected_attributes[1]: 0}]
                    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)
                    RW.fit(custom_data)
                    dataset_transf_train = RW.transform(custom_data)

                    results, pred_labels = evaluate_model_performance(X_train_method, X_test, self.protected_attributes, self.label_name,
                                                                      self.groups, model=model, weights=custom_data.instance_weights)
                elif method == 'Remedy':
                    columns_all = X_train_method.drop(columns=[self.label_name])
                    label_y = self.label_name
                    columns_protected = self.protected_attributes
                    temp2, names = get_temp(X_train_method, columns_protected, label_y)
                    unfair_group, unfair_names, skew_candidates, unfair_dict = get_unfair_group(columns_protected, [])
                    print(unfair_group, unfair_names, skew_candidates, unfair_dict)
                    all_names = candidate_groups(skew_candidates, unfair_dict, columns_protected, unfair_names)
                    names_values = name_val_dict(X_train_method, names)

                    all_names_lst = list(all_names.keys())[1:]
                    all_names_lst.reverse()
                    filter_count = 30
                    new_train_data = copy.deepcopy(X_train_method)

                    for a in all_names_lst:
                        temp2, names = get_temp(new_train_data, all_names[a], label_y)
                        temp, temp_g = get_temp_g(new_train_data, names, label_y)
                        temp_g = temp_g[temp_g['cnt'] > filter_count]
                        lst_of_counts = compute_lst_of_counts(temp, names, label_y)
                        need_pos, need_neg = compute_problematic_opt(temp2, temp_g, names, label_y, lst_of_counts)
                        new_train_data['skewed'] = 0
                        new_train_data["diff"] = 0
                        new_train_data = naive_duplicate(new_train_data, temp2, names, need_pos, need_neg, label_y)
                    new_train_label = pd.DataFrame(new_train_data, columns=[label_y])
                    new_train_label = new_train_label[label_y]
                    new_train_label = new_train_label.astype('int')

                    X_train_resampled_remedy = new_train_data.drop(columns=['skewed', 'diff'])

                    results, pred_labels = evaluate_model_performance(X_train_resampled_remedy, X_test, self.protected_attributes,
                                                                      self.label_name, self.groups, model=model)

                data_frames = []
                for key, values in results.items():
                    df_part = pd.DataFrame([values], index=[key])
                    data_frames.append(df_part)

                results_df = pd.concat(data_frames)
                results_df.index.name = 'Comparison'
                results_dict[method].append(results_df)

        combined_results = {method: pd.concat(results_dict[method]).groupby(level=0).mean() for method in self.oversampling_methods}
        return combined_results
