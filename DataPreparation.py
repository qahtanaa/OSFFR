import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

class DataPreparation():
    """
    ........
    """
    def __init__(self, df, sensitive, label, priv, unpriv, fav, unfav, categorical=[]):
        """
        Construct all necessary attributes for the data preparation.

        df : (pandas DataFrame) containing the data
        sensitive : (list(str)) specifying the column names of all sensitive features
        label : (str) specifying the label column
        priv : (list(dicts)) representation of the privileged groups
        unpriv : (list(dicts)) representation of the unprivileged groups
        fav : (str/int/..) value representing the favorable label
        unfav : (str/int/..) value representing the unfavorable label
        categorical : (list(str)) (optional) specifying column names of categorical features
        """
        self.df = df
        self.sensitive = sensitive
        self.label = label
        self.priv = priv
        self.unpriv = unpriv
        self.fav = fav
        self.unfav = unfav
        self.categorical = categorical

    def detect_missing_values(self):
        """
        Detect rows with missing values and remove them from the DataFrame.
        """
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        removed_rows = initial_rows - len(self.df)

        if removed_rows > 0:
            print(f"Detected {removed_rows} rows with missing values. Removed them.")
        else:
            print("No missing values detected.")  # pass

    def binary_label(self):
        """
        Ensure the decision label and sensitive attributes are encoded as binary, where:
        - Favorable label and privileged groups are encoded as 1.
        - Unfavorable label and unprivileged groups are encoded as 0.
        """
        if len(self.priv) != 2 or len(self.unpriv) != 2:
            raise ValueError("Both 'priv' and 'unpriv' must contain exactly two values.")

        number_label_values = self.df[self.label].nunique()
        if number_label_values == 2:
            print(f"The '{self.label}' column has only two unique values.")
            self.df[self.label] = (self.df[self.label].map({self.unfav: 0, self.fav: 1}).astype(int))
        else:
            print(f"The '{self.label}' column does not have exactly two unique values, as it should.")

        # Create mappings for each sensitive attribute
        race_mapping = {self.priv[0]: 1, self.unpriv[0]: 0}
        sex_mapping = {self.priv[1]: 1, self.unpriv[1]: 0}

        # Apply the mappings to the respective columns
        self.df[self.sensitive[0]] = (
            self.df[self.sensitive[0]]
            .replace(race_mapping)
        )

        self.df[self.sensitive[1]] = (
            self.df[self.sensitive[1]]
            .replace(sex_mapping)
        )

        self.df[self.sensitive[0]] = self.df[self.sensitive[0]].astype(int)
        self.df[self.sensitive[1]] = self.df[self.sensitive[1]].astype(int)

    def find_categorical_attributes(self):
        """
        Identify categorical attributes and encode.
        """
        self.attribute_types = {}

        for column in self.df.columns:
            if column == 'Group':
                continue  # Skip the 'Group' column
            elif column in self.categorical:
                self.attribute_types[column] = 'Categorical'
            elif self.df[column].nunique() == 2:
                self.attribute_types[column] = 'Categorical'
            else:
                num_float = 0
                num_text = 0
                thresh = 0.99
                num_att_in_column = len(self.df[column])

                for value in self.df[column]:
                    try:
                        float(value)
                        num_float += 1
                    except ValueError:
                        num_text += 1

                if num_float / num_att_in_column > thresh:
                    self.attribute_types[column] = 'Numerical'
                else:
                    self.attribute_types[column] = 'Categorical'
        # Boolean
        self.cat_features = []
        for attr in self.attribute_types:
            self.cat_features.append(self.attribute_types[attr] == 'Categorical')

        encoder_dict = dict()
        self.columns_categorical = self.df.columns[self.cat_features]

        for column in self.columns_categorical:
            le = LabelEncoder()
            encoded = le.fit_transform(self.df[column].astype(str).values)
            self.df[column] = pd.Series(encoded, index=self.df.index, dtype="int64")
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            encoder_dict[column] = mapping
        print(encoder_dict, 'encoder dict')
        self.numerical_features = [not feature for feature in self.cat_features]
        self.columns_numerical = self.df.columns[self.numerical_features]

        for column in self.columns_numerical:
            self.df.loc[:, column] = self.df[column].astype(float)

        return self.attribute_types, self.cat_features, self.numerical_features

    def create_group_column(self):
        """
        Create a 'Group' column in the DataFrame based on protected attributes, privileged/unprivileged conditions, and label.
        """
        group_combinations = pd.MultiIndex.from_product([self.df[sensitive].unique() for sensitive in self.sensitive] + [self.df[self.label].unique()], names=self.sensitive + [self.label])
        print(list(enumerate(group_combinations)))
        # Create a mapping between group combinations and their corresponding numbers
        group_mapping = {group: idx for idx, group in enumerate(group_combinations)}
        reverse_group_mapping = {idx: group for group, idx in group_mapping.items()}
        self.df['Group'] = pd.MultiIndex.from_frame(self.df[self.sensitive + [self.label]]).map(group_mapping)

        return reverse_group_mapping

    def train_test_split(self):
        X = self.df.loc[:, self.df.columns != self.label]
        y = self.df[self.label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=self.df['Group'])  # , random_state=42

        return self.X_train, self.y_train, self.X_test, self.y_test

    def standardization_numerical(self):

        numerical_cols = [
            col for col in self.X_train.columns
            if col in self.columns_numerical
        ]

        train_dataset_numerical = self.X_train[numerical_cols]
        test_dataset_numerical = self.X_test[numerical_cols]
        scaler = StandardScaler()
        scaler.fit(train_dataset_numerical)
        self.X_train[numerical_cols] = scaler.transform(train_dataset_numerical)
        self.X_test[numerical_cols] = scaler.transform(test_dataset_numerical)
        self.X_train = pd.concat(
            [self.X_train.reset_index(drop=True),
             self.y_train.reset_index(drop=True)],
            axis=1
        )

        self.X_test = pd.concat(
            [self.X_test.reset_index(drop=True),
             self.y_test.reset_index(drop=True)],
            axis=1
        )

        return self.X_train, self.X_test

    def prepare(self):
        """
        Perform all preprocessing steps.
        """
        self.detect_missing_values()
        self.binary_label()
        self.find_categorical_attributes()
        self.create_group_column()
        self.train_test_split()
        self.standardization_numerical()
        return self
