# OSFFR: An Oversampling Framework for Mitigating Representation Bias
OSFFR is a pre-processing framework designed to mitigate representation bias in datasets. This framework focuses on improving the balance of class labels across different groups within a dataset by using a new oversampling technique.

## Key Features

- **Group-Based Analysis**: OSFFR divides the dataset into groups based on combinations of protected attributes.
- **Imbalance Ratio Identification**: The framework identifies groups with lower ratios of receiving favorable class labels and targets these groups for balancing.
- **Boundary Sample Identification**: Uses the DBSCAN algorithm to identify samples with favorable class labels at the boundaries of each group.
- **Synthetic Sample Generation**: Utilizes the SMOTE-NC algorithm to generate synthetic samples around the group boundary, achieving a balanced ratio compared to other groups.

## Usage

### Reproducing the Results

This repository includes the code and datasets used to generate the results and images presented in the associated paper. The datasets provided are for binary classification and include binary labels and binary protected attributes.

#### Experiments

1. **Oversampling Comparison by Imbalance Ratio vs. Equal Sample Size**

   - File: `Comparison_IR_EqualSize_OS.ipynb`
   - Description: This notebook compares the effectiveness of oversampling based on equal sample size versus oversampling based on the Imbalance Ratio.
   - Usage: Upload one of the three provided datasets and run the notebook.

2. **Comparison with Other Methods from the Literature**

   - File: `ComparisonwOthers.ipynb`
   - Description: This notebook compares the proposed framework with existing methods in the literature.
   - Usage: To use this notebook, upload the selected dataset along with the `remedy_cust.py` and `reweighing_cust.py` files, and then run the notebook.



