
# Model Pipeline for Predicting Soil Parameters

## Description

This project contains a Python script, `run_project.py`, designed to run the entire model pipeline for predicting soil
parameters such as soil organic matter (SOM), total carbon (C), or total nitrogen (N). 
The script coordinates the workflow from data loading and preprocessing through model fitting and 
evaluation to outputting the results. It supports user interaction to specify the parameter to 
predict, the model to use, and various hyperparameters for the modeling process.

## Prerequisites

Before running the script, ensure you have the necessary libraries installed. 
You can install them using:

```bash
pip install -r requirements.txt
```

## Usage

To run the script, navigate to the directory containing `run_project.py` and execute the following 
command in your terminal:

```bash
python run_project.py
```

### User Inputs

The script will prompt you to enter the following details:

1. **Parameter to Predict**:
   - Choose from `som`, `c`, or `n`.

2. **Number of Bootstrap Iterations**:
   - Enter the desired number of iterations for the bootstrap analysis.

3. **Test Data Size**:
   - Specify the proportion of the dataset to be used as the test set (e.g., `0.2` for 80% training and 20% test).

4. **Model Choice**:
   - Choose the model to run:
     - Enter `1` for Elastic Net
     - Enter `2` for Gradient Boosting Tree
     - Enter `3` for Stacked Model

### Output

The script will generate output folders based on the selected options and parameters. 
The structure is as follows:

```
./outputs/
└── <model_name>/
    └── <num_iterations>_iterations/
        └── <train_percentage>_train_<test_percentage>_test/
            └── <PARAMETER>/
                ├── image_output/
                └── bootstrap_output/
```

- **`image_output/`**: Contains generated plots and visualizations of model performance.
- **`bootstrap_output/`**: Contains detailed results of the bootstrap analysis in CSV format.


### Important Note on Bootstrapping
Since bootstrapping involves random sampling with replacement, the dataset is randomly split into 
training and testing sets in each iteration based on the specified test data proportion. 
This process is repeated for the chosen number of iterations. Even when using the same number 
of iterations and the same split ratio, the randomness in sample selection can lead to slight 
variations in the final output each time the program is run.


## Author

Nayma Nur
