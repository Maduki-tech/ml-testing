# ml-testing

## Installation

- Create a virtual **env**
```bash
python3 -m venv env
```

- Install dependencies
```bash
pip install -r requirements.txt
```

## Code explained

### Functions

#### preprocessing.py

- `load_iris()`: Loads the Iris dataset from scikit-learn.
- `train_test_split()`: Splits the dataset into training and testing sets.
- `StandardScaler()`: Normalizes the feature matrix so that each feature has a mean value of 0 and a standard deviation of 1.
  This is important for many algorithms to perform well.

#### model.py

- `DecisionTreeClassifier()`: Creates a Decision Tree classifier. The random_state parameter is set for reproducibility.
- `fit()`: Trains the classifier on the training data.
- `predict()`: Makes predictions on the testing data.

## Writing tests

### preprocessing.py

- Ensure the function returns data split into training and testing sets correctly.
- Verify the shape of the returned arrays to ensure they are consistent with expectations (e.g., `X_train` and `y_train` have the same length).
- Check if the data normalization is applied correctly (this might require checking if the mean is close to 0 and the standard deviation is close to 1, though such tests can be a bit more involved and might not be strictly necessary for a simple practice project).

#### model.py

- Ensure the `train_and_predict` function returns predictions.
- You might also want to check that the predictions are of the correct length (matching the test set size) and possibly that the predictions are within the expected range of values (for the Iris dataset, this would be 0, 1, or 2, corresponding to the three species of Iris).
