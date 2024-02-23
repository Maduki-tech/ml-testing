from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    # Load the Iris dataset
    data = load_iris()
    X, y = data.data, data.target  # X is the feature matrix, y is the target array
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the feature matrix to have a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit to data, then transform it
    X_test_scaled = scaler.transform(X_test)  # Perform standardization by centering and scaling
    
    return X_train_scaled, X_test_scaled, y_train, y_test

"""
Comment this in and run the file to see how the Iris dataset looks like
"""
# def show_data_set():
#     _, ax = plt.subplots()
#     ax.set_title('Iris Dataset')
#     ax.set_xlabel('Sepal length (cm)')
#     ax.set_ylabel('Sepal width (cm)')
#     ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train)
#     plt.show()
#
# if __name__ == '__main__':
#     X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data()
#     show_data_set()
#

