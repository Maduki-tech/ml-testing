from sklearn.tree import DecisionTreeClassifier

def train_and_predict(X_train, X_test, y_train):
    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    
    # Make predictions on the testing data
    predictions = clf.predict(X_test)
    
    return predictions

