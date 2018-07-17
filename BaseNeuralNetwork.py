import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Set where our data is coming from and read it in
input_file = "adult_encoded.data"
df = pd.read_csv(input_file, header=0)

# Gets all rows that aren't titled salary (features)
X = df.loc[:, df.columns != "salary"]
X_split = np.array_split(X, 2)
X_train = X_split[0]
X_test = X_split[1]

# Gets the salary row (classified)
y = df.loc[:, df.columns == "salary"]
y_split = np.array_split(y, 2)
y_train = y_split[0]
y_test = y_split[1]

print "Debug: X Training Set Shape:" + str(X_train.shape)
print "Debug: y Training Set Shape:" + str(y_train.shape)

print "Debug: X Testing Set Shape:" + str(X_test.shape)
print "Debug: y Testing Set Shape:" + str(y_test.shape)

# Create our classifier
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(12, 2), random_state=None)

# Train with the training data
clf.fit(X_train, np.ravel(y_train))

# Make predictions and store them in an array, like the last homework
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Output prediction accuracy
print accuracy_score(y_train, y_train_pred)
print accuracy_score(y_test, y_test_pred)
