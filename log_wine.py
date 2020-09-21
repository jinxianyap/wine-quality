from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

red_df = pd.read_csv('./winequality-red.csv', sep=';')
white_df = pd.read_csv('./winequality-white.csv', sep=';')

red_df['category'] = [0 for x in red_df['quality']]
white_df['category'] = [1 for x in white_df['quality']]

red_df = red_df.drop('quality', axis=1)
white_df = white_df.drop('quality', axis=1)

X = pd.concat([red_df, white_df])

y = X['category']
X = X.drop('category', axis=1)

# normalize
X_features = X
X = StandardScaler().fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, shuffle=True)    

model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))