from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


red_df = pd.read_csv('./winequality-red.csv', sep=';')
white_df = pd.read_csv('./winequality-white.csv', sep=';')

red_df['goodquality'] = [1 if x > 5.0 else 0 for x in red_df['quality']]
white_df['goodquality'] = [1 if x > 5.0 else 0 for x in white_df['quality']]

red_df = red_df.drop('quality', axis=1)
white_df = white_df.drop('quality', axis=1)

X = pd.concat([red_df, white_df])

y = X['goodquality']
X = X.drop('goodquality', axis=1)

# normalize
X_features = X
X = StandardScaler().fit_transform(X)
        
# split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=5, shuffle=True)    

model = SVC(C=10.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(X_train, y_train))
# print(classification_report(y_test, y_pred))








