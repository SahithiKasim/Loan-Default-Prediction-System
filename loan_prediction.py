import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import RegressorChain 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR

# Load data
df = pd.read_csv('Data.csv')

# Data preprocessing
df_X = df.iloc[:, 2:13].copy()
df_X = pd.get_dummies(df_X)

df_y1 = df.iloc[:, 13:16].copy()
df_y1 = pd.get_dummies(df_y1)

df_y2 = df.iloc[:, 16:20].copy()
df_y2 = pd.get_dummies(df_y2)

# Data extraction & manipulation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_X)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(X_scaled['text_column'])
df_X = pd.concat([df_X.drop('text_column', axis=1), pd.DataFrame(X_tfidf.toarray())], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y1, test_size=0.2, random_state=0)

def accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(model.score(X_test, y_test))
    errors = np.abs(predictions - y_test)
    mape = 100 * (errors / y_test)
    for col in mape:
        accuracy = 100 - np.mean(mape[col])
        print('Accuracy:', round(accuracy, 2), '%.')

# Gradient Boosting Regressor
models = [
    GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0),
    GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=0),
    GradientBoostingRegressor(n_estimators=300, max_depth=7, random_state=0)
]

for idx, model in enumerate(models, start=1):
    model_name = f"GradientBoostingRegressor_{idx}"
    model.fit(X_train, y_train)
    joblib.dump(model, f"{model_name}.pkl")
    print(f"Model {idx} ({model_name}): ")
    accuracy(model, X_test, y_test)

# SVM Regressor
models = [
    SVR(kernel='rbf', C=1.0, epsilon=0.1),
    SVR(kernel='linear', C=1.0, epsilon=0.1),
    SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1),
]

for idx, model in enumerate(models, start=1):
    model_name = f"SVR_{idx}"
    model1 = RegressorChain(model, cv=3)
    model2 = MultiOutputRegressor(model, n_jobs=-1)
    
    # Fit and save the models for y1
    model1.fit(X_train, y_train)
    joblib.dump(model1, f"{model_name}_RegressorChain.pkl")
    print(f"Model {idx} ({model_name} - RegressorChain): ")
    accuracy(model1, X_test, y_test)

    model2.fit(X_train, y_train)
    joblib.dump(model2, f"{model_name}_MultipleOutput.pkl")
    print(f"Model {idx} ({model_name} - MultipleOutput): ")
    accuracy(model2, X_test, y_test)
