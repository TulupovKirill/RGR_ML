import streamlit as st 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, rand_score
import pickle
import tensorflow as tf
import numpy as np

def load_models():
    knn = pickle.load(open("models/knn.p", 'rb'))
    k_means = pickle.load(open("models/kmeans.p", 'rb'))
    grad_clf = pickle.load(open("models/grad_clf.p", 'rb'))
    stack_clf = pickle.load(open("models/stack_clf.p", 'rb'))
    bag_clf = pickle.load(open("models/bag_clf.p", 'rb'))
    dnn = tf.keras.models.load_model("models/dnn.h5")
    return knn, k_means, grad_clf,bag_clf, stack_clf, dnn

st.title("Модели")

uploaded_file = st.file_uploader("Choose a file .csv", type='csv')
df = pd.read_csv(uploaded_file)
st.write(df.head())

scaler = StandardScaler().fit(df.drop(['name', 'hazardous', 'est_diameter_max'], axis=1))
X = scaler.transform(df.drop(['name', 'hazardous', 'est_diameter_max'], axis=1))
X = pd.DataFrame(X)
Y = df['hazardous']
rus = RandomUnderSampler()

X_resampled, y_resampled = rus.fit_resample(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.2, random_state=42, stratify=y_resampled)

input_data = {}
feature_names = ['id', 'est_diameter_min', 'relative_velocity', 'miss_distance', 'absolute_magnitude']
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=None, placeholder="Type a number...")

button1 = st.button('Сделать предсказание')
button2 = st.button('Проверить на тестах')

knn, k_means, grad_clf,bag_clf, stack_clf, dnn = load_models()

if button1:
    result = []
    input_data = pd.DataFrame([input_data])
    input_data = scaler.transform(input_data)
    
    predict_knn = knn.predict(input_data)[0]
    predict_k_means = k_means.predict(input_data)[0]
    predict_grad = grad_clf.predict(input_data)[0]
    predict_bag = bag_clf.predict(input_data)[0]
    predict_stack = stack_clf.predict(input_data)[0]
    predict_dnn = dnn.predict(input_data)[0]
    
    st.write(f'Результат предсказания KNN - {predict_knn}')
    st.write(f'Результат предсказания K-Means - {predict_k_means}')
    st.write(f'Результат предсказания GradientBoostingClassifier - {predict_grad}')
    st.write(f'Результат предсказания BaggingClassifier - {predict_bag}')
    st.write(f'Результат предсказания StackingClassifier - {predict_stack}')
    st.write(f'Результат предсказания DNN - {predict_dnn}')
     
elif button2:
    predict_knn = knn.predict(X_test)
    predict_k_means = k_means.predict(X_test)
    predict_grad = grad_clf.predict(X_test)
    predict_bag = bag_clf.predict(X_test)
    predict_stack = stack_clf.predict(X_test)
    predict_dnn = dnn.predict(X_test)
    
    st.write(f'Результат метрики f1 - {f1_score(y_test, predict_knn)}')
    st.write(f'Результат метрики rand - {rand_score(k_means.labels_, y_resampled)}')
    st.write(f'Результат метрики f1 - {f1_score(y_test, predict_grad)}')
    st.write(f'Результат метрики f1 -{f1_score(y_test, predict_bag)}')
    st.write(f'Результат метрики f1 - {f1_score(y_test, predict_stack)}')
    st.write(f'Результат метрики f1 - {f1_score(y_test, np.around(dnn.predict(X_test, verbose=None)))}')