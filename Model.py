import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from sklearn.metrics import f1_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier

label_encoder = LabelEncoder()

df = pd.read_csv('libs/main.csv')

df.columns = ["Estado civil", "Modo de inscricao", "Ordem de inscricao", "Curso", "Periodo (diurno/noturno)", "Qualificacao anterior", "Nacionalidade", "Escolaridade da mae", "Escolaridade do pai", "Ocupacao da mae", "Ocupacao do pai", "Deslocado", "Necessidades educacionais especiais", "Inadimplente", "Mensalidades em dia", "Genero", "Bolsista", "Idade na matricula", "Internacional", "Unidades curriculares 1 semestre (creditadas)", "Unidades curriculares 1 semestre (matriculadas)", "Unidades curriculares 1 semestre (avaliadas)", "Unidades curriculares 1 semestre (aprovadas)", "Nota das unidades curriculares 1 semestre", "Unidades curriculares 1 semestre (nao avaliadas)", "Unidades curriculares 2 semestre (creditadas)", "Unidades curriculares 2 semestre (matriculadas)", "Unidades curriculares 2 semestre (avaliadas)", "Unidades curriculares 2 semestre (aprovadas)", "Nota das unidades curriculares 2 semestre", "Unidades curriculares 2 semestre (nao avaliadas)", "Taxa de desemprego", "Taxa de inflacao", "PIB", "Resultado"]

df.to_csv('libs/mainPTBR.csv', index=False)

df = pd.read_csv('libs/mainPTBR.csv')

# print(df)
df = df[df['Resultado'] != 'Enrolled']

df['Resultado'] = label_encoder.fit_transform(df['Resultado'])

classes = label_encoder.classes_
print(f'Classes: {classes}')
for index, class_name in enumerate(classes):
    print(f'{class_name}: {index}')

x = df.drop('Resultado', axis=1)
y = df['Resultado']


x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, stratify = y,  random_state = 5)


search_space = {
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0)
}

xgb = XGBClassifier(n_estimators=100, objective='binary:logistic', random_state=42)

bayes_search = BayesSearchCV(estimator=xgb, search_spaces=search_space, n_iter=25, cv=3, n_jobs=-1, verbose=2)
bayes_search.fit(x_treino, y_treino)


print(f"Best parameters: {bayes_search.best_params_}")
print(f"Best score: {bayes_search.best_score_}")


preds = bayes_search.predict(x_teste)

f1 = f1_score(y_teste, preds, average='weighted')

print(f1)

joblib.dump(bayes_search, 'libs/Dropout.pkl')

modelo = joblib.load('libs/Dropout.pkl')

dados_entrada = np.array([[1,8,5,2,1,1,1,13,10,6,10,1,0,0,1,1,0,20,0,0,0,0,0,0.0,0,0,0,0,0,0.0,0,10.8,1.4,1.74],[1,6,1,11,1,1,1,1,3,4,4,1,0,0,0,1,0,19,0,0,6,6,6,14.0,0,0,6,6,6,13.666666666666666,0,13.9,-0.3,0.79]])

previsao = modelo.predict(dados_entrada)

print(f'Resultado da previsão: {previsao}')
