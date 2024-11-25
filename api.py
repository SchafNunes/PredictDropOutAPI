from flask import Flask, request, jsonify

import pandas as pd

import joblib

app = Flask(__name__)

modelo = joblib.load('Dropout.pkl')

@app.route("/predict_titanic", methods=['POST'])
def predict_titanic():

    features = ['Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance', 'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'International', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP']

    dados = request.json

    if not all(feature in dados for feature in features):
        return jsonify("Dicionário com informação incompleta")
        
    try:

        df = pd.DataFrame(dados, index=[0])
        resultado = modelo.predict(df)

        response = {
            'descrição': 'Evasão' if resultado[0] == 1 else 'Matriculado'
        }
    
    except Exception as e:
        return jsonify("Erro ao inferir o resultado. Erro -> " + str(e))

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
