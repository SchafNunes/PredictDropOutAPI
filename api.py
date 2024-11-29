from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

import pandas as pd

import joblib

app = Flask(__name__)
CORS(app)  # Initialize CORS

modelo = joblib.load('libs/Dropout.pkl')


@app.route("/predict_dropout", methods=['POST'])
def predict_titanic():

    features = ["Estado civil", "Modo de inscricao", "Ordem de inscricao", "Curso", "Periodo (diurno/noturno)", "Qualificacao anterior", "Nacionalidade", "Escolaridade da mae", "Escolaridade do pai", "Ocupacao da mae", "Ocupacao do pai", "Deslocado", "Necessidades educacionais especiais", "Inadimplente", "Mensalidades em dia", "Genero", "Bolsista", "Idade na matricula", "Internacional", "Unidades curriculares 1 semestre (creditadas)", "Unidades curriculares 1 semestre (matriculadas)", "Unidades curriculares 1 semestre (avaliadas)", "Unidades curriculares 1 semestre (aprovadas)", "Nota das unidades curriculares 1 semestre", "Unidades curriculares 1 semestre (nao avaliadas)", "Unidades curriculares 2 semestre (creditadas)", "Unidades curriculares 2 semestre (matriculadas)", "Unidades curriculares 2 semestre (avaliadas)", "Unidades curriculares 2 semestre (aprovadas)", "Nota das unidades curriculares 2 semestre", "Unidades curriculares 2 semestre (nao avaliadas)", "Taxa de desemprego", "Taxa de inflacao", "PIB"]

    dados = request.json

    if not all(feature in dados for feature in features):
        return jsonify("Dicionario com informacao incompleta")
        
    try:
        df = pd.DataFrame(dados, index=[0])
        resultado = modelo.predict(df)

        response = {
            'descricao': 'Evasao' if resultado[0] == 0 else 'Graduado'
        }
    
    except Exception as e:
        return jsonify("Erro ao inferir o resultado. Erro -> " + str(e))

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
