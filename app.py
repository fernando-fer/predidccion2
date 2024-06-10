from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request y convertirlos a flotantes
        L = float(request.form['L'])
        R = float(request.form['R'])
        A_M = float(request.form['A_M'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[L, R, A_M]], columns=['L', 'R', 'A_M'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        predicted_class = str(prediction[0])  # Convertir la predicción a cadena
        
        app.logger.debug(f'Predicción: {predicted_class}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': predicted_class})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': 'Error en la solicitud. Detalles en el registro del servidor.'}), 400

if __name__ == '__main__':
    app.run(debug=True)