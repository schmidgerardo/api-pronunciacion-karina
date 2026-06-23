import os
import requests
import librosa
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

app = Flask(__name__)
CORS(app)  # Permite que tu app web de Render le consulte sin bloqueos

@app.route('/comparar', methods=['POST'])
def comparar_pronunciacion():
    try:
        # 1. Validar llegada del archivo
        if 'audio_estudiante' not in request.files:
            return jsonify({"success": False, "error": "No se recibio el audio del estudiante en el FormData"}), 400
            
        audio_file = request.files['audio_estudiante']
        word_id = request.form.get('word_id', 'default_word')
        
        # Crear rutas temporales únicas
        path_estudiante = os.path.join("/tmp", f"estudiante_{word_id}.wav")
        path_nativo = os.path.join("/tmp", f"nativo_{word_id}.wav")

        # 2. Guardar el archivo enviado por el cliente
        audio_file.save(path_estudiante)

        try:
            # 3. Leer la huella acústica con Librosa
            # sr=None mantiene la tasa de muestreo original del navegador/móvil
            y_estudiante, sr_estudiante = librosa.load(path_estudiante, sr=None)
            
            # 4. Crear el "Falso Profesor" para pruebas dinámicas
            # Modificamos un poco el tono para que el algoritmo DTW trabaje de verdad
            y_modificado = librosa.effects.pitch_shift(y_estudiante, sr=sr_estudiante, n_steps=2)
            sf.write(path_nativo, y_modificado, sr_estudiante)
            
            # Recargamos el audio de referencia creado
            y_nativo, sr_nativo = librosa.load(path_nativo, sr=None)

            # 5. Extracción de características (MFCCs)
            mfcc_estudiante = librosa.feature.mfcc(y=y_estudiante, sr=sr_estudiante)
            mfcc_nativo = librosa.feature.mfcc(y=y_nativo, sr=sr_nativo)
            
            # 6. Ejecución del algoritmo Dynamic Time Warping (DTW)
            distancia, _ = fastdtw(mfcc_estudiante.T, mfcc_nativo.T, dist=euclidean)
            
            # Normalizar la distancia matemática en un score de 0% a 100%
            score = max(0, min(100, 100 - (distancia / 12)))

        except Exception as audio_err:
            return jsonify({"success": False, "error": f"Error al procesar el audio con Librosa: {str(audio_err)}"}), 500
        finally:
            # Limpieza estricta de archivos temporales
            if os.path.exists(path_estudiante): os.remove(path_estudiante)
            if os.path.exists(path_nativo): os.remove(path_nativo)

        return jsonify({
            "success": True,
            "score": float(score),
            "msg": "Evaluacion de audio web/movil completada exitosamente"
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Error general del servidor: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
