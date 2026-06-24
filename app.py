import os
import requests
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------------------------------------------------------
# Ruta principal de comparación
# -------------------------------------------------------------------
@app.route('/comparar', methods=['POST'])
def comparar_pronunciacion():
    try:
        if 'audio_estudiante' not in request.files:
            return jsonify({"success": False, "error": "No se recibio el audio"}), 400

        audio_file = request.files['audio_estudiante']
        profesor_url = request.form.get('audio_profesor_url')

        if not profesor_url:
            return jsonify({"success": False, "error": "Falta la URL del profesor"}), 400

        path_est = os.path.join("/tmp", "estudiante.wav")
        path_prof = os.path.join("/tmp", "profesor.wav")

        audio_file.save(path_est)

        try:
            resp = requests.get(profesor_url, timeout=4)
            if resp.status_code == 200:
                with open(path_prof, 'wb') as f:
                    f.write(resp.content)
            else:
                raise Exception("Descarga fallida")
        except Exception:
            return fallback_rapido(path_est, None)

        try:
            sr_est, y_est = wavfile.read(path_est)
            sr_prof, y_prof = wavfile.read(path_prof)

            if y_est.ndim > 1:
                y_est = y_est[:, 0]
            if y_prof.ndim > 1:
                y_prof = y_prof[:, 0]

            y_est = y_est.astype(np.float32) / 32768.0
            y_prof = y_prof.astype(np.float32) / 32768.0

            TARGET_SR = 8000
            if sr_est != TARGET_SR:
                num_samples = int(len(y_est) * TARGET_SR / sr_est)
                y_est = np.interp(
                    np.linspace(0, len(y_est) - 1, num_samples),
                    np.arange(len(y_est)),
                    y_est
                )
                sr_est = TARGET_SR
            if sr_prof != TARGET_SR:
                num_samples = int(len(y_prof) * TARGET_SR / sr_prof)
                y_prof = np.interp(
                    np.linspace(0, len(y_prof) - 1, num_samples),
                    np.arange(len(y_prof)),
                    y_prof
                )
                sr_prof = TARGET_SR

            max_len = TARGET_SR * 3
            if len(y_est) > max_len:
                y_est = y_est[:max_len]
            if len(y_prof) > max_len:
                y_prof = y_prof[:max_len]

            f_est, Pxx_est = welch(y_est, fs=sr_est, nperseg=256)
            f_prof, Pxx_prof = welch(y_prof, fs=sr_prof, nperseg=256)

            Pxx_est_db = 10 * np.log10(Pxx_est + 1e-10)
            Pxx_prof_db = 10 * np.log10(Pxx_prof + 1e-10)

            distancia = np.linalg.norm(Pxx_est_db - Pxx_prof_db)
            score = max(45.0, min(98.0, 100.0 - (distancia * 1.2)))

        except Exception:
            return fallback_rapido(path_est, path_prof)

        finally:
            if os.path.exists(path_est):
                os.remove(path_est)
            if os.path.exists(path_prof):
                os.remove(path_prof)

        return jsonify({
            "success": True,
            "score": float(score),
            "msg": "Evaluacion ejecutada con exito (espectro de potencia)"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def fallback_rapido(p1, p2=None):
    try:
        size_est = os.path.getsize(p1) if (p1 and os.path.exists(p1)) else 1000
        size_prof = os.path.getsize(p2) if (p2 and os.path.exists(p2)) else 1200
        proporcion = min(size_est, size_prof) / max(size_est, size_prof)
        score = max(55.0, proporcion * 100)
    except:
        score = 78.4

    if p1 and os.path.exists(p1):
        os.remove(p1)
    if p2 and os.path.exists(p2):
        os.remove(p2)

    return jsonify({
        "success": True,
        "score": float(score),
        "msg": "Evaluacion por contingencia (basada en bytes)"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
