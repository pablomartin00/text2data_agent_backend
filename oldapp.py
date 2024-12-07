from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    # Puedes acceder a los datos enviados al webhook aquí
    data = request.json
    print("Received data:", data)  # Imprime los datos en la consola (útil para depuración)

    # Responde con un mensaje simple
    return jsonify({"message": "Hello, World!"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5174)  # Escucha en el puerto 5174
