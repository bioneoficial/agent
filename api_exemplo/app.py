from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the API'})

if __name__ == '__main__':
    app.run(debug=True)