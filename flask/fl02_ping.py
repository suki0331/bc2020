from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello33():
    return"<h1>네이버 ㅗ 안녕 세계 world</h1>""ㅗh""ㅋ"

    
@app.route('/ping', methods=['GET'])
def ping():
    return"<h1>pong</h1>"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
