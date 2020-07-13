from flask import Flask

app = Flask(__name__)
@app.route('/') # 주소
def hello333():
    return"<h1>hello sangwoo world</h1>" "<h1>im am 2years old</h1>"


@app.route('/bit')  # 주소
def hello3():
    return"<h1>hello bit</h1>"

@app.route('/bit/bitcamp')  # 주소
def hello23():
    return"<h1>hello </h1>"

if __name__ == '__main__':
    app.run(host='127.0.0.74', port=8888, debug=True)