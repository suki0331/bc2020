from flask import Flask, Response, make_response

app = Flask(__name__)

@app.route('/')
def response_test():
    custom_response = Response("[★]Custom Response", 200, 
                                {"Program" : "Flask Web Application"})
    print("[★]")
    return make_response(custom_response)

@app.before_first_request
def before_first_request(): # 얘가 우선적으로 실행이되고 웹서버가 돌아감
    print("[1]앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다.")
    # print("이 서버는 개인 자산이니 건들지 말 것.")
    # print("곧 자료를 전송합니다.")

@app.before_request
def before_request():
    print("[2] 매 HTTP 요청이 처리되기 전에 실행됩니다.")

@app.after_request
def after_request(response):
    print("[3] 매 HTTP 요청이 처리되고 나서 실행됩니다.")
    return response

@app.teardown_request
def teardown_request(exception):
    print("[4] 매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다.")

@app.teardown_appcontext
def teardown_appcontext(exception):
    print("[5] HTTP 요청의 애플리케이션 컨텍스트가 종료될 때 실행된다.")


if __name__ == '__main__':
    app.run(host='127.0.0.30', port=5001)
