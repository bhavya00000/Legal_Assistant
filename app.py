import os
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_babel import Babel, get_locale, gettext
from flask_session import Session
from chat import get_response

app = Flask(__name__)
babel = Babel(app)

app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = os.environ.get('FLASK_SECRET_KEY')


app.config['BABEL_DEBUG'] = True
Session(app)

app.config['LANGUAGES'] = {
    'en': 'English',
    'hi': 'Hindi',
}

@app.route('/setlang/<lang>')
def set_language(lang):
    session['lang'] = lang
    print(f"Session lang set to: {session['lang']}")
    return redirect(request.referrer)

def custom_locale_selector():
    if 'lang' in session:
        return session['lang']
    return request.accept_languages.best_match(app.config['LANGUAGES'].keys())

babel.init_app(app, locale_selector=custom_locale_selector)


@app.route('/')
def home():
    current_lang = session.get('lang', 'Default')
    return render_template('file1.html', current_lang=current_lang)
@app.route('/kyr')
def ask_query():
    current_lang = session.get('lang', 'Default')
    return render_template('kyr.html', current_lang=current_lang)
@app.route('/labour')
def labour():
    current_lang = session.get('lang', 'Default')
    return render_template('labour.html', current_lang=current_lang)
@app.route('/login')
def login():
    current_lang = session.get('lang', 'Default')
    return render_template('login.html', current_lang=current_lang)


@app.get("/chat")
def index():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True)
