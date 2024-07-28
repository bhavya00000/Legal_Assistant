import os
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_babel import Babel, get_locale, gettext
from flask_session import Session
from chat import get_response


import torch
import torch.nn as nn
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

def train_model():
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'final loss: {loss.item():.4f}')

    data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training complete. file saved to {FILE}')

@app.before_first_request
def initialize_model():
    print("Training model...")
    train_model()
    print("Model training completed.")

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
