from flask import Flask, request, render_template
from train_vector import summarize

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/getSentence', methods=['GET'])
def getSentence():
    title = request.args.get('title')
    para = request.args.get('para')
    return summarize(para, title)

@app.route('/signin', methods=['POST'])
def signin():
    username = request.form['username']
    password = request.form['password']
    if username=='admin' and password=='password':
        return render_template('signin-ok.html', username=username)
    return render_template('form.html', message='Bad username or password', username=username)

if __name__ == '__main__':
    app.run()
