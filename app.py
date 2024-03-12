from flask_sqlalchemy import SQLAlchemy
import os
from model_call import *
from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqldb://root:Ssd11012004+-=@localhost:3306/birds_audio'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class AudioFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    audio_data = db.Column(db.LargeBinary(length = (2**32)-1))
    
#with app.app_context():
    #db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/references.html')
def references():
    return render_template('references.html')

@app.route('/audio.html', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        if 'audioInput' not in request.files:
            return redirect(request.url)
        audio = request.files['audioInput']
        audio_data = audio.read()
        if audio.filename == '':
            return redirect(request.url)
        base_directory = os.path.abspath(os.path.dirname(__file__))
        uploads_dir = os.path.join(base_directory, 'uploads')
        
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
            
        file_path = os.path.join(uploads_dir, audio.filename)
        with open(file_path,'wb') as f:
            f.write(audio_data)
        data = get_predictions(file_path)
        new_audio = AudioFile(audio_data=audio_data)
        db.session.add(new_audio)
        db.session.commit()
        return render_template('audio.html', data = data)
    return render_template('audio.html')

""" @app.route('/play/<filename>')
def play_audio(filename):
    base_directory = os.path.abspath(os.path.dirname(__file__))
    uploads_dir = os.path.join(base_directory, 'uploads')
    file_path = os.path.join(uploads_dir, filename)

    return send_file(filename = file_path, mimetype='audio/wav') """

if __name__ == '__main__':
    app.run(debug=True)
