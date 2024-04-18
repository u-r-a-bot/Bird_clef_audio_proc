from flask_sqlalchemy import SQLAlchemy
import os
import pandas as pd
import requests
from werkzeug.utils import secure_filename
from model_call import *
from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, jsonify
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqldb://root:Ssd11012004+-=@localhost:3306/birds_audio'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class AudioFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    audio_data = db.Column(db.LargeBinary(length = (2**32)-1))
    
#with app.app_context():
    #db.create_all()
    
def fetch_bird_image(bird_name, api_key = "ffPnG8Fl26jOZAJnlpO4nwid7eZ_WaPcMY_Wasy_TPU"):
    url = f"https://api.unsplash.com/search/photos/?query={bird_name}&client_id={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if 'results' in data and len(data['results']) > 0:
        image_url = data['results'][0]['urls']['regular']
        return image_url
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/references.html')
def references():
    return render_template('references.html')

@app.route('/audio.html', methods=['GET', 'POST'])
def audio():
    return render_template('audio.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('file')
    print(files)
    if len(files) == 0:
        return jsonify({'error': 'No selected files'}), 400

    results = []
    for file in files:
        print(file.filename)
        audio_data = file.read()
        print(file.filename)
        base_directory = os.path.abspath(os.path.dirname(__file__))
        uploads_dir = os.path.join(base_directory, 'uploads')
        
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
            
        file_path = os.path.join(uploads_dir, file.filename)
        with open(file_path,'wb') as f:
            f.write(audio_data)
        data = get_predictions(file_path)
        
        df = pd.read_csv('train_metadata.csv', usecols=['common_name','latitude','longitude'])
        res = df.loc[df.loc[:,'common_name'] == data]
        res = res.loc[:,["latitude","longitude"]]
        res = res.dropna()
        lat, long = res.latitude, res.longitude
        lat_list = lat.tolist()
        long_list = long.tolist()
        image_url = None
        try:
            image_url = fetch_bird_image(data, )
        
        except Exception:
            print("Image not retrieved")
        
        new_audio = AudioFile(audio_data=audio_data)
        db.session.add(new_audio)
        db.session.commit()
        results.append({'filename': file.filename, 'predictions': data, 'imageURL': image_url, 'latitude': lat_list, 'longitude': long_list})

    return jsonify({'results': results}), 200

""" @app.route('/audio.html', methods=['GET', 'POST'])
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
        image_url = None
        try:
            image_url = fetch_bird_image(data, )
        
        except Exception:
            print("Image not retrieved")
        
        new_audio = AudioFile(audio_data=audio_data)
        db.session.add(new_audio)
        db.session.commit()
        return render_template('audio.html', data = data, image_url = image_url)
        
    return render_template('audio.html') """

""" @app.route('/play/<filename>')
def play_audio(filename):
    base_directory = os.path.abspath(os.path.dirname(__file__))
    uploads_dir = os.path.join(base_directory, 'uploads')
    file_path = os.path.join(uploads_dir, filename)

    return send_file(filename = file_path, mimetype='audio/wav') """

if __name__ == '__main__':
    app.run(debug=True)
