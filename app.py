from flask_sqlalchemy import SQLAlchemy

from flask import Flask, render_template, request, redirect, url_for, send_file
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqldb://root:Ssd11012004+-=@localhost:3306/birds_audio'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class AudioFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    audio_data = db.Column(db.LargeBinary(length = (2**32)-1))
    
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio_file' not in request.files:
        return redirect(request.url)
    audio = request.files['audio_file']
    if audio.filename == '':
        return redirect(request.url)
    audio_data = audio.read()
    new_audio = AudioFile(audio_data=audio_data)
    db.session.add(new_audio)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/play_audio/<int:audio_id>')
def play_audio(audio_id):
    audio_file = AudioFile.query.get(audio_id)
    if audio_file:
        return send_file(audio_file.audio_data, mimetype='audio/wav')
    else:
        return 'Audio file not found', 404

if __name__ == '__main__':
    app.run(debug=True)
