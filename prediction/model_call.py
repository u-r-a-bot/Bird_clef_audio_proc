import torch 
from torch import nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
import librosa

path_to_model = Path("text")

    
def preprocess_audio(audio_path, sampling_rate = 32000 , num_mels = 128 , 
                 min_freq = 128 , max_freq = 16000 , n_fft = 3200 
                 , hop_length = 80 , num_samples  = None):
    if num_samples == None:
        num_samples  = 5*sampling_rate
    audio , sr = librosa.load(audio_path)
    if sampling_rate != sr:
        audio = librosa.resample(audio , orig_sr = sr , target_sr = sampling_rate)
    if audio.shape[0] < num_samples:
        num_missing_samples = num_samples - audio.shape[0]
        last_dim_padding = (0, num_missing_samples)
        audio = np.pad(audio , last_dim_padding)
    if audio.shape[0] > num_mels:
        audio = audio[:num_samples]
    audio = librosa.feature.melspectrogram(y = audio , sr = sampling_rate,
                                                  n_fft = n_fft , hop_length = hop_length,
                                                  n_mels = num_mels , fmin = min_freq,
                                                  fmax = max_freq)
    audio = librosa.power_to_db(audio).astype(np.float32)
    return audio


def get_predictions(inp_path , path_to_model = path_to_model): 
    loaded_model = torch.load(f = path_to_model)
    loaded_model.eval()
    audio = preprocess_audio(audio_path=inp_path)
    with torch.inference_mode():
        y_logits = loaded_model(audio)
        y_preds = torch.softmax(y_logits, dim = 1)
        y_label = torch.argmax(y_preds , dim = 1)
    y_label = y_label.numpy()
    return y_label



