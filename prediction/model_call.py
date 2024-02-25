import torch 
from torch import nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
import librosa
import timm
import requests

class BirdClefModel(nn.Module):
    def __init__(self , output_size = 264):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True)
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.classifier = nn.Sequential(nn.Dropout(p = 0.2 , inplace = True)
            ,nn.Linear(in_features = 1280 , out_features = output_size , bias = True))
        
        
    def forward(self , x):
        return self.model(x)

    
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
    audio = torch.from_numpy(audio)
    audio  = torch.stack([audio , audio , audio])
    audio = audio.unsqueeze(dim = 0)
    return audio


def get_predictions(inp_path ): 
    state_dict_file_url = "https://github.com/u-r-a-bot/Bird_clef_audio_proc/raw/main/prediction/model_effnet_b0_state_dict.pth"
    model_path_state_dict = Path("model_effnet_b0_state_dict.pth")
    print("Model Downloading")
    request = requests.get(state_dict_file_url)
    with open(model_path_state_dict, "wb") as f:
        f.write(request.content)
    
    audio = preprocess_audio(audio_path=inp_path)
    try:
        loaded_model = torch.jit.load(path_to_script)
    except:
        print("Loading through script failed trying to use model load")
        try:
            loaded_model = torch.load("model_effnet_b0_model.pth")
            print("Load Successful")
        except:
            print("Loading directly Failed. Loading model using state dict")
            try:
                loaded_model = BirdClefModel()
                loaded_model.load_state_dict(torch.load(model_path_state_dict))
                print("Load Successful")
            except Exception as e:
                print(e)
                print("Model Loading Failed  :( Try again Later")
                pass
    loaded_model.eval()
    with torch.inference_mode():
        y_logits = loaded_model(audio)
        y_preds = torch.softmax(y_logits, dim = 1)
        y_label = torch.argmax(y_preds , dim = 1)
    y_label = y_label.numpy()[0]
    return y_label



