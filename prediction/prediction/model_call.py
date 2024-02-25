import torch 
from torch import nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
import librosa
import timm
path_to_model = Path("text")

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


def get_predictions(inp_path , path_to_model = path_to_model): 
    audio = preprocess_audio(audio_path=inp_path)
    try:
        loaded_model = torch.jit.load(path_to_model)
    except:
        print("Loading through script failed trying to use model load")
        try:
            loaded_model = torch.load("model_effnet_b0_model.pth")
        except:
            print("Loading directly Failed. Loading model using state dict")
            try:
                loaded_model = BirdClefModel()
                loaded_model.load_state_dict(torch.load("model_effnet_b0_state_dict.pth"))
            except:
                print("Model Loading Failed  :( Try again Later")
                pass
    loaded_model.eval()
    with torch.inference_mode():
        y_logits = loaded_model(audio)
        y_preds = torch.softmax(y_logits, dim = 1)
        y_label = torch.argmax(y_preds , dim = 1)
    y_label = y_label.numpy()
    return y_label



