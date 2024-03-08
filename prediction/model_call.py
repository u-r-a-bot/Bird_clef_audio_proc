import torch 
from torch import nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
import librosa
import timm
import requests
target_labels = [
        'abethr1', 'abhori1', 'abythr1', 'afbfly1', 'afdfly1', 'afecuc1',
        'affeag1', 'afgfly1', 'afghor1', 'afmdov1', 'afpfly1', 'afpkin1',
        'afpwag1', 'afrgos1', 'afrgrp1', 'afrjac1', 'afrthr1', 'amesun2',
        'augbuz1', 'bagwea1', 'barswa', 'bawhor2', 'bawman1', 'bcbeat1',
        'beasun2', 'bkctch1', 'bkfruw1', 'blacra1', 'blacuc1', 'blakit1',
        'blaplo1', 'blbpuf2', 'blcapa2', 'blfbus1', 'blhgon1', 'blhher1',
        'blksaw1', 'blnmou1', 'blnwea1', 'bltapa1', 'bltbar1', 'bltori1',
        'blwlap1', 'brcale1', 'brcsta1', 'brctch1', 'brcwea1', 'brican1',
        'brobab1', 'broman1', 'brosun1', 'brrwhe3', 'brtcha1', 'brubru1',
        'brwwar1', 'bswdov1', 'btweye2', 'bubwar2', 'butapa1', 'cabgre1',
        'carcha1', 'carwoo1', 'categr', 'ccbeat1', 'chespa1', 'chewea1',
        'chibat1', 'chtapa3', 'chucis1', 'cibwar1', 'cohmar1', 'colsun2',
        'combul2', 'combuz1', 'comsan', 'crefra2', 'crheag1', 'crohor1',
        'darbar1', 'darter3', 'didcuc1', 'dotbar1', 'dutdov1', 'easmog1',
        'eaywag1', 'edcsun3', 'egygoo', 'equaka1', 'eswdov1', 'eubeat1',
        'fatrav1', 'fatwid1', 'fislov1', 'fotdro5', 'gabgos2', 'gargan',
        'gbesta1', 'gnbcam2', 'gnhsun1', 'gobbun1', 'gobsta5', 'gobwea1',
        'golher1', 'grbcam1', 'grccra1', 'grecor', 'greegr', 'grewoo2',
        'grwpyt1', 'gryapa1', 'grywrw1', 'gybfis1', 'gycwar3', 'gyhbus1',
        'gyhkin1', 'gyhneg1', 'gyhspa1', 'gytbar1', 'hadibi1', 'hamerk1',
        'hartur1', 'helgui', 'hipbab1', 'hoopoe', 'huncis1', 'hunsun2',
        'joygre1', 'kerspa2', 'klacuc1', 'kvbsun1', 'laudov1', 'lawgol',
        'lesmaw1', 'lessts1', 'libeat1', 'litegr', 'litswi1', 'litwea1',
        'loceag1', 'lotcor1', 'lotlap1', 'luebus1', 'mabeat1', 'macshr1',
        'malkin1', 'marsto1', 'marsun2', 'mcptit1', 'meypar1', 'moccha1',
        'mouwag1', 'ndcsun2', 'nobfly1', 'norbro1', 'norcro1', 'norfis1',
        'norpuf1', 'nubwoo1', 'pabspa1', 'palfly2', 'palpri1', 'piecro1',
        'piekin1', 'pitwhy', 'purgre2', 'pygbat1', 'quailf1', 'ratcis1',
        'raybar1', 'rbsrob1', 'rebfir2', 'rebhor1', 'reboxp1', 'reccor',
        'reccuc1', 'reedov1', 'refbar2', 'refcro1', 'reftin1', 'refwar2',
        'rehblu1', 'rehwea1', 'reisee2', 'rerswa1', 'rewsta1', 'rindov',
        'rocmar2', 'rostur1', 'ruegls1', 'rufcha2', 'sacibi2', 'sccsun2',
        'scrcha1', 'scthon1', 'shesta1', 'sichor1', 'sincis1', 'slbgre1',
        'slcbou1', 'sltnig1', 'sobfly1', 'somgre1', 'somtit4', 'soucit1',
        'soufis1', 'spemou2', 'spepig1', 'spewea1', 'spfbar1', 'spfwea1',
        'spmthr1', 'spwlap1', 'squher1', 'strher', 'strsee1', 'stusta1',
        'subbus1', 'supsta1', 'tacsun1', 'tafpri1', 'tamdov1', 'thrnig1',
        'trobou1', 'varsun2', 'vibsta2', 'vilwea1', 'vimwea1', 'walsta1',
        'wbgbir1', 'wbrcha2', 'wbswea1', 'wfbeat1', 'whbcan1', 'whbcou1',
        'whbcro2', 'whbtit5', 'whbwea1', 'whbwhe3', 'whcpri2', 'whctur2',
        'wheslf1', 'whhsaw1', 'whihel1', 'whrshr1', 'witswa1', 'wlwwar',
        'wookin1', 'woosan', 'wtbeat1', 'yebapa1', 'yebbar1', 'yebduc1',
        'yebere1', 'yebgre1', 'yebsto1', 'yeccan1', 'yefcan', 'yelbis1',
        'yenspu1', 'yertin1', 'yesbar1', 'yespet1', 'yetgre1', 'yewgre1'
        ]

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
            loaded_model = torch.load("model_effnet_b0_model.pth", map_location=torch.device('cpu'))
            print("Load Successful")
        except:
            print("Loading directly Failed. Loading model using state dict")
            try:
                loaded_model = BirdClefModel()
                loaded_model.load_state_dict(torch.load(model_path_state_dict, map_location=torch.device('cpu')))
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
    return target_labels[y_label]



