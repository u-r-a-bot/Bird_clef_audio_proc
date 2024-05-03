import torchaudio
import torch 
from torch import nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
import librosa
import timm 
import torchvision
from torchvision import transforms
import requests
import gdown

specie_to_name = {'abethr1': 'African Bare-eyed Thrush',
 'abhori1': 'African Black-headed Oriole',
 'abythr1': 'Abyssinian Thrush',
 'afbfly1': 'African Blue Flycatcher',
 'afdfly1': 'African Dusky Flycatcher',
 'afecuc1': 'African Emerald Cuckoo',
 'affeag1': 'African Fish-Eagle',
 'afgfly1': 'African Gray Flycatcher',
 'afghor1': 'African Gray Hornbill',
 'afmdov1': 'Mourning Collared-Dove',
 'afpfly1': 'African Paradise-Flycatcher',
 'afpkin1': 'African Pygmy Kingfisher',
 'afpwag1': 'African Pied Wagtail',
 'afrgos1': 'African Goshawk',
 'afrgrp1': 'African Green-Pigeon',
 'afrjac1': 'African Jacana',
 'afrthr1': 'African Thrush',
 'amesun2': 'Amethyst Sunbird',
 'augbuz1': 'Augur Buzzard',
 'bagwea1': 'Baglafecht Weaver',
 'barswa': 'Barn Swallow',
 'bawhor2': 'Black-and-white-casqued Hornbill',
 'bawman1': 'Black-and-white Mannikin',
 'bcbeat1': 'Blue-cheeked Bee-eater',
 'beasun2': 'Beautiful Sunbird',
 'bkctch1': 'Black-crowned Tchagra',
 'bkfruw1': 'Black-faced Rufous-Warbler',
 'blacra1': 'Black Crake',
 'blacuc1': 'Black Cuckoo',
 'blakit1': 'Black Kite',
 'blaplo1': 'Blacksmith Lapwing',
 'blbpuf2': 'Black-backed Puffback',
 'blcapa2': 'Black-collared Apalis',
 'blfbus1': 'Black-fronted Bushshrike',
 'blhgon1': 'Black-headed Gonolek',
 'blhher1': 'Black-headed Heron',
 'blksaw1': 'Black Sawwing',
 'blnmou1': 'Blue-naped Mousebird',
 'blnwea1': 'Black-necked Weaver',
 'bltapa1': 'Black-throated Apalis',
 'bltbar1': 'Black-throated Barbet',
 'bltori1': 'Black-tailed Oriole',
 'blwlap1': 'Black-winged Lapwing',
 'brcale1': 'Brown-chested Alethe',
 'brcsta1': 'Bristle-crowned Starling',
 'brctch1': 'Brown-crowned Tchagra',
 'brcwea1': 'Brown-capped Weaver',
 'brican1': 'Brimstone Canary',
 'brobab1': 'Brown Babbler',
 'broman1': 'Bronze Mannikin',
 'brosun1': 'Bronze Sunbird',
 'brrwhe3': 'Kikuyu White-eye',
 'brtcha1': 'Brown-tailed Chat',
 'brubru1': 'Brubru',
 'brwwar1': 'Brown Woodland-Warbler',
 'bswdov1': 'Blue-spotted Wood-Dove',
 'btweye2': 'Brown-throated Wattle-eye',
 'bubwar2': 'Buff-bellied Warbler',
 'butapa1': 'Buff-throated Apalis',
 'cabgre1': "Cabanis's Greenbul",
 'carcha1': 'Cape Robin-Chat',
 'carwoo1': 'Cardinal Woodpecker',
 'categr': 'Cattle Egret',
 'ccbeat1': 'Cinnamon-chested Bee-eater',
 'chespa1': 'Chestnut Sparrow',
 'chewea1': 'Chestnut Weaver',
 'chibat1': 'Chinspot Batis',
 'chtapa3': 'Chestnut-throated Apalis',
 'chucis1': "Chubb's Cisticola",
 'cibwar1': 'Cinnamon Bracken-Warbler',
 'cohmar1': 'Common House-Martin',
 'colsun2': 'Collared Sunbird',
 'combul2': 'Common Bulbul',
 'combuz1': 'Common Buzzard',
 'comsan': 'Common Sandpiper',
 'crefra2': 'Crested Francolin',
 'crheag1': 'Crowned Eagle',
 'crohor1': 'Crowned Hornbill',
 'darbar1': "D'Arnaud's Barbet",
 'darter3': 'African Darter',
 'didcuc1': 'Dideric Cuckoo',
 'dotbar1': 'Double-toothed Barbet',
 'dutdov1': 'Dusky Turtle-Dove',
 'easmog1': 'Eastern Mountain Greenbul',
 'eaywag1': 'Western Yellow Wagtail',
 'edcsun3': 'Eastern Double-collared Sunbird',
 'egygoo': 'Egyptian Goose',
 'equaka1': 'Equatorial Akalat',
 'eswdov1': 'Emerald-spotted Wood-Dove',
 'eubeat1': 'European Bee-eater',
 'fatrav1': 'Fan-tailed Raven',
 'fatwid1': 'Fan-tailed Widowbird',
 'fislov1': "Fischer's Lovebird",
 'fotdro5': 'Fork-tailed Drongo',
 'gabgos2': 'Gabar Goshawk',
 'gargan': 'Garganey',
 'gbesta1': 'Greater Blue-eared Starling',
 'gnbcam2': 'Gray-backed Camaroptera',
 'gnhsun1': 'Green-headed Sunbird',
 'gobbun1': 'Golden-breasted Bunting',
 'gobsta5': 'Golden-breasted Starling',
 'gobwea1': 'Golden-backed Weaver',
 'golher1': 'Goliath Heron',
 'grbcam1': 'Green-backed Camaroptera',
 'grccra1': 'Gray Crowned-Crane',
 'grecor': 'Great Cormorant',
 'greegr': 'Great Egret',
 'grewoo2': 'Green Woodhoopoe',
 'grwpyt1': 'Green-winged Pytilia',
 'gryapa1': 'Gray Apalis',
 'grywrw1': 'Gray Wren-Warbler',
 'gybfis1': 'Gray-backed Fiscal',
 'gycwar3': 'Gray-capped Warbler',
 'gyhbus1': 'Gray-headed Bushshrike',
 'gyhkin1': 'Gray-headed Kingfisher',
 'gyhneg1': 'Gray-headed Nigrita',
 'gyhspa1': 'Northern Gray-headed Sparrow',
 'gytbar1': 'Gray-throated Barbet',
 'hadibi1': 'Hadada Ibis',
 'hamerk1': 'Hamerkop',
 'hartur1': "Hartlaub's Turaco",
 'helgui': 'Helmeted Guineafowl',
 'hipbab1': "Hinde's Pied-Babbler",
 'hoopoe': 'Eurasian Hoopoe',
 'huncis1': "Hunter's Cisticola",
 'hunsun2': "Hunter's Sunbird",
 'joygre1': 'Joyful Greenbul',
 'kerspa2': 'Kenya Rufous Sparrow',
 'klacuc1': "Klaas's Cuckoo",
 'kvbsun1': 'Eastern Violet-backed Sunbird',
 'laudov1': 'Laughing Dove',
 'lawgol': "Lawrence's Goldfinch",
 'lesmaw1': 'Lesser Masked-Weaver',
 'lessts1': 'Lesser Striped Swallow',
 'libeat1': 'Little Bee-eater',
 'litegr': 'Little Egret',
 'litswi1': 'Little Swift',
 'litwea1': 'Little Weaver',
 'loceag1': 'Long-crested Eagle',
 'lotcor1': 'Long-tailed Cormorant',
 'lotlap1': 'Long-toed Lapwing',
 'luebus1': "Lühder's Bushshrike",
 'mabeat1': 'Madagascar Bee-eater',
 'macshr1': "Mackinnon's Shrike",
 'malkin1': 'Malachite Kingfisher',
 'marsto1': 'Marabou Stork',
 'marsun2': 'Mariqua Sunbird',
 'mcptit1': 'Mouse-colored Penduline-Tit',
 'meypar1': "Meyer's Parrot",
 'moccha1': 'Mocking Cliff-Chat',
 'mouwag1': 'Mountain Wagtail',
 'ndcsun2': 'Northern Double-collared Sunbird',
 'nobfly1': 'Northern Black-Flycatcher',
 'norbro1': 'Northern Brownbul',
 'norcro1': 'Northern Crombec',
 'norfis1': 'Northern Fiscal',
 'norpuf1': 'Northern Puffback',
 'nubwoo1': 'Nubian Woodpecker',
 'pabspa1': 'Parrot-billed Sparrow',
 'palfly2': 'Pale Flycatcher',
 'palpri1': 'Pale Prinia',
 'piecro1': 'Pied Crow',
 'piekin1': 'Pied Kingfisher',
 'pitwhy': 'Pin-tailed Whydah',
 'purgre2': 'Purple Grenadier',
 'pygbat1': 'Pygmy Batis',
 'quailf1': 'Quailfinch',
 'ratcis1': 'Rattling Cisticola',
 'raybar1': 'Red-and-yellow Barbet',
 'rbsrob1': 'Red-backed Scrub-Robin',
 'rebfir2': 'Red-billed Firefinch',
 'rebhor1': 'Northern Red-billed Hornbill',
 'reboxp1': 'Red-billed Oxpecker',
 'reccor': 'Red-cheeked Cordonbleu',
 'reccuc1': 'Red-chested Cuckoo',
 'reedov1': 'Red-eyed Dove',
 'refbar2': 'Red-fronted Barbet',
 'refcro1': 'Red-faced Crombec',
 'reftin1': 'Red-fronted Tinkerbird',
 'refwar2': 'Red-fronted Prinia',
 'rehblu1': 'Red-headed Bluebill',
 'rehwea1': 'Red-headed Weaver',
 'reisee2': "Reichenow's Seedeater",
 'rerswa1': 'Red-rumped Swallow',
 'rewsta1': 'Red-winged Starling',
 'rindov': 'Ring-necked Dove',
 'rocmar2': 'Rock Martin',
 'rostur1': "Ross's Turaco",
 'ruegls1': "Rüppell's Starling",
 'rufcha2': 'Rufous Chatterer',
 'sacibi2': 'African Sacred Ibis',
 'sccsun2': 'Scarlet-chested Sunbird',
 'scrcha1': 'Snowy-crowned Robin-Chat',
 'scthon1': 'Scaly-throated Honeyguide',
 'shesta1': "Shelley's Starling",
 'sichor1': 'Silvery-cheeked Hornbill',
 'sincis1': 'Singing Cisticola',
 'slbgre1': 'Slender-billed Greenbul',
 'slcbou1': 'Slate-colored Boubou',
 'sltnig1': 'Slender-tailed Nightjar',
 'sobfly1': 'Southern Black-Flycatcher',
 'somgre1': 'Sombre Greenbul',
 'somtit4': 'Somali Tit',
 'soucit1': 'Southern Citril',
 'soufis1': 'Southern Fiscal',
 'spemou2': 'Speckled Mousebird',
 'spepig1': 'Speckled Pigeon',
 'spewea1': 'Spectacled Weaver',
 'spfbar1': 'Spot-flanked Barbet',
 'spfwea1': 'Speckle-fronted Weaver',
 'spmthr1': 'Spotted Morning-Thrush',
 'spwlap1': 'Spur-winged Lapwing',
 'squher1': 'Squacco Heron',
 'strher': 'Striated Heron',
 'strsee1': 'Streaky Seedeater',
 'stusta1': "Stuhlmann's Starling",
 'subbus1': 'Sulphur-breasted Bushshrike',
 'supsta1': 'Superb Starling',
 'tacsun1': 'Tacazze Sunbird',
 'tafpri1': 'Tawny-flanked Prinia',
 'tamdov1': 'Tambourine Dove',
 'thrnig1': 'Thrush Nightingale',
 'trobou1': 'Tropical Boubou',
 'varsun2': 'Variable Sunbird',
 'vibsta2': 'Violet-backed Starling',
 'vilwea1': 'Village Weaver',
 'vimwea1': 'Vitelline Masked-Weaver',
 'walsta1': "Waller's Starling",
 'wbgbir1': 'White-bellied Go-away-bird',
 'wbrcha2': 'White-browed Robin-Chat',
 'wbswea1': 'White-browed Sparrow-Weaver',
 'wfbeat1': 'White-fronted Bee-eater',
 'whbcan1': 'White-bellied Canary',
 'whbcou1': 'White-browed Coucal',
 'whbcro2': 'White-browed Crombec',
 'whbtit5': 'White-bellied Tit',
 'whbwea1': 'White-headed Buffalo-Weaver',
 'whbwhe3': 'Pale White-eye',
 'whcpri2': 'White-chinned Prinia',
 'whctur2': 'White-crested Turaco',
 'wheslf1': 'White-eyed Slaty-Flycatcher',
 'whhsaw1': 'White-headed Sawwing',
 'whihel1': 'White Helmetshrike',
 'whrshr1': 'White-rumped Shrike',
 'witswa1': 'Wire-tailed Swallow',
 'wlwwar': 'Willow Warbler',
 'wookin1': 'Woodland Kingfisher',
 'woosan': 'Wood Sandpiper',
 'wtbeat1': 'White-throated Bee-eater',
 'yebapa1': 'Yellow-breasted Apalis',
 'yebbar1': 'Yellow-billed Barbet',
 'yebduc1': 'Yellow-billed Duck',
 'yebere1': 'Yellow-bellied Eremomela',
 'yebgre1': 'Yellow-bellied Greenbul',
 'yebsto1': 'Yellow-billed Stork',
 'yeccan1': 'Yellow-crowned Canary',
 'yefcan': 'Yellow-fronted Canary',
 'yelbis1': 'Yellow Bishop',
 'yenspu1': 'Yellow-necked Francolin',
 'yertin1': 'Yellow-rumped Tinkerbird',
 'yesbar1': 'Yellow-spotted Barbet',
 'yespet1': 'Yellow-spotted Bush Sparrow',
 'yetgre1': 'Yellow-throated Greenbul',
 'yewgre1': 'Yellow-whiskered Greenbul'}

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


def preprocess(filepath, sampling_rate=32000, num_mels=224, min_freq=128, max_freq=16000,
               n_fft=1024, hop_length=512, num_samples=5*32_000, device='cpu'):

    audio_tensor, sr = torchaudio.load(filepath)
    

    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, sampling_rate).to(device)
        audio_tensor = resampler(audio_tensor)

    if num_samples is not None:
        if audio_tensor.shape[1] > num_samples:
            audio_tensor = audio_tensor[:, :num_samples]
        elif audio_tensor.shape[1] < num_samples:
            num_missing_samples = num_samples - audio_tensor.shape[1]
            last_dim_padding = (0, num_missing_samples)
            audio_tensor = torch.nn.functional.pad(audio_tensor, last_dim_padding)
    

    time_shift = torchaudio.transforms.TimeMasking(5)
    audio_tensor = time_shift(audio_tensor)

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=num_mels,
                                                     f_min=min_freq,
                                                     f_max=max_freq).to(device)
    audio_tensor_mel_spec = transform(audio_tensor.to(device))
    
    audio_tensor_mel_spec_db = torchaudio.transforms.AmplitudeToDB()(audio_tensor_mel_spec)
    
    mel_spec_img = torch.stack([audio_tensor_mel_spec_db.squeeze()] * 3)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        normalize
    ])
    preprocessed_img = transform(mel_spec_img)
    
    return preprocessed_img




def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive.

    Args:
        file_id (str): The file ID from Google Drive.
        destination (str): The destination path where the file will be saved.
    """
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)


def get_top_indices(softmax_logits, threshold_margin, threshold_entropy):

    if len(softmax_logits.shape) == 1:
        num_top_probs_needed = min(6, len(softmax_logits))
        top_indices = torch.argsort(softmax_logits, descending=True)[:num_top_probs_needed]
    else:

        top_indices = torch.argsort(softmax_logits, descending=True)
        margin = softmax_logits[top_indices[0]] - softmax_logits[top_indices[-1]]

        entropy = -torch.sum(softmax_logits * torch.log(softmax_logits), dim=1).item()

        if margin > threshold_margin and entropy > threshold_entropy:
            num_top_probs_needed = 6
        elif margin > threshold_margin or entropy > threshold_entropy:
            num_top_probs_needed = 4
        else:
            num_top_probs_needed = 3

        top_indices = top_indices[:num_top_probs_needed]

    return top_indices




def get_predictions(inp_path, threshold_margin=0.1, threshold_entropy=0.1): 
    file_id = "1uuYaUhLnZGdzT5iLaFweACWrY7WNET-3"
    destination = "efficientvit.pth"
    model_path = Path("efficientvit.pth")
    if model_path.exists():
        print("Skipping download")
    else:    
        print("Model Downloading")
        download_file_from_google_drive(file_id, destination)
    
    audio = preprocess(filepath=inp_path).unsqueeze(dim=0)
#     print(f"Shape of audio = {audio.shape}")
    loaded_model = torch.load(model_path)
    print("Model Loaded Successfully")
    loaded_model.eval()
    with torch.inference_mode():
        y_logits = loaded_model(audio)
        y_preds = torch.softmax(y_logits, dim=1).squeeze()
        top_indices = get_top_indices(y_preds, threshold_margin, threshold_entropy)
        top_probs = [y_preds[i].item() for i in top_indices]
        tot_sum = sum(top_probs)
        top_probs = [top_probs[i] / tot_sum for i in range(len(top_probs))]
        top_classes = [target_labels[i] for i in top_indices]
        top_class_names = [specie_to_name[i] for i in top_classes]
        probs_dict = {}
        for name, probs in zip(top_class_names , top_probs):
            probs_dict[name] = probs
    return probs_dict
