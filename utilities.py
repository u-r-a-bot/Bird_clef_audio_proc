import librosa
import librosa.display
import numpy as np
device = "cpu"
import matplotlib.pyplot as plt
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup



def get_wikipedia_IUCN_image(species_name):
    search_url = f"https://en.wikipedia.org/wiki/{species_name}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table_element = soup.find('table', class_='infobox biota')
    if table_element:

        img_tags = table_element.find_all('img')
        for img_tag in img_tags:

            if "Status_iucn" in img_tag['src']:
                image_url = img_tag['src']
                return "https:"+image_url
    return None




def generate_response(Bird_name:str)-> str:
    GOOGLE_API_KEY=""

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    prompt= f"Tell me about this bird {Bird_name} in about 20 lines of continuous text its body color description and all other stuff its common locations to be found, etc."
    response = model.generate_content(prompt)
    return response.text
# generate_response(Bird_name  = "African Dusky flycatcher")




def specshow_with_separator(file_path, save_path, sampling_rate=32000, n_fft=1024, hop_length=512, num_mels=224,
                            min_freq=128, max_freq=16000, num_samples=None, device=device, separator_time=5,
                           displ_image = False):
    
    audio_path = file_path
    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    
    if num_samples is None:
        num_samples = sampling_rate * 5
    
#     if len(audio) > num_samples:
#         audio = audio[:num_samples]
    elif len(audio) < num_samples:
        audio = librosa.util.pad_center(audio, num_samples)

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                              n_mels=num_mels, fmin=min_freq, fmax=max_freq)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='gray_r')
    plt.axvline(x=separator_time, color='r', linestyle='--')
    plt.axis('off') 
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()