import os
from PIL import Image
from scipy.io import wavfile
from core.util import Logger
import numpy as np
import python_speech_features
import csv
import time
import json
import torch
import random
from scipy.io import wavfile

from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def modify_file_path(path):
    # Split the path into directory and file name
    dir_name, file_name = os.path.split(path)
    # Split the file name into the base and extension
    base_name, ext = os.path.splitext(file_name)

    try:
        # Convert the base name to a float, round it to two decimals, and convert it back to a string
        rounded_base_name = f"{float(base_name):.2f}"

        # Construct the new file name
        new_file_name = rounded_base_name + ext

        # Construct the new full path
        new_path = os.path.join(dir_name, new_file_name)

        return new_path
    except ValueError:
        # If the base name is not a number, return the original path
        return path

def get_valid_path(path: str) -> str:
    """
    Returns the original path if it exists, otherwise tries replacing the last colon with an underscore.
    """
    if os.path.exists(path):
        return path

    # Replace only the colon before the instance number (not timestamps)
    alt_path = path.replace(":", "_", 1)  
    if os.path.exists(alt_path):
        return alt_path

    # If neither exists, return the original (or raise an error if you'd prefer)
    return path

def preprocessRGBData(rgb_data):
    rgb_data = rgb_data.astype('float32')
    rgb_data = rgb_data/255.0
    rgb_data = rgb_data - np.asarray((0.485, 0.456, 0.406))

    return rgb_data


def _pil_loader(path, target_size):
    if "WASD" in path:
        path = modify_file_path(path)
    path = get_valid_path(path) #check if : filepath exist, else convert path to _

    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.resize(target_size)
            return img.convert('RGB')
    except OSError as e:
        print(path)
        print("***********************************************")
        return Image.new('RGB', target_size)


def set_up_log_and_ws_out(models_out, opt_config, experiment_name, headers=None):
    target_logs = os.path.join(models_out, experiment_name + '_logs.csv')
    target_models = os.path.join(models_out, experiment_name)
    print('target_models', target_models)
    if not os.path.isdir(target_models):
        os.makedirs(target_models)
    log = Logger(target_logs, ';')

    if headers is None:
        log.writeHeaders(['epoch', 'train_loss', 'train_auc', 'train_map',
                          'val_loss', 'val_auc', 'val_map'])
    else:
        log.writeHeaders(headers)

    # Dump cfg to json
    dump_cfg = opt_config.copy()
    for key, value in dump_cfg.items():
        if callable(value):
            try:
                dump_cfg[key]=value.__name__
            except:
                dump_cfg[key]='CrossEntropyLoss'
    json_cfg = os.path.join(models_out, experiment_name+'_cfg.json')
    with open(json_cfg, 'w') as json_file:
      json.dump(dump_cfg, json_file)

    models_out = os.path.join(models_out, experiment_name)
    return log, models_out


def csv_to_list(csv_path):
    as_list = None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        as_list = list(reader)
        #print(as_list)
    return as_list


def _generate_mel_spectrogram(audio_clip, sample_rate):
    mfcc = zip(*python_speech_features.mfcc(audio_clip, sample_rate))
    audio_features = np.stack([np.array(i) for i in mfcc])
    audio_features = np.expand_dims(audio_features, axis=0)
    return audio_features


def _fit_audio_clip(audio_clip, sample_rate, video_clip_lenght):
    target_audio_length = (1.0/27.0)*sample_rate*video_clip_lenght
    pad_required = int((target_audio_length-len(audio_clip))/2)
    if pad_required > 0:
        audio_clip = np.pad(audio_clip, pad_width=(pad_required, pad_required),
                            mode='reflect')
    if pad_required < 0:
        audio_clip = audio_clip[-1*pad_required:pad_required]

    return audio_clip




def overlap(audio, noiseAudio):
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = np.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * np.log10(np.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * np.log10(np.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = np.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio
    return audio.astype(np.int16)

def round_filename(path):
    # Split the path into directory and file
    dir_name, file_name = os.path.split(path)

    # Split the file name into the base (number) and extension (.jpg)
    base, ext = os.path.splitext(file_name)

    # Round the base to two decimal places and convert back to string
    rounded_base = f"{float(base):.2f}"

    # Construct the new file name
    new_file_name = rounded_base + ext

    # Construct the new full path
    new_path = os.path.join(dir_name, new_file_name)

    return new_path



def load_av_clip_from_metadata(clip_meta_data, frames_source, audio_source,
                               audio_offset, target_size):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]
    #print(ts_sequence)

    min_ts = float(clip_meta_data[0][1])
    max_ts = float(clip_meta_data[-1][1])
    entity_id = clip_meta_data[0][0]

    selected_frames = [os.path.join(frames_source, entity_id, ts+'.jpg') for ts in ts_sequence]
    #print(selected_frames)
    #selected_frames = [round_filename(path) for path in selected_frames]
    video_data = [_pil_loader(sf, target_size) for sf in selected_frames]
    audio_file = os.path.join(audio_source, entity_id+'.wav')
    audio_file = get_valid_path(audio_file) #check if : filepath exist, else convert path to _

    try:
        sample_rate, audio_data = wavfile.read(audio_file)
        print(audio_file)
    except:
        sample_rate, audio_data = 16000,  np.zeros((16000*10))
        print(audio_file)
        print("no audio no audio no audio no audio no audio no audio no audio")

    offset_dict = {}

    if entity_id.split(":")[0] in offset_dict:
        if "220926" in entity_id:
            print("offset", offset_dict[entity_id.split(":")[0]]/30)
            videoOffset = 0 # -offset_dict[entity_id.split(":")[0]]/30
        else:
            print("offset", offset_dict[entity_id.split(":")[0]]/25)
            videoOffset = 0 # -offset_dict[entity_id.split(":")[0]]/25
    else:
        videoOffset = 0
    audio_start = int((min_ts-audio_offset-videoOffset)*sample_rate)
    print("start", audio_start/sample_rate)
    audio_end = int((max_ts-audio_offset-videoOffset)*sample_rate)
    print("end", audio_end/sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    if len(audio_clip) < sample_rate*(2/25):
        audio_clip = np.zeros((int(sample_rate*(len(clip_meta_data)/25))))

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(selected_frames))

    signal = torch.from_numpy(np.float32(audio_clip))
    print("sigshape",signal.shape)
    speakerEmb = classifier.encode_batch(signal)
    speakerEmb_np = speakerEmb[0][0].squeeze().cpu().numpy().astype(np.float32)
    print("sembshape",type(speakerEmb_np))

    #wavfile.write('/home2/bstephenson/overlap1.wav', 16000, audio_clip)
    #if "train" in audio_file:
    #    print("train, train********************************************")
        #audio_clip = overlap(audio_clip, noise_audio_data)
    audio_features = _generate_mel_spectrogram(audio_clip, sample_rate)

    return video_data, audio_features, speakerEmb_np
