import os
import torch
import pandas as pd
import core.config as exp_conf


class Logger():
    def __init__(self, targetFile, separator=';'):
        self.targetFile = targetFile
        self.separator = separator

    def writeHeaders(self, headers):
        with open(self.targetFile, 'a') as fh:
            for aHeader in headers:
                fh.write(aHeader + self.separator)
            fh.write('\n')

    def writeDataLog(self, dataArray):
        with open(self.targetFile, 'a') as fh:
            for dataItem in dataArray:
                fh.write(str(dataItem) + self.separator)
            fh.write('\n')

def configure_backbone(backbone, size, pretrained_arg=True, num_classes_arg=2):
    return backbone(pretrained=pretrained_arg, rgb_stack_size=size,
                    num_classes=num_classes_arg)

def configure_backbone_forward_phase(backbone, pretrained_weights_path, size, pretrained_arg=False, num_classes_arg=2):
    return backbone(pretrained_weights_path, rgb_stack_size=size,
                    num_classes=num_classes_arg)

def load_train_video_set():
    #files = os.listdir('.../AVA/csv/train')
    #files = os.listdir('/home2/bstephenson/GraVi-T/data/annotations/ava_activespeaker_train_v1.0')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos

import re

def clean_filename(filename):
    # Use regular expression to remove the pattern of _digits-digits at the end
    cleaned_filename = re.sub(r'_\d+-\d+$', '', filename)
    return cleaned_filename

def load_val_video_set():

    io_config = exp_conf.STE_inputs

    mode = io_config['mode']

    if mode == "ava_train":
        csv_file = io_config['ava_csv_train_full']
    elif mode == "ava_val":
        csv_file = io_config['ava_csv_val_full']
    elif mode == "pepper_back":
        csv_file = io_config['csv_train_full']
    elif mode == "pepper_high":
        csv_file = io_config['csv_train_full']
    elif mode == "wasd_train":
        csv_file = io_config['wasd_csv_train_full']
    elif mode == "wasd_val":
        csv_file = io_config['wasd_csv_val_full']
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Read unique video IDs from the appropriate CSV
    videos = pd.read_csv(csv_file)["video_id"].unique().tolist()


    videos.sort()
    return videos
