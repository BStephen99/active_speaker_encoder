import os
import torch
import pandas as pd


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
    #files = os.listdir('.../AVA/csv/val')
    #files = os.listdir('/home2/bstephenson/GraVi-T/data/annotations/ava_activespeaker_test_v1.0')
    #files = os.listdir('/home2/bstephenson/GraVi-T/data/annotations/ava_activespeaker_train_v1.0')
    #videos = pd.read_csv("/home2/bstephenson/ASDNet/ava_activespeaker_train_augmented.csv")["video_id"].unique()
    #videos = [f[:-18] for f in files]
    #videos = pd.read_csv('/home2/bstephenson/ASDNet/avaStyleCSV.csv')["video_id"].tolist()
    #videos = pd.read_csv('/home2/bstephenson/ASDNet/ava220926.csv')["video_id"].unique().tolist()
    #videos = pd.read_csv('/home2/bstephenson/ASDNet/oursHighTest.csv')["video_id"].unique().tolist()
    #videos = pd.read_csv('/home2/bstephenson/ASDNet/oursHighTest.csv')["video_id"].unique().tolist()
    videos = pd.read_csv('/media/brooke/PPM_Brooke/Mine/ASDNet/ASDNet/ava_activespeaker_val_augmented.csv')["video_id"].unique().tolist()
    #videos = pd.read_csv('/home2/bstephenson/WASD/WASD/csv/train_orig.csv')["video_id"].unique().tolist()
    #videos = pd.read_csv('/home2/bstephenson/WASD/WASD/csv/val_orig.csv')["video_id"].unique().tolist()
    #videos = [clean_filename(v) for v in videos]
    #videos = pd.read_csv('/home2/bstephenson/ASDNet/ava220926downsample.csv')["video_id"].unique().tolist()
    videos.sort()
    return videos
