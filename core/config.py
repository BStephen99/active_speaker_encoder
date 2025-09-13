import sys

import torch.nn as nn
import torch.optim as optim
import core.models as mdet


STE_inputs = {
    # input files
    'ava_csv_train_full': '/path/to/ava/ava_activespeaker_train_augmented.csv',
    'ava_csv_val_full': '/path/to/ava/ava_activespeaker_val_augmented.csv',
    'ava_csv_test_full': '/path/to/ava/ava_activespeaker_test_augmented.csv',

    'csv_train_full': '/path/to/peppermint/annotations.csv',

    'wasd_csv_train_full': '/path/to/wasd/csv/train_orig.csv',
    'wasd_csv_val_full': '/path/to/wasd/csv/val_orig.csv',

    # Data config
    'ava_audio_dir': '/path/to/ava/slice_audio',
    'wasd_audio_dir': '/path/to/wasd/clips_audios/',
    'audio_dir': '/path/to/peppermint/slice_audio_ours',

    'ava_video_dir': '/path/to/ava/crops',
    'wasd_video_dir': '/path/to/wasd/clips_videos/',
    'video_dir': '/path/to/peppermint/cropsOurs',

    'mode': "pepper_back", #pepper_back, pepper_high, ava_train, ava_test or wasd_train, wasd_test  -choose the dataset to process in the forward phase.

    'models_out':'./model'
}

ASC_inputs = {
    # input files
    'features_train_full': '.../train_forward/*.csv',
    'features_val_full': '.../val_forward/*.csv',

    # Data config
    'models_out': '...'
}

ASC_inputs_forward = {
    # input files
    'features_train_full': '...',
    'features_val_full': '.../val_forward'
}

#Optimization params
STE_optimization_params = {
    # Net Arch
    'backbone': mdet.resnet18_two_streams,

    # Optimization config
    'optimizer': optim.Adam,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 3e-4,
    'epochs': 100,
    'step_size': 40,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 64,
    'threads': 0
}

STE_forward_params = {
    # Net Arch
    'backbone': mdet.resnet18_two_streams_forward,

    # Batch Config
    'batch_size': 1,
    'threads': 1
}

ASC_optimization_params = {
    # Optimizer config
    'optimizer': optim.Adam,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 3e-6,
    'epochs': 15,
    'step_size': 10,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 64,
    'threads': 0
}

ASC_forward_params = {
    # Batch Config
    'batch_size': 1,
    'threads': 1
}
