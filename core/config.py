import sys

import torch.nn as nn
import torch.optim as optim
#import core.modelsOrig as mdet
import core.models as mdet
#import core.modelsCrossModality as mdet

STE_inputs = {
    # input files
    #'csv_val_full': '/home2/bstephenson/ASDNet/ava_activespeaker_train_augmented.csv',
    'csv_val_full': '/media/brooke/PPM_Brooke/Mine/ASDNet/ASDNet/ava_activespeaker_val_augmented.csv',
    #'csv_val_full': '/home2/bstephenson/ASDNet/ava_activespeaker_test_augmented.csv',
    #csv_val_full: '/media/brooke/PPM_Brooke/Mine/ASDNet/ASDNet/ava_activespeaker_val_augmented.csv',
    #csv_val_full: '/media/brooke/PPM_Brooke/Mine/ASDNet/ASDNet/oursBackTest.csv',
    #'csv_val_full': '/home2/bstephenson/ASDNet/oursHighTrain.csv',
    #'csv_val_full': '/home2/bstephenson/ASDNet/oursHighTrain.csv',
    #'csv_val_full': '/home2/bstephenson/ASDNet/oursHighTest.csv',

    #'csv_val_full':'/home2/bstephenson/WASD/WASD/csv/train_orig.csv',
    #'csv_val_full':'/home2/bstephenson/WASD/WASD/csv/val_orig.csv',

    # Data config
    #'audio_dir': '..../instance_wavs_time/',
    'audio_dir': '/media/brooke/PPM_Brooke/Mine/ASDNet/ASDNet/slice_audio',
    #'audio_dir': '/home2/bstephenson/ASDNet/slice_audio_ours/',
    #'audio_dir':'/home2/bstephenson/WASD/WASD/clips_audios/',
    'video_dir': '/media/brooke/PPM_Brooke/Mine/ASDNet/ASDNet/crops',
    #'video_dir': '/home2/bstephenson/ASDNet/cropsOurs/',
    #'video_dir': '/media/brooke/PPM_Brooke/Mine/ASDNet/ASDNet/cropsOurs/back',
    #'video_dir':"/home2/bstephenson/WASD/WASD/clips_videos/",
    'models_out':'/home/brooke/Documents/active-speakers-context/active-speakers-context/model'
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
    #'batch_size': 32,
    'threads': 4
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
