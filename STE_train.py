import os
import sys
import torch

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.dataset import AudioVideoDatasetAuxLosses
from core.optimization import optimize_av_losses
from core.io import set_up_log_and_ws_out
from core.util import configure_backbone

import core.custom_transforms as ct
import core.config as exp_conf


if __name__ == '__main__':
    #experiment Reproducibility
    torch.manual_seed(11)
    torch.cuda.manual_seed(22)
    torch.backends.cudnn.deterministic = True

    clip_lenght = int(sys.argv[1])
    cuda_device_number = str(sys.argv[2])
    image_size = (144, 144) #Dont forget to assign this same size on ./core/custom_transforms

    # check these 3 are in order, everythine else is kind of automated
    #model_name = 'ste_encoder
    #model_name = 'ste_encoder_overlapNoise'
    #model_name = 'ste_encoder_Ours'
    #model_name = 'ste_encoder_AllCombined'
    #model_name = 'ste_encoder_WASD'
    #model_name = 'ste_encoder_AllCombined_FilteredSet'
    model_name = 'ste_encoder_OURS_FilteredSet'
    #model_name = 'ste_encoderCrossModal'
    io_config = exp_conf.STE_inputs
    opt_config = exp_conf.STE_optimization_params
    opt_config['batch_size'] = 128

    # io config
    log, target_models = set_up_log_and_ws_out(io_config['models_out'],
                                               opt_config, model_name)

    # cuda config
    backbone = configure_backbone(opt_config['backbone'], clip_lenght)
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+cuda_device_number if has_cuda else 'cpu')
    backbone = backbone.to(device)

    #Optimization config
    criterion = opt_config['criterion']
    optimizer = opt_config['optimizer'](backbone.parameters(),
                                        lr=opt_config['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt_config['step_size'],
                                    gamma=opt_config['gamma'])

    video_data_transforms = {
        'train': ct.video_train,
        'val': ct.video_val
    }

    #video_train_path = os.path.join(io_config['video_dir'], 'train')
    #video_train_path1 = os.path.join("/home2/bstephenson/ASDNet/cropsOurs/", 'train')
    video_train_path1 = os.path.join("/home2/bstephenson/ASDNet/cropsOurs/", 'back')
    video_train_path4 = os.path.join("/home2/bstephenson/ASDNet/cropsOurs/", 'high')
    video_train_path2 = os.path.join("/home2/bstephenson/ASDNet/crops/", 'train')
    video_train_path3 = os.path.join("/home2/bstephenson/WASD/WASD/clips_videos/", 'train')

    #audio_train_path = os.path.join(io_config['audio_dir'], 'train')
    audio_train_path1 = os.path.join("/home2/bstephenson/ASDNet/slice_audio_ours/", 'back')
    audio_train_path4 = os.path.join("/home2/bstephenson/ASDNet/slice_audio_ours/", 'high')
    audio_train_path2 = os.path.join("/home2/bstephenson/ASDNet/slice_audio/", 'train')
    audio_train_path3 = os.path.join("/home2/bstephenson/WASD/WASD/clips_audios/", 'train')


    #video_val_path = os.path.join(io_config['video_dir'], 'val')
    #video_val_path = os.path.join("/home2/bstephenson/ASDNet/crops/", 'val')
    video_val_path = os.path.join("/home2/bstephenson/WASD/WASD/clips_videos/", 'val')
    #audio_val_path = os.path.join(io_config['audio_dir'], 'val')
    #audio_val_path = os.path.join('/home2/bstephenson/ASDNet/slice_audio/', 'val')
    audio_val_path = os.path.join("/home2/bstephenson/WASD/WASD/clips_audios/", 'val')

    """
    d_train = AudioVideoDatasetAuxLosses(audio_train_path, video_train_path,
                                      io_config['csv_train_full'], clip_lenght,
                                      image_size, video_data_transforms['train'],
                                      do_video_augment=True)
    """

    """
    d_train1 = AudioVideoDatasetAuxLosses(audio_train_path1, video_train_path1,
                                      '/home2/bstephenson/ASDNet/ava220928.csv', clip_lenght,
                                      image_size, video_data_transforms['train'],
                                      do_video_augment=True)
    print("loaded")
    """

    d_train1 = AudioVideoDatasetAuxLosses(audio_train_path1, video_train_path1,
                                      '/home2/bstephenson/ASDNet/oursBackTrain.csv', clip_lenght,
                                      image_size, video_data_transforms['train'],
                                      do_video_augment=True)

    
    d_train2 = AudioVideoDatasetAuxLosses(audio_train_path2, video_train_path2,
                                      '/home2/bstephenson/ASDNet/ava_activespeaker_train_augmented.csv', clip_lenght,
                                      image_size, video_data_transforms['train'],
                                      do_video_augment=True)

    

    d_train3 = AudioVideoDatasetAuxLosses(audio_train_path4, video_train_path4,
                                      '/home2/bstephenson/ASDNet/oursHighTrain.csv', clip_lenght,
                                      image_size, video_data_transforms['train'],
                                      do_video_augment=True)

    """
    d_train4 = AudioVideoDatasetAuxLosses(audio_train_path1, video_train_path1,
                                      #'/home2/bstephenson/ASDNet/ava220926downsample.csv', clip_lenght,
                                      '/home2/bstephenson/ASDNet/ava220926.csv', clip_lenght,
                                      image_size, video_data_transforms['train'],
                                      do_video_augment=True)
    """
    

    d_train5 = AudioVideoDatasetAuxLosses(audio_train_path3, video_train_path3,
                                      '/home2/bstephenson/WASD/WASD/csv/train_orig.csv', clip_lenght,
                                      image_size, video_data_transforms['train'],
                                      do_video_augment=True)

    """
    d_val = AudioVideoDatasetAuxLosses(audio_val_path, video_val_path,
                                    '/home2/bstephenson/ASDNet/ava_activespeaker_val_augmented.csv', clip_lenght,
                                    image_size, video_data_transforms['val'],
                                    do_video_augment=False)


    d_train2 = AudioVideoDatasetAuxLosses(audio_train_path2, video_train_path2,
                                      '/home2/bstephenson/ASDNet/ava_activespeaker_train_augmented.csv', clip_lenght,
                                      image_size, video_data_transforms['train'],
                                      do_video_augment=True)

    d_val = AudioVideoDatasetAuxLosses(audio_val_path, video_val_path,
                                    '/home2/bstephenson/ASDNet/ava_activespeaker_val_augmented.csv', clip_lenght,
                                    image_size, video_data_transforms['val'],
                                    do_video_augment=False)



    d_train2 = AudioVideoDatasetAuxLosses(audio_train_path2, video_train_path2,
                                      '/home2/bstephenson/WASD/WASD/csv/train_orig.csv', clip_lenght,
                                      image_size, video_data_transforms['train'],
                                      do_video_augment=True)

    """
    """
    d_val = AudioVideoDatasetAuxLosses(audio_val_path, video_val_path,
                                    '/home2/bstephenson/WASD/WASD/csv/val_orig.csv', clip_lenght,
                                    image_size, video_data_transforms['val'],
                                    do_video_augment=False)
    """

    d_val1 = AudioVideoDatasetAuxLosses(audio_train_path1, video_train_path1,
                                      '/home2/bstephenson/ASDNet/oursBackTest.csv', clip_lenght,
                                      image_size, video_data_transforms['val'],
                                      do_video_augment=True)

    d_val2 = AudioVideoDatasetAuxLosses(audio_train_path4, video_train_path4,
                                      '/home2/bstephenson/ASDNet/oursHighTest.csv', clip_lenght,
                                      image_size, video_data_transforms['val'],
                                      do_video_augment=True)


    #print(len(d_train2))


    #dl_train = DataLoader(d_train3, batch_size=opt_config['batch_size'],
    #                      shuffle=True, num_workers=opt_config['threads'])
    #dl_train = DataLoader(d_train1 + d_train2 + d_train3 + d_train5, batch_size=opt_config['batch_size'],
    #                      shuffle=True, num_workers=opt_config['threads'])
    dl_train = DataLoader(d_train1 + d_train3, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'])
    #dl_train = DataLoader(d_train2, batch_size=opt_config['batch_size'],
    #                      shuffle=True, num_workers=opt_config['threads'])
    dl_val = DataLoader(d_val1 + d_val2, batch_size=opt_config['batch_size'],
                        shuffle=True, num_workers=opt_config['threads'])

    model = optimize_av_losses(backbone, dl_train, dl_val, device,
                                  criterion, optimizer, scheduler,
                                  num_epochs=opt_config['epochs'],
                                  models_out=target_models, log=log)
