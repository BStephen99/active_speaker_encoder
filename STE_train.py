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

from torch.utils.data import ConcatDataset


if __name__ == '__main__':
    #experiment Reproducibility
    torch.manual_seed(11)
    torch.cuda.manual_seed(22)
    torch.backends.cudnn.deterministic = True

    clip_lenght = int(sys.argv[1])
    cuda_device_number = str(sys.argv[2])
    image_size = (144, 144) #Dont forget to assign this same size on ./core/custom_transforms

    # check these 3 are in order, everythine else is kind of automated
    model_name = 'ste_encoder_tester'
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



    back_video_train_path = os.path.join(io_config['video_dir'], 'back')
    high_video_train_path = os.path.join(io_config['video_dir'], 'high')
    ava_video_train_path = os.path.join(io_config['ava_video_dir'], 'train')
    wasd_video_train_path = os.path.join(io_config['wasd_video_dir'], 'train')

    back_audio_train_path = os.path.join(io_config['audio_dir'], 'back')
    high_audio_train_path = os.path.join(io_config['audio_dir'], 'high')
    ava_audio_train_path = os.path.join(io_config['ava_audio_dir'], 'train')
    wasd_audio_train_path = os.path.join(io_config['wasd_audio_dir'], 'train')

    back_video_val_path = os.path.join(io_config['video_dir'], 'back')
    high_video_val_path = os.path.join(io_config['video_dir'], 'high')
    wasd_video_val_path = os.path.join(io_config['wasd_video_dir'], 'val')
    ava_video_val_path = os.path.join(io_config['ava_video_dir'], 'val')

    back_audio_val_path = os.path.join(io_config['audio_dir'], 'back')
    high_audio_val_path = os.path.join(io_config['audio_dir'], 'high')
    wasd_audio_val_path = os.path.join(io_config['wasd_audio_dir'], 'val')
    ava_audio_val_path = os.path.join(io_config['ava_audio_dir'], 'val')

   
  

    def make_dataset(name, split):
      """Factory to create datasets only when needed."""
      if name == "back":
          audio, video, csv = back_audio_train_path, back_video_train_path, io_config['csv_train_full']
      elif name == "high":
          audio, video, csv = high_audio_train_path, high_video_train_path, io_config['csv_train_full']
      elif name == "ava":
          if split == "train":
              audio, video, csv = ava_audio_train_path, ava_video_train_path, io_config["ava_csv_train_full"]
          else:
              audio, video, csv = ava_audio_val_path, ava_video_val_path, io_config["ava_csv_val_full"]
      elif name == "wasd":
          if split == "train":
              audio, video, csv = wasd_audio_train_path, wasd_video_train_path, io_config["wasd_csv_train_full"]
          else:
              audio, video, csv = wasd_audio_val_path, wasd_video_val_path, io_config["wasd_csv_val_full"]
      else:
          raise ValueError(f"Unknown dataset name: {name}")

      transform = video_data_transforms['train'] if split == "train" else video_data_transforms['val']
      do_video_augment = (split == "train")

      return AudioVideoDatasetAuxLosses(audio, video, csv, clip_lenght, image_size,
                                        transform, do_video_augment=do_video_augment,
                                        train_or_test=split)




    train_names = ["back", "high"]               # choose here
    val_names   = ["back", "high", "ava"]        # or ["ava", "wasd"], etc.

    train_datasets = [make_dataset(name, "train") for name in train_names]
    val_datasets   = [make_dataset(name, "test") for name in val_names]

    combined_train = ConcatDataset(train_datasets)
    combined_val   = ConcatDataset(val_datasets)

    



    # DataLoaders
    dl_train = DataLoader(combined_train, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'], pin_memory=True)



    dl_val = DataLoader(combined_val, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'], pin_memory=True)

    model = optimize_av_losses(backbone, dl_train, dl_val, device,
                                  criterion, optimizer, scheduler,
                                  num_epochs=opt_config['epochs'],
                                  models_out=target_models, log=log)
