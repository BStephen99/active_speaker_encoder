
PepperMint Role short-term encoder

The code in this repo was used to train the short-term encoder for the baseline model of the PepperMint role dataset.
It is primarily adapted from the https://github.com/fuankarion/active-speakers-context and https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py (the temporal shift module), as per the process used for the GraVi-T encoder (https://github.com/IntelLabs/GraVi-T).


Setup

Download the repo:

git clone https://github.com/BStephen99/active_speaker_encoder.git
cd active-speakers-context

Create a new conda environment:

conda env create -f environment.yml -n ASD

Process the data (face crops and audio clips):

Three datasets were used in our experiments: PepperMint Role (ours), AVA ActiveSpeaker Dataset, and WASD. 

-Preprocessed face crops and audio clips for the PepperMint Role dataset are available at https://repository.ortolang.fr/api/content/peppermint/head/ in the AudioClips and FaceCrops folders.

Download AVA videos from https://github.com/cvdfoundation/ava-dataset.

Download WASD videos from https://github.com/Tiago-Roxo/WASD

Process audio clips using the following scripts:
./data/extract_audio_tracks.py
./data/slice_audio_tracks.py

Extract face crops using:
./data/extract_face_crops_time.py


To train a new encoder, specify the location of the csv files and face crop and audio clip folders in ./core/config.py

Define the training and test sets (train_names, val_names) to be used in ./STE_train.py, as well as the model_name.

To train the model, run: python3 STE_train.py 11 0



To obtain visual and audio embeddings, define the mode (i.e., the dataset you would like to process - "pepper_back", #pepper_back, pepper_high, ava_train, ava_test or wasd_train, wasd_test ) in the config file.  

Define the path to the model (model_weights) and the output directory (target_directory) in STE_forward.py.

To process, run python3 STE_forward.py 11 0





