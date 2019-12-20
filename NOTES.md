# Field Notes for Running Tacotron-2 Locally

## Requirements

- **Machine Setup:**
You need to have python 3 installed along with [Tensorflow 1.3, 1.4, or 1.5](https://www.tensorflow.org/install/).

This implementation will not work (easily) with Tensorflow 2.0 without some adaptations!

You need to install some Linux dependencies to ensure audio libraries work properly:
> apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools

Finally, you can install the requirements. If you are an Anaconda user: (else replace **pip** with **pip3** and **python** with **python3**)

> pip install -r requirements.txt

# WORKFLOW

Since the two parts of the global model are trained separately, we can start by training the feature prediction model to use his predictions later during the wavenet training.

Before proceeding, you must pick the hyperparameters that suit best your needs. While it is possible to change the hyper parameters from command line during preprocessing/training, I still recommend making the changes once and for all on the **hparams.py** file directly.

To pick optimal fft parameters, I have made a **griffin_lim_synthesis_tool** notebook that you can use to invert real extracted mel/linear spectrograms and choose how good your preprocessing is. All other options are well explained in the **hparams.py** and have meaningful names so that you can try multiple things with them.

# Preprocessing
Before running the following steps, please make sure you are inside **Tacotron-2 folder**

> cd Tacotron-2

Preprocessing can then be started using:

> python preprocess.py

dataset can be chosen using the **--dataset** argument. If using M-AILABS dataset, you need to provide the **language, voice, reader, merge_books and book arguments** for your custom need. Default is **Ljspeech**.

Example preprocessing of LJSpeech-Mini dataset:

> conda activate speechml
> python preprocess.py --dataset='LJSpeech-Mini' --base_dir=../datasets/


Example M-AILABS:

> python preprocess.py --dataset='M-AILABS' --language='en_US' --voice='female' --reader='mary_ann' --merge_books=False --book='northandsouth'

This should take no longer than a **few minutes.**

# Training:
To **train both models** sequentially (one after the other):
> python train.py --model='Tacotron-2'

To only train the feature prediction model and wavenet separately use:
> python train.py --model='Tacotron'
> python train.py --model='WaveNet'

By default checkpoints will be made each **5000 steps** and stored under **logs-Tacotron** and **logs-Wavenet**

**Note:**
- If model argument is not provided, training will default to Tacotron-2 model training. (both models)
- Please refer to train arguments under [train.py](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/train.py) for a set of options you can use.
- It is now possible to make wavenet preprocessing alone using **wavenet_proprocess.py**.

### EXAMPLE TRAINING WITH VERY SMALL STEP NUMBER AND MORE REGULAR CHECKPOINTING, AND IGNORE ANY PREEXISTING CHECKPOINTS...
> python train.py --model='Tacotron-2' --tacotron_input=../datasets/ljmini_data/train.txt --restore=False --summary_interval=10 --embedding_interval=10 --checkpoint_interval=20 --tacotron_train_steps=40 --wavenet_train_steps=40 --tf_log_level=1

Checkpoints end up in logs-Tacotron-2/taco_pretrained/

# Monitoring the training process

Point tensorboard at the training process like so...
> conda activate speechml
> tensorboard --logdir=logs-Tacotron-2/

Once the tensorboard server has finished booting, visit it in a browser at http://jons-MBP:6006

# Synthesis:
To **synthesize audio** in an **End-to-End** (text to audio) manner (both models at work):
> python synthesize.py --model='Tacotron-2'

For the spectrogram prediction network (separately), there are **three types** of mel spectrograms synthesis:

- **Evaluation** (synthesis on custom sentences). This is what we'll usually use after having a full end to end model.
> python synthesize.py --model='Tacotron'

- **Natural synthesis** (let the model make predictions alone by feeding last decoder output to the next time step).
> python synthesize.py --model='Tacotron' --mode='synthesis' --GTA=False

- **Ground Truth Aligned synthesis** (DEFAULT: the model is assisted by true labels in a teacher forcing manner). This synthesis method is used when predicting mel spectrograms used to train the wavenet vocoder. (yields better results as stated in the paper)
> python synthesize.py --model='Tacotron' --mode='synthesis' --GTA=True

Synthesizing the **waveforms** conditionned on previously synthesized Mel-spectrograms (separately) can be done with:
> python synthesize.py --model='WaveNet'

**Note:**
- If model argument is not provided, synthesis will default to Tacotron-2 model synthesis. (End-to-End TTS)
- Please refer to synthesis arguments under [synthesize.py](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/synthesize.py) for a set of options you can use.
