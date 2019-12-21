# Field Notes

This package is Tacotron-2 test trainer packaged for use with Google Cloud Compute.

This is a new branch on the fork of the original Tacotron-2 repository.
https://www.atlassian.com/git/tutorials/using-branches
```
git branch gcloud
git checkout gcloud
...do commits...
git push --set-upstream origin gcloud
```

TRAINING COMMAND FROM "OLD" TACOTRON *train.py*
```
python train.py --tacotron_input='../datasets/ljmini_data/train.txt' --model='Tacotron-2'

```

## RECOMMENDED PROJECT STRUCTURE
From (https://cloud.google.com/ml-engine/docs/packaging-trainer#project-structure)[https://cloud.google.com/ml-engine/docs/packaging-trainer#project-structure]

You can structure your training application in any way you like. However, the following structure is commonly used in AI Platform samples.

```
* ProjectRoot/
 * setup.py
 * trainer/
  * __init__.py
  * task.py
  * model.py
  * util.py
 * subpackageX/
  * __init__.py
  * somemodule.py
```

Your `setup.py` file should look something like this:
```
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['some_PyPI_package>=1.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
```

## A Tip on Training Neural Networks

While training neural networks, mostly a target error is defined alongside with a maximum amount of iterations to train. So for example, a target error could be 0.001MSE. Once this error has been reached, the training will stop - if this error has not been reached after the maximum amount of iterations, the training will also stop.

But it seems like you want to train until you know the network can't do any better. Saving the 'best' parameters like you're doing is a fine approach, but do realise that once some kind of minimum cost has been reached, the error won't fluctuate that much anymore. It won't be like the error suddenly goes up significantly, so it is not completely necessary to save the network.

There is no such thing as 'minimal cost' - the network is always trying to go to some local minima, and it will always be doing so. There is not really way you (or an algorithm) can figure out that there is no better error to be reached anymore.

tl;dr - just set a target reasonable target error alongside with a maximum amount of iterations.

## GET YOUR CLOUD STORAGE SET UP
Make a unique ID for my storage bucket...
```
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-taco1
echo $BUCKET_NAME
REGION=europe-west4
gsutil mb -l $REGION gs://$BUCKET_NAME
```


## HOW TO RUN LOCAL TEST
```
cd Tacotron-2.gcloud.test/
DATASETDIR=../datasets/LJSpeech-Mini/
PREPROCESS_DIR=../datasets/preprocess_ljspeechmini/
TACO_INPUT=$PREPROCESS_DIR/train.txt

gcloud ai-platform local train --module-name=trainer.task --package-path=trainer -- --datasetdir=$DATASETDIR --datasetformat=LJSpeech-1.1 --preprocess=True --preprocess-output-dir=$PREPROCESS_DIR --tacotron-input-dir=$PREPROCESS_DIR --modeltype='Tacotron-2' --tacotron_train_steps=5 --wavenet_train_steps=5 --verbosity=DEBUG
```

## Try to synthesize something...

```
python synthesize.py --model='Tacotron-2' --model_dir=output/logs-Tacotron-2/

```



## HOW TO PACKAGE FOR CLOUD PROCESSING

## HOW TO SUBMIT A job

https://cloud.google.com/ml-engine/docs/training-jobs
