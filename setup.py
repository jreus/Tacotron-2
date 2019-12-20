from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'falcon==1.2.0',
    'inflect==0.2.5',
    'audioread==2.1.5',
    'librosa==0.5.1',
    'matplotlib==2.0.2',
    'numpy==1.14.0',
    'scipy==1.0.0',
    'tqdm==4.11.2',
    'Unidecode==0.4.20',
    'pyaudio==0.2.11',
    'sounddevice==0.3.10',
    'lws',
    'keras'
    ]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='TACOTRON-2 SIMPLE TRAINING EXAMPLE'
)
