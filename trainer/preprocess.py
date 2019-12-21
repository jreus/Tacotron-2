import argparse
import os
from multiprocessing import cpu_count

from datasets import preprocessor
from trainer.hparams import hparams
from tqdm import tqdm


def preprocess(args, input_folders, out_dir, hparams):
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)
	print("RUNNING PREPROCESSOR ON {}".format(input_folders))
	metadata = preprocessor.build_from_path(hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

def norm_data(args):

	merge_books = (args.merge_books=='True')

	print('Selecting data folders..')
	supported_dataset_formats = ['LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS']
	if args.datasetformat not in supported_dataset_formats:
		raise ValueError('dataset type {} does not belong to supported dataset types: {}'.format(args.datasetformat, supported_dataset_formats))

	if args.datasetformat.startswith('LJSpeech'):
		return args.datasetdir


	if args.datasetformat == 'M-AILABS':
		supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU',
			'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
		if args.language not in supported_languages:
			raise ValueError('Please enter a supported language to use from M-AILABS dataset! \n{}'.format(supported_languages))

		supported_voices = ['female', 'male', 'mix']
		if args.voice not in supported_voices:
			raise ValueError('Please enter a supported voice option to use from M-AILABS dataset! \n{}'.format(supported_voices))

		path = os.path.join(args.datasetdir, args.language, 'by_book', args.voice)
		supported_readers = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
		if args.reader not in supported_readers:
			raise ValueError('Please enter a valid reader for your language and voice settings! \n{}'.format(
				supported_readers))

		path = os.path.join(path, args.reader)
		supported_books = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
		if merge_books:
			return [os.path.join(path, book) for book in supported_books]

		else:
			if args.book not in supported_books:
				raise ValueError('Please enter a valid book for your reader settings! \n{}'.format(
					supported_books))

			return [os.path.join(path, args.book)]


def run_preprocess(args, hparams):
	input_folders = norm_data(args)
	output_folder = args.preprocess_output_dir
	print("INPUT FOLDERS!", input_folders)
	preprocess(args, input_folders, output_folder, hparams)
