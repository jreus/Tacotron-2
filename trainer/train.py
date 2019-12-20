import argparse
import os
from time import sleep

import trainer.infolog as infolog
import tensorflow as tf
from trainer.hparams import hparams
from trainer.infolog import log
from tacotron.synthesize import tacotron_synthesize
from tacotron.train import tacotron_train
from wavenet_vocoder.train import wavenet_train

log = infolog.log


def save_seq(file, sequence, input_path):
	'''Save Tacotron-2 training state to disk. (To skip for future runs)
	'''
	sequence = [str(int(s)) for s in sequence] + [input_path]
	with open(file, 'w') as f:
		f.write('|'.join(sequence))

def read_seq(file):
	'''Load Tacotron-2 training state from disk. (To skip if not first run)
	'''
	if os.path.isfile(file):
		with open(file, 'r') as f:
			sequence = f.read().split('|')

		return [bool(int(s)) for s in sequence[:-1]], sequence[-1]
	else:
		return [0, 0, 0], ''

def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
	run_name = args.name or args.model
	log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
	os.makedirs(log_dir, exist_ok=True)
	infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name, args.slack_url)
	return log_dir, modified_hp

def train(args, log_dir, hparams):
	state_file = os.path.join(log_dir, 'state_log')
	#Get training states
	(taco_state, GTA_state, wave_state), input_path = read_seq(state_file)

	if not taco_state:
		log('\n#############################################################\n')
		log('Tacotron Train\n')
		log('###########################################################\n')
		checkpoint = tacotron_train(args, log_dir, hparams)
		tf.reset_default_graph()
		#Sleep 1/2 second to let previous graph close and avoid error messages while synthesis
		sleep(0.5)
		if checkpoint is None:
			raise('Error occured while training Tacotron, Exiting!')
		taco_state = 1
		save_seq(state_file, [taco_state, GTA_state, wave_state], input_path)
	else:
		checkpoint = os.path.join(log_dir, 'taco_pretrained/')

	if not GTA_state:
		log('\n#############################################################\n')
		log('Tacotron GTA Synthesis\n')
		log('###########################################################\n')
		input_path = tacotron_synthesize(args, hparams, checkpoint)
		tf.reset_default_graph()
		#Sleep 1/2 second to let previous graph close and avoid error messages while Wavenet is training
		sleep(0.5)
		GTA_state = 1
		save_seq(state_file, [taco_state, GTA_state, wave_state], input_path)
	else:
		input_path = os.path.join('tacotron_' + args.output_dir, 'gta', 'map.txt')

	if input_path == '' or input_path is None:
		raise RuntimeError('input_path has an unpleasant value -> {}'.format(input_path))

	if not wave_state:
		log('\n#############################################################\n')
		log('Wavenet Train\n')
		log('###########################################################\n')
		checkpoint = wavenet_train(args, log_dir, hparams, input_path)
		if checkpoint is None:
			raise ('Error occured while training Wavenet, Exiting!')
		wave_state = 1
		save_seq(state_file, [taco_state, GTA_state, wave_state], input_path)

	if wave_state and GTA_state and taco_state:
		log('TRAINING IS ALREADY COMPLETE!!')
