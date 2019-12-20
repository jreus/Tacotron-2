# 2019 Jon Reus
#
"""Runs the training of the TACOTRON-2 two-part model"""

import argparse
import json
import os

import tensorflow as tf

from trainer.hparams import hparams
import trainer.preprocess as preprocess
import trainer.train as train


def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters
    set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
            'index' in tf_config['task']):
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=[
                '/job:ps',
                '/job:worker/task:%d' % tf_config['task']['index']
            ])
    return None


def train_and_evaluate(args):
    """Run the training and evaluate using the high level API."""

    def train_input():
        """Input function returning batches from the training
        data set from training.
        """
        return input_module.input_fn(
            args.train_files,
            num_epochs=args.num_epochs,
            batch_size=args.train_batch_size,
            num_parallel_calls=args.num_parallel_calls,
            prefetch_buffer_size=args.prefetch_buffer_size)

    def eval_input():
        """Input function returning the entire validation data
        set for evaluation. Shuffling is not required.
        """
        return input_module.input_fn(
            args.eval_files,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_parallel_calls=args.num_parallel_calls,
            prefetch_buffer_size=args.prefetch_buffer_size)

    train_spec = tf.estimator.TrainSpec(
        train_input, max_steps=args.train_steps)

    exporter = tf.estimator.FinalExporter(
        'census', input_module.SERVING_FUNCTIONS[args.export_format])
    eval_spec = tf.estimator.EvalSpec(
        eval_input,
        steps=args.eval_steps,
        exporters=[exporter],
        name='census-eval')

    run_config = tf.estimator.RunConfig(
        session_config=_get_session_config_from_env_var())
    run_config = run_config.replace(model_dir=args.job_dir)
    print('Model dir %s' % run_config.model_dir)
    estimator = model.build_estimator(
        embedding_size=args.embedding_size,
        # Construct layers sizes with exponential decay
        hidden_units=[
            max(2, int(args.first_layer_size * args.scale_factor**i))
            for i in range(args.num_layers)
        ],
        config=run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    # INPUT ARGUMENTS FOR PREPROCESSOR
    PARSER.add_argument('--datasetdir', default='gs://jcr.tacotrontest/datasets/', help='GCS file or local paths to training data in LJSpeech or M-AILABS format')
    PARSER.add_argument('--hparams', default='', help='Hyperparameter overrides hparams.py as a comma-separated list of name=value pairs')
    PARSER.add_argument('--dataset', default='LJSpeech-Mini')
    PARSER.add_argument('--preprocess', default=True, type=bool, help='Set this to false to skip preprocessing step')
    PARSER.add_argument('--language', default='en_US')
    PARSER.add_argument('--voice', default='female')
    PARSER.add_argument('--reader', default='mary_ann')
    PARSER.add_argument('--merge_books', default='False')
    PARSER.add_argument('--book', default='northandsouth')
    PARSER.add_argument('--output', default='training_data')
    PARSER.add_argument('--n_jobs', type=int, default=os.cpu_count(), help='Optional, number of worker process to parallelize across')
    # INPUT ARGUMENTS FOR TRAINING
    PARSER.add_argument('--tacotron_input', default='gs://jcr.tacotrontest/datasets/training_data/train.txt')
    PARSER.add_argument('--wavenet_input', default='gs://jcr.tacotrontest/datasets/tacotron_output/gta/map.txt')
    PARSER.add_argument('--logdir', help='Name of logging directory.', default='logs')
    PARSER.add_argument('--modeltype', default='Tacotron-2')
    PARSER.add_argument('--input_dir', default='gs://jcr.tacotrontest/datasets/training_data', help='folder to contain inputs sentences/targets')
    PARSER.add_argument('--output_dir', default='gs://jcr.tacotrontest/datasets/output', help='folder to contain synthesized mel spectrograms')
    PARSER.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training')
    PARSER.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
    PARSER.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    # These step numbers and intervals probably need to be a couple borders
    #  of magnitude higher (see original train.py call)
    PARSER.add_argument('--summary_interval', type=int, default=15,
    help='Steps between running summary ops')
    PARSER.add_argument('--embedding_interval', type=int, default=10,
    help='Steps between updating embeddings projection visualization')
    PARSER.add_argument('--checkpoint_interval', type=int, default=20,
    help='Steps between writing checkpoints')
    PARSER.add_argument('--eval_interval', type=int, default=10,
    help='Steps between eval on test data')
    PARSER.add_argument('--tacotron_train_steps', type=int, default=40, help='total number of tacotron training steps')
    PARSER.add_argument('--wavenet_train_steps', type=int, default=40, help='total number of wavenet training steps')
    PARSER.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    PARSER.add_argument('--verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'], default='INFO')
    PARSER.add_argument('--slack_url', default=None, help='slack webhook notification destination link')

    args = PARSER.parse_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Suppress C++ level warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # PREPROCESSING
    if args.preprocess is True:
        print('initializing preprocessing..')
        modified_hp = hparams.parse(args.hparams)
        assert args.merge_books in ('False', 'True')
        preprocess.run_preprocess(args, modified_hp)

    # MODEL TRAINING
    accepted_models = ['Tacotron', 'WaveNet', 'Tacotron-2']
    if args.modeltype not in accepted_models:
        raise ValueError('please enter a valid model to train: {}'.format(accepted_models))
    log_dir, hparams = train.prepare_run(args)
    if args.modeltype == 'Tacotron':
        train.tacotron_train(args, log_dir, hparams)
    elif args.modeltype == 'WaveNet':
        train.wavenet_train(args, log_dir, hparams, args.wavenet_input)
    elif args.modeltype == 'Tacotron-2':
        train.train(args, log_dir, hparams)
    else:
        raise ValueError('Model provided {} unknown! {}'.format(args.modeltype, accepted_models))
