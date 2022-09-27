import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import torch
import logging
from utils.utils import create_logger, copy_all_src
from utils.functions import seed_everything
from TSPTester import TSPTester as Tester

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE and torch.cuda.is_available()
CUDA_DEVICE_NUM = 0

##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'seed': 2023,
    'model_load': {
        'path': '../../pretrained/var_size_exp1/pomo_adam',  # directory path of pre-trained model and log files saved.
        'epoch': 78125,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 7000,
    'test_batch_size': 10000,
    'augmentation_enable': True,
    'test_robustness': False,
    'aug_factor': 8,
    'aug_batch_size': 50,
    'test_set_path': './adv_tsp100_uniform.pkl',  # '../../data/TSP/tsp100_uniform.pkl'
    'test_set_opt_sol_path': './sol_adv_tsp100_uniform.pkl',  # '../../data/TSP/gurobi/tsp100_uniform.pkl'
    'fine_tune_params': {
        'enable': True,  # evaluate few-shot generalization
        'fine_tune_episodes': 3000,  # how many data used to fine-tune the pretrained model
        'k': 50,  # gradient decent steps in the inner-loop optimization of meta-learning method
        'fine_tune_batch_size': 64,  # the batch size of the inner-loop optimization
        'fine_tune_set_path': './adv_tsp100_uniform.pkl',
        'augmentation_enable': False,
        'optimizer': {
            'lr': 1e-4 * 0.1,
            'weight_decay': 1e-6
        },
    }
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test_tsp_n50',
        'filename': 'log.txt'
    }
}


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    seed_everything(tester_params['seed'])

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    # TODO: 1. why not use test dataset to fine-tune the model?
    main()
