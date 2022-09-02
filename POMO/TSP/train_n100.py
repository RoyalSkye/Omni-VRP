import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import torch
import logging
from utils.utils import create_logger, copy_all_src
from utils.functions import seed_everything
from TSPTrainer import TSPTrainer as Trainer
from TSPTrainer_Meta import TSPTrainer as Trainer_Meta

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE and torch.cuda.is_available()
CUDA_DEVICE_NUM = 0  # $ nohup python -u train_n100.py 2>&1 &, no need to use CUDA_VISIBLE_DEVICES=0

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

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [3001, ],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'seed': 1234,
    'epochs': 1000,
    'train_episodes': 100000,  # number of instances per epoch
    'train_batch_size': 64,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'general.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 510,  # epoch version of pre-trained model to laod.

    },
    'meta_params': {
        'enable': False,  # whether use meta-learning or not
        'meta_method': 'reptile',  # choose from ['reptile', 'reptile_ats']
        'data_type': 'distribution',  # choose from ["size", "distribution", "size_distribution"]
        'epochs': 10417,  # the number of meta-model updates: (1000*100000) / (3*50*64)
        'B': 3,  # the number of tasks in a mini-batch
        'k': 50,  # gradient decent steps in the inner-loop optimization of meta-learning method
        'meta_batch_size': 64,  # the batch size of the inner-loop optimization
        'num_task': 50,  # the number of tasks in the training task set
        'alpha': 0.99,  # used for the outer-loop optimization of reptile
        'alpha_decay': 0.999,
    }
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_n100__3000epoch',
        'filename': 'log.txt'
    }
}


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    seed_everything(trainer_params['seed'])

    if not trainer_params['meta_params']['enable']:
        print(">> Start POMO Training.")
        trainer = Trainer(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params)
    else:
        print(">> Start POMO-Meta Training.")
        trainer = Trainer_Meta(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    main()
