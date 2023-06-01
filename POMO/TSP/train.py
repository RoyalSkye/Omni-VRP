import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import torch
import logging
from utils.utils import create_logger, copy_all_src
from utils.functions import seed_everything
from TSPTrainer_pomo import TSPTrainer as Trainer_pomo
from TSPTrainer_meta import TSPTrainer as Trainer_meta

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
    'meta_update_encoder': True,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'seed': 1234,
    # 'batch_size': 64,
    'logging': {
        'model_save_interval': 50000,
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
    # load previous checkpoint for meta-training
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': '../../pretrained/debug',  # directory path of pre-trained model and log files saved.
        'epoch': 100000,  # epoch version of pre-trained model to laod.
    },
    # load pretrain model for meta-training instead of meta-training from scratch
    'pretrain_load': {
        'enable': False,
        'path': '../../pretrained/POMO-TSP/checkpoint-3000-tsp100-instance-norm.pt',  # be careful with the type of the normalization layer
    }
}

meta_params = {
    'enable': True,  # whether use meta-learning or not
    'curriculum': True,  # adaptive sample task
    'meta_method': 'maml',  # choose from ['maml', 'fomaml', 'maml_fomaml', 'reptile']
    'data_type': 'size_distribution',  # choose from ["size", "distribution", "size_distribution"]
    'epochs': 250000,  # the number of meta-model updates: (250*100000) / (1*5*64)
    'early_stop_epoch': 50000,  # switch from maml to fomaml
    'B': 1,  # the number of tasks in a mini-batch
    'k': 1,  # gradient decent steps in the inner-loop optimization of meta-learning method
    'L': 0,  # bootstrap steps
    'meta_batch_size': 64,  # will be divided by 2 if problem_size >= 150
    'update_weight': 100,  # update weight of each task per X iters
    'sch_epoch': 225000,  # for the task scheduler of size setting, where sch_epoch = 0.9 * epochs
    'solver': 'lkh3_offline',  # solver used to update the task weights, choose from ["bootstrap", "lkh3_online", "lkh3_offline", "best_model"]
    'alpha': 0.99,  # params for the outer-loop optimization of reptile
    'alpha_decay': 0.999,  # params for the outer-loop optimization of reptile
    'beta': 0.9,  # loss weight
}

logger_params = {
    'log_file': {
        'desc': 'train_tsp',
        'filename': 'log.txt'
    }
}


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    seed_everything(trainer_params['seed'])

    if not meta_params['enable']:
        print(">> Start TSP-POMO Training.")
        trainer = Trainer_pomo(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params, meta_params=meta_params)
    elif meta_params['meta_method'] in ['maml', 'fomaml', 'maml_fomaml', 'reptile']:
        print(">> Start TSP-POMO-{} Training.".format(meta_params['meta_method']))
        trainer = Trainer_meta(env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params, meta_params=meta_params)
    else:
        raise NotImplementedError

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


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occumpy_mem(cuda_device):
    """
    Occupy GPU memory in advance for size setting.
    """
    torch.cuda.set_device(cuda_device)
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    block_mem = int((total-used) * 0.5)
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x


if __name__ == "__main__":
    if meta_params["data_type"] in ["size", "size_distribution"]:
        occumpy_mem(CUDA_DEVICE_NUM)  # reserve GPU memory for large size instances
    main()
