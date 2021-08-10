import os
import sys
import json
import csv
import random
import argparse
import torch
import dataloaders
import models
import inspect
import math
from datetime import datetime
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import wandb
from wandb import AlertLevel
from sampling.randm import Random_Sampling
from sampling.entropy import Entropy_Sampling
from sampling.lc import Lc_Sampling
from sampling.margin import Margin_Sampling
from sampling.dbal import Dbal_Sampling


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    parser.add_argument('--batch_size', type=int, 
        help='Batch size for active learning')
    parser.add_argument('--init_batch_size', type=int, 
        help='Initial batch size for active learning')
    parser.add_argument('--max_episode', type=int, 
        help='Maximum active learning episodes')
    parser.add_argument('--sampling_strategy', type=str, default='Random', 
        help='Uncertainty sampling for active learning')
    parser.add_argument('--seed', type=int, 
        help='Seed for training')
    parser.add_argument('--exp_name', type=str,
        help='Experiment name')
    parser.add_argument('--ntrain', type=int, 
        help='Number of images for training')

    parser.add_argument('--resume_al', default=False, help='Resume active learning', action='store_true')

    args = parser.parse_args()

    return args


def load_config(args):
    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    
    return config


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def create_expdir(cfg, args):
    #setting up working directory
    if not os.path.exists(cfg['work_dir']):
        os.mkdir(cfg['work_dir'])
    # Create "DATASET" specific directory
    dataset_out_dir = os.path.join(cfg['work_dir'], cfg['dataset'])
    if not os.path.exists(dataset_out_dir):
        os.mkdir(dataset_out_dir)
    dataset_out_dir = os.path.join(dataset_out_dir, args.sampling_strategy)
    if not os.path.exists(dataset_out_dir):
        os.mkdir(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory all logs, label, unlabeled, val sets are stored 
    # E.g., output/dataset/{timestamp or cfg.EXP_NAME based on arguments passed}
    if cfg['exp_name'] == 'auto':
        now = datetime.now()
        exp_dir = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
    else:
        exp_dir = cfg['exp_name']

    exp_dir = os.path.join(dataset_out_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    else:
        print("=============================")
        print("Experiment directory already exists: {}. Reusing it may lead to loss of old data in the directory.".format(exp_dir))
        print("=============================")
    cfg['exp_dir'] = exp_dir
    cfg['val_loader']['args']['load_from'] = os.path.join(cfg['exp_dir'], "val.csv")
    return cfg


def initial_sampling(args, config):
    #create the train image set
    train_file = os.path.join(config["data_dir"],"trainval.csv")
    image_reader = csv.reader(open(train_file, 'rt'))
    image_set = [r[0] for r in image_reader]
    
    random.shuffle(image_set)
    
    if config['dataset'] == 'Apple':
        num_train = 681
    elif config['dataset'] == 'Wheat':
        num_train = 2047
    else:
        num_train = 1203
        
    train_set = image_set[:num_train]
    val_set = image_set[num_train:]
    
    
    with open(os.path.join(config['exp_dir'], "train.csv"), 'w') as f:
        writer = csv.writer(f)
        for image in train_set:
            writer.writerow([image])
            
    with open(os.path.join(config['exp_dir'], "val.csv"), 'w') as f:
        writer = csv.writer(f)
        for image in val_set:
            writer.writerow([image])

    
    #create initial labeled and unlabeled image set
    initial_labeled = random.sample(train_set, args.init_batch_size)
    initial_labeled.sort()
    initial_unlabeled = list(set(train_set) - set(initial_labeled))
    initial_unlabeled.sort()


    with open(os.path.join(config['episode_dir'], "labeled.txt"), 'w') as f:
        writer = csv.writer(f)
        for image in initial_labeled:
            writer.writerow([image])
            
    with open(os.path.join(config['episode_dir'], "unlabeled.txt"), 'w') as f:
        writer = csv.writer(f)
        for image in initial_unlabeled:
            writer.writerow([image])



def al():
    #getting args and config
    args = parse_args()
    config = load_config(args)
    
    config['seed'] = args.seed
    config['exp_name'] = args.exp_name
    
    
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    #number of gpus tp use
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        
    if args.sampling_strategy == 'Random':
        strategy = Random_Sampling()
    elif args.sampling_strategy == 'Entropy':
        strategy = Entropy_Sampling()
    elif args.sampling_strategy == 'Confidence':
        strategy = Lc_Sampling()
    elif args.sampling_strategy == 'Margin':
        strategy = Margin_Sampling()
    elif args.sampling_strategy == 'Dbal':
        strategy = Dbal_Sampling()

    if args.resume_al == False :
        episode = 0
        config = create_expdir(config, args)
        config = strategy.create_episodedir(config, episode)
        initial_sampling(args, config)
    else:
        episode = config['episode']



    while episode < args.max_episode:

        print("Training on the new labeled pool")
        config = strategy.train_model(args=args, config=config)
        print("Updating pools!")
        config = strategy.update_pools(args=args, config=config, episode=episode)
        with open(os.path.join(config['exp_dir'], "episodes_completed.txt"), "a") as f:
            f.write("Episode {0} done!\n".format(episode))
        episode +=1
        

    print("Training on the final labeled pool")
    with open(os.path.join(config['exp_dir'], "episodes_completed.txt"), "a") as f:
            f.write("Final training in progress....\n")

        
    config = strategy.train_model(args=args, config=config)
        
    print('Training Done!')
    with open(os.path.join(config['exp_dir'], "episodes_completed.txt"), "a") as f:
            f.write("Training done!\n")



if __name__=='__main__':
    al()


