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

class Random_Sampling():
    def __init__(self):
        pass
    
    def get_instance(self, module, name, config, *args):
        # GET THE CORRESPONDING CLASS / FCT 
        return getattr(module, config[name]['type'])(*args, **config[name]['args'])
    
    def create_episodedir(self, cfg, episode):
        episode_dir = os.path.join(cfg['exp_dir'], "episode"+str(episode))
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        else:
            print("=============================")
            print("Episode directory already exists: {}. Reusing it may lead to loss of old data in the directory.".format(episode_dir))
            print("=============================")

        cfg['episode'] = episode
        cfg['episode_dir'] = episode_dir
        cfg['trainer']['save_dir'] = os.path.join(episode_dir,cfg['trainer']['original_save_dir'])
        cfg['trainer']['log_dir'] = os.path.join(episode_dir,cfg['trainer']['original_log_dir'])

        cfg['labeled_loader']['args']['load_from'] = os.path.join(episode_dir, "labeled.txt")
        cfg['unlabeled_loader']['args']['load_from'] = os.path.join(episode_dir, "unlabeled.txt")

        return cfg

    
    def random_sample(self, args, config):
        #create the train image set
        unlabeled_file = os.path.join(config["episode_dir"],"unlabeled.txt")
        unlabeled_reader = csv.reader(open(unlabeled_file, 'rt'))
        unlabeled_image_set = [r[0] for r in unlabeled_reader]

        #create initial labeled and unlabeled image set
        new_batch = random.sample(unlabeled_image_set, args.batch_size)

        labeled = os.path.join(config['episode_dir'],"labeled.txt")
        labeled_reader = csv.reader(open(labeled, 'rt'))
        labeled_image_set = [r[0] for r in labeled_reader]
        new_labeled = labeled_image_set + new_batch
        new_labeled.sort()

        unlabeled = os.path.join(config['episode_dir'],"unlabeled.txt")
        unlabeled_reader = csv.reader(open(unlabeled, 'rt'))
        unlabeled_image_set = [r[0] for r in unlabeled_reader]
        new_unlabeled = list(set(unlabeled_image_set) - set(new_batch))
        new_unlabeled.sort()

        return new_labeled, new_unlabeled

    def train_model(self, args, config):
        train_logger = Logger()

        # DATA LOADERS
        labeled_loader = self.get_instance(dataloaders, 'labeled_loader', config)
        val_loader = self.get_instance(dataloaders, 'val_loader', config)
        test_loader = self.get_instance(dataloaders, 'test_loader', config)

        # MODEL
        model = self.get_instance(models, 'arch', config, labeled_loader.dataset.num_classes)
        #print(f'\n{model}\n')

        # LOSS
        loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

        # TRAINING
        trainer = Trainer(
            model=model,
            loss=loss,
            resume=args.resume,
            config=config,
            train_loader=labeled_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_logger=train_logger)

        trainer.train()

        config['checkpoint_dir'] = trainer._get_checkpoint_dir()
        config_save_path = os.path.join(config['checkpoint_dir'], 'updated_config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=True)


        return config


    def update_pools(self, args, config, episode):
        new_labeled, new_unlabeled = self.random_sample(args, config)

        config = self.create_episodedir(config, episode+1)

        with open(os.path.join(config['episode_dir'], "labeled.txt"), 'w') as f:
            writer = csv.writer(f)
            for image in new_labeled:
                writer.writerow([image])

        with open(os.path.join(config['episode_dir'], "unlabeled.txt"), 'w') as f:
            writer = csv.writer(f)
            for image in new_unlabeled:
                writer.writerow([image])

        return config


