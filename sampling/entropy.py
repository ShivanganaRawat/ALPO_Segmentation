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

class Entropy_Sampling():
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


    def entropy_score(self, prob_map):
        return (-1) * np.sum(np.multiply(prob_map, np.log(prob_map)))


    def update_pools(self, args, config, episode):
        unlabeled_loader = self.get_instance(dataloaders, 'unlabeled_loader', config)

        unlabeled_file = os.path.join(config["episode_dir"],"unlabeled.txt")
        unlabeled_reader = csv.reader(open(unlabeled_file, 'rt'))
        unlabeled_image_set = [r[0] for r in unlabeled_reader]

        # Model
        model = self.get_instance(models, 'arch', 
            config, unlabeled_loader.dataset.num_classes)
        availble_gpus = list(range(torch.cuda.device_count()))
        device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(os.path.join(config['exp_dir'], 
            "best_model.pth"), map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        # If during training, we used data parallel
        if 'module' in list(checkpoint.keys())[0] and not isinstance(model, 
            torch.nn.DataParallel):
            # for gpu inference, use data parallel
            if "cuda" in device.type:
                model = torch.nn.DataParallel(model)
            else:
            # for cpu inference, remove module
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:]
                    new_state_dict[name] = v
                checkpoint = new_state_dict

        # load
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

        information_content = []
        tbar = tqdm(unlabeled_loader, ncols=130)
        with torch.no_grad():
            for img_idx, (data, target) in enumerate(tbar):
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                output = output.squeeze(0).cpu().numpy()
                output = F.softmax(torch.from_numpy(output))
                uncertainty_score = self.entropy_score(output.numpy())
                information_content.append([unlabeled_image_set[img_idx], 
                    uncertainty_score])

        information_content = sorted(information_content, 
            key= lambda x: x[1], reverse=True)
        information_content = information_content[:args.batch_size]

        new_batch = [x[0] for x in information_content]

        labeled = os.path.join(config['episode_dir'],"labeled.txt")
        labeled_reader = csv.reader(open(labeled, 'rt'))
        labeled_image_set = [r[0] for r in labeled_reader]

        new_labeled = labeled_image_set + new_batch
        new_labeled.sort()

        new_unlabeled = list(set(unlabeled_image_set) - set(new_batch))
        new_unlabeled.sort()



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

