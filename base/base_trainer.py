import os
import logging
import json
import math
import torch
import datetime
from utils import helpers
from utils import logger
import utils.lr_scheduler
from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback
import csv
import wandb

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class BaseTrainer:
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, test_loader=None, train_logger=None):
        self.model = model
        self.loss = loss
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1
        self.improved = False
        self.moving_miou = [float("-inf") for i in range(5)]
        self.best_val = float("-inf")
        self.not_improved_count = 0

        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        if config["use_synch_bn"]:
            self.model = convert_model(self.model)
            self.model = DataParallelWithCallback(self.model, device_ids=availble_gpus)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
        self.model.to(self.device)

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.csv_log = os.path.join(config['exp_dir'], "val_overall_metrics.csv")
        self.train_csv_log = os.path.join(config['exp_dir'], "train_overall_metrics.csv")
        self.test_csv_log = os.path.join(config['exp_dir'], "test_overall_metrics.csv")
        self.plot_log = os.path.join(config['exp_dir'], "test_miou.csv")
        self.plot = os.path.join(config['exp_dir'], "plot.csv")
        self.movingavg_log = os.path.join(config['exp_dir'], "moving_avg.csv")
        self.train_iou = os.path.join(config['exp_dir'], "train_miou.csv")
        self.val_iou = os.path.join(config['exp_dir'], "val_miou.csv")
        self.best_classiou = -math.inf

        # OPTIMIZER
        if self.config['optimizer']['differential_lr']:
            if isinstance(self.model, torch.nn.DataParallel):
                trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.module.get_decoder_params())},
                                    {'params': filter(lambda p:p.requires_grad, self.model.module.get_backbone_params()), 
                                    'lr': config['optimizer']['args']['lr'] / 10}]
            else:
                trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.get_decoder_params())},
                                    {'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()), 
                                    'lr': config['optimizer']['args']['lr'] / 10}]
        else:
            trainable_params = filter(lambda p:p.requires_grad, self.model.parameters())
        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)

        # MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = cfg_trainer.get('early_stop', math.inf)

        # CHECKPOINTS & TENSOBOARD
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        self.checkpoint_dir = self.config['episode_dir']#os.path.join(cfg_trainer['save_dir'], self.config['name'], start_time)
        helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        if resume: self._resume_checkpoint(resume)

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus
    
    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            # RUN TRAIN (AND VAL)
            results = self._train_epoch(epoch)
            
            self.logger.info(f'\n         ## Info for epoch {epoch} ## ')
            for k, v in results.items():
                self.logger.info(f'         {str(k):15s}: {v}')
            
            with open(self.train_csv_log, "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.config['episode'], epoch])
                for k,v in results.items():
                    writer.writerow([k,v])

            with open(self.train_iou, "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.config['episode'], epoch, results['Mean_IoU'], results['Class_IoU'][0], results['Class_IoU'][1]])
            if self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)
                
                with open(self.csv_log, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([self.config['episode'], epoch])
                        for k,v in results.items():
                            writer.writerow([k,v])
                
                
                with open(self.val_iou, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.config['episode'], epoch, results['Mean_IoU'], results['Class_IoU'][0], results['Class_IoU'][1]])

                # LOGGING INFO
                self.logger.info(f'\n         ## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    self.logger.info(f'         {str(k):15s}: {v}')
            
            if self.train_logger is not None:
                log = {'epoch' : epoch, **results}
                self.train_logger.add_entry(log)

            # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
            if self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
                try:
                    if self.mnt_mode == 'min': self.improved = (log[self.mnt_metric] < self.mnt_best)
                    else: self.improved = (log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                    break
                    
                if self.improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stoping:
                    self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    self.logger.warning('Training Stoped')
                    break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)
                if self.improved:
                    test_results = self._test_epoch(epoch)
                    
                    self.logger.info(f'\n         ## Info for epoch {epoch} ## ')
                    for k, v in test_results.items():
                        self.logger.info(f'         {str(k):15s}: {v}')
                    
                    with open(self.plot_log, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([self.config['episode'], epoch, test_results['Mean_IoU'], test_results['Class_IoU'][0], test_results['Class_IoU'][1]])
                        
                    if self.best_classiou < test_results['Class_IoU'][1]:
                        self.best_classiou = test_results['Class_IoU'][1]
                        
                    with open(self.test_csv_log, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([self.config['episode'], epoch])
                        for k,v in test_results.items():
                            writer.writerow([k,v])
                        
        with open(self.plot, "a") as f:
            writer = csv.writer(f)
            writer.writerow([self.best_classiou])
                        
            

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        if save_best:
            filename = os.path.join(self.config['exp_dir'], f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError
        
    def _get_checkpoint_dir(self):
        return self.checkpoint_dir

    
