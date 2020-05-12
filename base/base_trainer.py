import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model_g, model_d, criterion_g, criterion_d, criterion_adversarial, metric_ftns, optimizer_g, optimizer_d, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model_g = model_g.to(self.device)
        self.model_d = model_d.to(self.device)
        if len(device_ids) > 1:
            self.model_g = torch.nn.DataParallel(model_g, device_ids=device_ids)
            self.model_d = torch.nn.DataParallel(model_d, device_ids=device_ids)
        
        self.criterion_g = criterion_g
        self.criterion_d = criterion_d
        self.criterion_adversarial = criterion_adversarial
        self.metric_ftns = metric_ftns
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.pre_training_iterations = cfg_trainer['pre_training_iterations']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError
    
    @abstractmethod
    def _pre_train_generator(self):
        """
        Training generator separately before actual
        gan training starts.

        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        self._pre_train_generator(self.pre_training_iterations)
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        

        arch_g = type(self.model_g).__name__
        arch_d = type(self.model_d).__name__
        self.logger.info("Saving checkpoint generator: {} ...".format(filename))
        state = {
            'arch_d': arch_d,
            'arch_g': arch_g,
            'epoch': epoch,
            'state_dict_g': self.model_g.state_dict(),
            'state_dict_d': self.model_d.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
 
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth..")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint_g['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architectures params from checkpoint.
        if checkpoint['config']['arch_d'] != self.config['arch_d'] or checkpoint['config']['arch_g'] != self.config['arch_g']:
            self.logger.warning("Warning: Architectures configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model_g.load_state_dict(checkpoint['state_dict_g'])
        self.model_d.load_state_dict(checkpoint['state_dict_d'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer_g']['type'] != self.config['optimizer_g']['type'] or checkpoint['config']['optimizer_d']['type'] != self.config['optimizer_d']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            # This would reload the state of optimizers such as learning rate...
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
