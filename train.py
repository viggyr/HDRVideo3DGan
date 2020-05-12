import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as Generator, Discriminator
from parse_config import ConfigParser
from trainer import Trainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    generator = config.init_obj('arch_generator', module_arch)
    logger.info(generator)
    
    discriminator = config.init_obj('arch_discriminator', module_arch)
    logger.info(discriminator)

    # get function handles of loss and metrics
    criterion_discriminator = getattr(module_loss, config['discriminator_loss'])
    criterion_generator = getattr(module_loss, config['generator_loss'])_loss
    criterion_adversarial = getattr(module_loss, config['adversarial'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params_g = filter(lambda p: p.requires_grad, generator.parameters())
    trainable_params_d = filter(lambda p: p.requires_grad, discriminator.parameters())
    optimizer_g = config.init_obj('optimizer', torch.optim, trainable_params_g)
    optimizer_d = config.init_obj('optimizer', torch.optim, trainable_params_d)


    lr_scheduler_g = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_g)
    lr_scheduler_d = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer_d)

    trainer = Trainer(generator_model, discriminator_model, criterion_generator, criterion_discriminator, criterion_adversarial, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler_g=lr_scheduler_g,
                     lr_scheduler_d=lr_scheduler_d)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='HDR Video 3D Gan')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
