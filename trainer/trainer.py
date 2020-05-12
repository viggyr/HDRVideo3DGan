import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model_g,model_d, criterion_generator, criterion_discriminator,criterion_adversarial, metric_ftns, optimizer_g,optimizer_d, config, data_loader,
                 valid_data_loader=None, lr_scheduler_g=None,lr_scheduler_d=None, len_epoch=None):
        super().__init__(model_g,model_d, criterion_g, criterion_d,criterion_adversarial, metric_ftns, optimizer_g,optimizer_d, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d

        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('generator_loss','discriminator_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('generator_loss','discriminator_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        
    def _pre_train_generator(self, iterations: int):
        self.model_g.train()
        for batch_idx, (im1,im2,im3, target) in enumerate(self.data_loader):
            if batch_idx==iterations:
                return
            im1,im2,im3,target = im1.to(self.device),im2.to(self.device),im3.to(self.device), target.to(self.device)

            self.optimizer_g.zero_grad()
            output = self.model_g(im1, im2, im3)
            loss = self.criterion_g(output, target)
            loss.backward()
            self.optimizer_g.step()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains psnr and losses in this epoch.
        """
        self.model_g.train()
        self.model_d.train()
        self.train_metrics.reset()
        for batch_idx, (im1,im2,im3, hdr) in enumerate(self.data_loader):
            im1,im2,im3,hdr = im1.to(self.device),im2.to(self.device),im3.to(self.device), hdr.to(self.device)

            # Since the output dimensions would be 5 (batch_size, channels, frames,height, width) but we have only one frame, we squeeze and remove that dimension
            output = self.model_g((im1, im2, im3)).squeeze(2)
            
            self.optimizer_d.zero_grad()
            
            for param in model_d.parameters():
                param.requires_grad = False
            pred_real = self.model_d(torch.cat([im1, im2, im3, hdr], dim=1))

            # using detach to make sure generator is not getting trained while we train discriminator
            pred_fake = self.model_d(torch.cat([im1, im2, im3, output], dim=1).detach())
            loss = self.criterion_d(pred_real, pred_fake)
            loss.backward()
            self.optimizer_d.step()
             #Train Generator
            optimizer_g.zero_grad()

            pixel_loss = criterion_g(output, hdr)

            # this is adversarial loss. Fool the discrimnator by generating better pictures
            for param in model_d.parameters():
                param.requires_grad = False
            g_loss = self.criterion_adversarial(discriminator(torch.cat([im1, im2, im3, output], dim=1)))

            total_loss = 100*pixel_loss + g_loss

            total_loss.backward()    
            optimizer_g.step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('discriminator_loss', criterion_d.item())
            self.train_metrics.update('generator_loss', total_loss.item())
            # Update PSNR
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, hdr))

            if batch_idx % self.log_step == 0:
                self.logger.debug(f'Train Epoch: {epoch} {self._progress(batch_idx)} total_loss_g: {total_loss.item()} discriminator_loss: {criterion_d.item()}')
                display = torch.stack([im1.cpu().data[0], im2.cpu().data[0], im3.cpu().data[0], hdr.cpu().data[0],output.cpu().data[0]], dim=1)
                self.writer.add_image("Sequence", make_grid(display, nrow=5, normalize=True))



            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model_d.eval()
        self.model_g.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (im1,im2,im3, hdr) in enumerate(self.valid_data_loader):
                im1,im2,im3,hdr = im1.to(self.device),im2.to(self.device),im3.to(self.device), hdr.to(self.device)

                # Since the output dimensions would be 5 (batch_size, channels, frames,height, width) but we have only one frame, we squeeze and remove that dimension
                output = self.model_g((im1, im2, im3)).squeeze(2)

                pred_real = self.model_d(torch.cat([im1, im2, im3, hdr], dim=1))

                # using detach to make sure generator is not getting trained while we train discriminator
                pred_fake = self.model_d(torch.cat([im1, im2, im3, output], dim=1))
                loss = self.criterion_d(pred_real, pred_fake)

                pixel_loss = criterion_g(output, hdr)
                g_loss = self.criterion_adversarial(discriminator(torch.cat([im1, im2, im3, output], dim=1)))
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('discriminator_loss', criterion_d.item())
                self.valid_metrics.update('generator_loss', total_loss.item())                
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                display = torch.stack([im1.cpu().data[0], im2.cpu().data[0], im3.cpu().data[0], hdr.cpu().data[0],output.cpu().data[0]], dim=1)
                self.writer.add_image("Sequence", make_grid(display, nrow=5, normalize=True))


#         # add histogram of model parameters to the tensorboard
#         for name, p in self.model.named_parameters():
#             self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
