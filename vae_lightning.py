from argparse import ArgumentParser
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torchvision
import os
import torch.utils as utils


class VAELightning(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.inp_fc1 = nn.Linear(784, 400)
        self.enc_fc1 = nn.Linear(400, 20)
        self.enc_fc2 = nn.Linear(400, 20)
        self.dec_fc1 = nn.Linear(20, 400)
        self.dec_fc2 = nn.Linear(400, 784)
        self.hparams = hparams

    def encoder(self, x):
        h1 = F.relu(self.inp_fc1(x))
        return self.enc_fc1(h1), self.enc_fc2(h1)

    def decoder(self, z):
        h2 = F.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_fc2(h2))

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, recons_x, x, mu, logvar):
        binary_cross_entropy = F.binary_cross_entropy_with_logits(recons_x, x.view(-1, 784), reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return binary_cross_entropy + kl_divergence

    def forward(self, z):
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, logvar = self.encoder(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        x_hat = self(z)
        loss = self.loss_function(x_hat, x, mu, logvar)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, logvar = self.encoder(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        x_hat = self(z)
        val_loss = self.loss_function(z, x, mu, logvar)
        return {'val_loss': val_loss, 'x_hat': x_hat}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        x_hat = outputs[-1]['x_hat']
        grid_image = torchvision.utils.make_grid(x_hat)
        self.logger.experiment.add_image('images', grid_image, 0)
        logs = {'avg_val_loss': val_loss}
        return {'log': logs, 'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        x, _ = batch
        mu, logvar = self.encoder(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        x_hat = self(z)
        test_loss = self.loss_function(z, x, mu, logvar)
        logs = {'test_loss':test_loss}
        return {'test_loss': test_loss, 'log':logs,'progress_bar': test_loss}

    def test_epoch_end(self,outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        x_hat = outputs[-1]['x_hat']
        grid_image = torchvision.utils.make_grid(x_hat)
        self.logger.experiment.add_image('images', grid_image, 0)
        logs ={'avg_test_loss': test_loss}
        return {'log': logs, 'progress_bar': logs, 'test_loss': test_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(os.getcwd(), train=False, download=True, transform=transform)

        # train/val split
        mnist_train, mnist_val = utils.data.random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def save_image(self, data, filename):
        img = data.clone().clamp(0, 255).numpy()
        img = img[0].transpose(1, 2, 0)
        img = Image.fromarray(img, mode='RGB')
        img.save(filename)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size= self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()

    vae = VAELightning(hparams=args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(vae)








