from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from torchvision.transforms import Compose, ToTensor
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError, Loss, RunningAverage

SEED = 6789
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = Compose([ToTensor()])
train_data = MNIST(train=True, download=True, transform=data_transform, root='../data/')
val_data = MNIST(train=False, download=True, transform=data_transform, root='../data/')
image = train_data[0][0]
label = train_data[0][1]
print('Length of training dataset', len(train_data))
print('Length of validation dataset', len(val_data))
print('image.shape : ', image.shape)
print('label : ', label)

img = plt.imshow(image.squeeze().numpy(), cmap='gray')
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, **kwargs)
val_loader = DataLoader(val_data, shuffle=False, batch_size=32, **kwargs)

for batch in train_loader:
    x, y = batch
    break

print('Shape of X', x.shape)
print('Shape of Y', y.shape)
fixed_images = x.to(device)


class VaeIgnite(nn.Module):

    def __init__(self):
        super().__init__()
        self.inp_fc1 = nn.Linear(784, 400)
        self.enc_fc1 = nn.Linear(400, 20)
        self.enc_fc2 = nn.Linear(400, 20)
        self.dec_fc1 = nn.Linear(20, 400)
        self.dec_fc2 = nn.Linear(400, 784)

    def encoder(self, x):
        h1 = F.relu(self.inp_fc1(x))
        return self.enc_fc1(h1), self.enc_fc2(h1)

    def decoder(self, z):
        h2 = F.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_fc2(h2))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def loss_function(recons_x, x, mu, logvar):
        binary_cross_entropy = F.binary_cross_entropy_with_logits(recons_x, x.view(-1, 784), reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return binary_cross_entropy + kl_divergence

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


model = VaeIgnite().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def process_function(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, _ = batch
    x = x.to(device)
    x_pred, mu, logvar = model(x)
    loss = VaeIgnite.loss_function(x_pred, x, mu, logvar)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, _ = batch
        x = x.to(device)
        x_pred, mu, logvar = model(x)
        kwargs = {'mu': mu, 'logvar': logvar}
        return x_pred, x, kwargs


trainer = Engine(process_function)
evaluator = Engine(evaluate_function)
training_history = {'mse': []}
validation_history = {'mse': []}

RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')

MeanSquaredError(output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'mse')


@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    avg_loss = engine.state.metrics['loss']
    print("Trainer Results - Epoch {} - Avg loss: {:.2f}"
          .format(engine.state.epoch, avg_loss))


def print_logs(engine, dataloader, mode, history_dict):
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    avg_mse = metrics['mse']
    print(mode + " Results - Epoch {} - Avg mse: {:.2f}"
          .format(engine.state.epoch, avg_mse))

    for key in evaluator.state.metrics.keys():
        history_dict[key].append(evaluator.state.metrics[key])


trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, train_loader, 'Training', training_history)
trainer.add_event_handler(Events.EPOCH_COMPLETED, print_logs, val_loader, 'Validation', validation_history)


def compare_images(engine, save_img=False):
    epoch = engine.state.epoch
    reconstructed_images = model(fixed_images.view(-1, 784))[0].view(-1, 1, 28, 28)
    comparison = torch.cat([fixed_images, reconstructed_images])
    if save_img:
        save_image(comparison.detach().cpu(), 'reconstructed_epoch_' + str(epoch) + '.png', nrow=8)
    comparison_image = make_grid(comparison.detach().cpu(), nrow=8)
    fig = plt.figure(figsize=(5, 5));
    output = plt.imshow(comparison_image.permute(1, 2, 0));
    plt.title('Epoch ' + str(epoch));
    plt.show();


trainer.add_event_handler(Events.STARTED, compare_images, save_img=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), compare_images, save_img=False)

e = trainer.run(train_loader, max_epochs=20)


plt.plot(range(20), training_history['mse'], 'blue', label='training')
plt.plot(range(20), validation_history['mse'], 'orange', label='validation')
plt.xlim(0, 20);
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Mean Squared Error on Training/Validation Set')
plt.legend();
