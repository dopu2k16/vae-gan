import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
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


class VAEIgnite(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAEIgnite().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def kld_loss(x_pred, x, mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


bce_loss = nn.BCELoss(reduction='sum')


def process_function(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, _ = batch
    x = x.to(device)
    x = x.view(-1, 784)
    x_pred, mu, logvar = model(x)
    BCE = bce_loss(x_pred, x)
    KLD = kld_loss(x_pred, x, mu, logvar)
    loss = BCE + KLD
    loss.backward()
    optimizer.step()
    return loss.item(), BCE.item(), KLD.item()


def evaluate_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, _ = batch
        x = x.to(device)
        x = x.view(-1, 784)
        x_pred, mu, logvar = model(x)
        kwargs = {'mu': mu, 'logvar': logvar}
        return x_pred, x, kwargs


trainer = Engine(process_function)
evaluator = Engine(evaluate_function)
training_history = {'bce': [], 'kld': [], 'mse': []}
validation_history = {'bce': [], 'kld': [], 'mse': []}
RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'bce')
RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'kld')
MeanSquaredError(output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'mse')
Loss(bce_loss, output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'bce')
Loss(kld_loss).attach(evaluator, 'kld')


@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    avg_loss = engine.state.metrics['loss']
    avg_bce = engine.state.metrics['bce']
    avg_kld = engine.state.metrics['kld']
    print("Trainer Results - Epoch {} - Avg loss: {:.2f} Avg bce: {:.2f} Avg kld: {:.2f}"
          .format(engine.state.epoch, avg_loss, avg_bce, avg_kld))


def print_logs(engine, dataloader, mode, history_dict):
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    avg_mse = metrics['mse']
    avg_bce = metrics['bce']
    avg_kld = metrics['kld']
    avg_loss =  avg_bce + avg_kld
    print(
        mode + " Results - Epoch {} - Avg mse: {:.2f} Avg loss: {:.2f} Avg bce: {:.2f} Avg kld: {:.2f}"
        .format(engine.state.epoch, avg_mse, avg_loss, avg_bce, avg_kld))
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
plt.plot(range(20), training_history['bce'], 'blue', label='training')
plt.plot(range(20), validation_history['bce'], 'orange', label='validation')
plt.xlim(0, 20);
plt.xlabel('Epoch')
plt.ylabel('BCE')
plt.title('Binary Cross Entropy on Training/Validation Set')
plt.legend();

plt.plot(range(20), training_history['kld'], 'blue', label='training')
plt.plot(range(20), validation_history['kld'], 'orange', label='validation')
plt.xlim(0, 20);
plt.xlabel('Epoch')
plt.ylabel('KLD')
plt.title('KL Divergence on Training/Validation Set')
plt.legend();

plt.plot(range(20), training_history['mse'], 'blue', label='training')
plt.plot(range(20), validation_history['mse'], 'orange', label='validation')
plt.xlim(0, 20);
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Mean Squared Error on Training/Validation Set')
plt.legend();
