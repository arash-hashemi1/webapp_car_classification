from model_trainer import ModelTrainer
from loader import load_data
from transform import transform_image
import torch
from resnet_classifier import ResNet101
from torch import optim
from torch.optim import lr_scheduler
from torch import nn
import numpy as np
import random
import os

# setting seeds
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# setting device

device = 'cuda' if torch.cuda.is_available() else 'cpu'


image_dir = '/content/drive/MyDrive/CarClassificationProject/image_full'
path_train = os.path.join(image_dir, 'Train') # set path to train folder
path_test = os.path.join(image_dir, 'Test') # set path to test folder
path_save = '/content/drive/MyDrive/CarClassificationProject/models' # set path to save models

batch_size = 128

train_dl, test_dl, num_classes, class_to_idx = load_data(path_train, path_test,
                                           transform_image(), batch_size)

# setting model parameters

hidden_1 = 1024
hidden_2 = 512


model = ResNet101(hidden_1=hidden_1, hidden_2=hidden_2, num_target_classes=num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()

for p in model.feature_extractor.parameters():
  p.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# setting model trainer parameters


n_epochs = 100
step_size = 30
gamma = 0.5
save_interval = 10
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


model_v0 = ModelTrainer(model, train_dl, test_dl, loss_fn,
                        optimizer, scheduler, n_epochs,
                        save_interval, path_save, device)

results = model_v0.train()

# plotting results
import matplotlib.pyplot as plt

train_loss = results["train_loss"]
test_loss = results["test_loss"]
train_acc = results["train_acc"]
test_acc =  results["test_acc"]

fig, axs = plt.subplots(2, 2, layout='constrained')
axs[0, 0].plot(train_loss)
axs[0, 0].set_title('train loss')
axs[0, 1].plot(test_loss)
axs[0, 1].set_title('test loss')
axs[1, 0].plot(train_acc)
axs[1, 0].set_title('train acc')
axs[1, 1].plot(test_acc)
axs[1, 1].set_title('test acc')