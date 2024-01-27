import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from tqdm import tqdm
from model_classifier.loader import load_data
from model_classifier.transform import transform_image
from model_classifier.model_trainer import ModelTrainer
import os
import random
import numpy as np

## setting seeds
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## setting device

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_dir = '/content/drive/MyDrive/CarClassificationProject/image_full'
path_train = os.path.join(image_dir, 'Train') # set path to train folder
path_test = os.path.join(image_dir, 'Test') # set path to test folder
path_save = '/content/drive/MyDrive/CarClassificationProject/models' # set path to save models


# setting model parameters
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32

# create dataloaders

train_dl, test_dl, num_classes, class_to_idx = load_data(path_train, path_test,
                                               transform_image(), BATCH_SIZE)

# loading the EfficientNet model parameters
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
model = torchvision.models.efficientnet_b1(weights=weights).to(device)


auto_transforms = weights.transforms()

for param in model.features.parameters():
    param.requires_grad = False

# replace the last layer of the model with a new, untrained layer that has the same number of output units as the number of classes
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=num_classes, # same number of output units as the number of classes
                    bias=True)).to(device)


# Defining loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Setup training and save the results
Model_v0 = ModelTrainer(model=model,
                      train_dataloader=train_dl,
                      test_dataloader=test_dl,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      epochs=100,
                      device=device)
results = Model_v0.train()

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")