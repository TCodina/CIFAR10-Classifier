"""
Main class to train, evaluate, and save different models on the CIFAR-10 dataset.
TODO: app_parameters, and hyperparameters would be better passed as parser arguments
TODO: Many of the savings could be done more efficiently by using Writters.
"""

import os
from time import time
from datetime import datetime as dt
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # Just to log fancy training progress bar
import pandas as pd
import numpy as np

from utils.util import set_seed, get_accuracy
from dataset import CIFARDataset


class CIFARTrainingApp:
    def __init__(self, data_train, data_val, app_parameters, hyperparameters, transformations, design):

        # Dataset
        self.data_train = data_train
        self.data_val = data_val

        # App parameters
        self.app_parameters = app_parameters
        self.num_workers = app_parameters['num_workers']
        self.save_path = app_parameters['save_path']
        if app_parameters['seed'] is None:
            self.seed = int(dt.now().timestamp())
        else:
            self.seed = int(app_parameters['seed'])

        # Hyperparameters
        self.hyperparameters = hyperparameters
        self.batch_size = hyperparameters['batch_size']
        self.epochs = hyperparameters['epochs']
        self.model_parameters = hyperparameters['model_parameters']
        self.optimizer_parameters = hyperparameters['optimizer_parameters']
        if 'scheduler_parameters' in hyperparameters.keys():
            self.scheduler_parameters = hyperparameters['scheduler_parameters']
        else:
            self.scheduler_parameters = None

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using {self.device} device')

        # Metrics
        self.metrics_train = {'loss': [], 'acc': []}
        self.metrics_val =  {'loss': [], 'acc': []}
        self.best_accuracy = 0.0
        
        # Initialize design (model, criterion and optimizer)
        self.design = design
        self.init_design()

        # Transformations
        self.transformations = transformations
        self.transformation_train = transformations['train']
        self.transformation_val = transformations['val']

        # Timestamp to identify training runs
        time_str = dt.now().strftime('%Y-%m-%d_%H.%M.%S')  
        self.file_path = f'{self.save_path}_{time_str}'
        
        # To continue training from checkpoint
        self.start_epoch = 0
        self.elapsed_time = timedelta(0)
        
    def main(self, validate_bool=True, save_bool=False):
        """
        Here is where the magic happens. It trains the model for all epochs, stores the training and
        validation metrics, and displays the results.
        """
        print("Loading data...")
        loader_train, loader_val = self.init_dataloader()
        n_train = len(loader_train)
        n_val = len(loader_val)

        print(f"\nStart training with {n_train}/{n_val} batches (trn/val) "
              f"of size {self.batch_size} for {self.epochs} epochs\n")

        # Open and write "header" of the txt file if file does not exist already (in case of checkpoint)
        if save_bool and not os.path.exists(self.file_path + '.txt'):
          with open(self.file_path + '.txt', 'w') as file:
            file.write(str(self.design) + '\n')
            file.write(str(self.transformations) + '\n')
            file.write(str(self.app_parameters) + '\n')
            file.write(str(self.hyperparameters) + '\n')

        start = time() - self.elapsed_time.seconds
        for epoch_ndx in range(1 + self.start_epoch, self.epochs + 1):

            # Train and store metrics for a single epoch
            self.train(epoch_ndx, loader_train)

            # Validation
            if validate_bool:
                loss_val, accuracy_val = self.validate(loader_val, n_val)

                # Save epoch metrics
                self.metrics_val['loss'].append(loss_val)
                self.metrics_val['acc'].append(accuracy_val)

                # Log metrics
                print(f'Val Loss: {loss_val:.3f} Val Acc: {(100 * accuracy_val):.2f}%', end="")

                # Check for best accuracy and updates it
                if accuracy_val > self.best_accuracy:
                  self.best_accuracy = accuracy_val
                  print(' (Best)', end="")

                # Log elapsed time
                end = time()
                self.elapsed_time = timedelta(seconds=int(end-start))
                print(f' ({str(self.elapsed_time)})\n')

                if save_bool:
                  # Save metric to txt file
                  with open(self.file_path + '.txt', 'a') as file:
                    file.write(f'\nEpoch {epoch_ndx}/{self.epochs} Val Loss: {loss_val:.3f} Val Acc {(100 * accuracy_val):.2f}% ({str(self.elapsed_time)})\n')
                  # Save model's state
                  self.save_model(epoch_ndx, stamp='')
                  # If best performance achieved, update best-model file
                  if accuracy_val == self.best_accuracy:
                    self.save_model(epoch_ndx, stamp='_best')
                    
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                print(f'lr: {self.scheduler.get_last_lr()[0]}')

        # Compute accuracy over entire training set
        loss_train, accuracy_train = self.validate(loader_train, n_train)
        # Log training metrics
        print(f'Train Loss: {loss_train:.3f} Train Acc: {(100 * accuracy_train):.2f}%', end="")

        # Log and save best accuracy after all training epochs
        if validate_bool:
          print(f'\nBest accuracy: {(100 * self.best_accuracy):.2f}%')
          
          if save_bool:
            with open(self.file_path + '.txt', 'a') as file:
              file.write(f'\nTrain Loss: {loss_train:.3f} Train Acc {(100 * accuracy_train):.2f}%\n')
              file.write(f'\nBest accuracy: {(100 * self.best_accuracy):.2f}%')

    def init_design(self):

        set_seed(self.seed)

        self.model = self.design['model'](**self.model_parameters).to(self.device)

        self.criterion = self.design['criterion']()

        self.optimizer = self.design['optimizer'](self.model.parameters(), **self.optimizer_parameters)
        
        if 'scheduler' in self.design.keys():
            self.scheduler = self.design['scheduler'](self.optimizer, **self.scheduler_parameters)
        else:
            self.scheduler = None

    def init_dataloader(self):

        dataset_train = CIFARDataset(self.data_train, transform=self.transformation_train)
        dataset_val = CIFARDataset(self.data_val, transform=self.transformation_val)

        loader_train = DataLoader(dataset_train,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  shuffle=True)
        loader_val = DataLoader(dataset_val,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True)

        return loader_train, loader_val

    def train(self, epoch_ndx, trainloader):

        # To log fancy progress bar
        loop = tqdm(enumerate(trainloader, start=1), total=len(trainloader))	
        loop.set_description(f"Epoch {epoch_ndx}/{self.epochs}")

        self.model.train()
        for i, (x, y) in loop:

            # Send to device
            x = x.to(self.device)
            y = y.to(self.device)
            # Set gradients to zero
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.model(x)
            # Compute loss
            loss = self.criterion(y_pred, y.squeeze())
            # Backward pass
            loss.backward()
            # Optimize
            self.optimizer.step()
            # Compute statistics
            loss_train = loss.item()
            accuracy_train = get_accuracy(y_pred, y)
            # Save batch metrics
            self.metrics_train['loss'].append(loss_train)
            self.metrics_train['acc'].append(accuracy_train)

            # Update progress bar
            loop.set_postfix_str(f"Loss = {loss_train:.3f}, Acc = {(100 * accuracy_train):.2f}%")

        print('')

    def validate(self, loader, n_loader):

        # Initialize running loss and accuracy
        loss = 0.0
        accuracy = 0.0

        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                # Send to device
                x = x.to(self.device)
                y = y.to(self.device)
                # Forward, and compute loss
                y_pred = self.model(x)
                loss_batch = self.criterion(y_pred, y.squeeze())
                # Update statistics
                loss += loss_batch.item()
                accuracy += get_accuracy(y_pred, y)

            loss /= n_loader
            accuracy /= n_loader

        return loss, accuracy

    def get_metrics(self):
      """
      Returns metrics as Pandas DataFrames ready to be plotted
      """

      metrics_train_df = pd.DataFrame.from_dict(self.metrics_train)
      metrics_val_df = pd.DataFrame.from_dict(self.metrics_val)
      # Change indices to plot training and validation metrics in same plot
      step =  len(metrics_val_df) / len(metrics_train_df)
      metrics_train_df.index = np.arange(step, len(metrics_val_df) + step, step)
      metrics_val_df.index = np.arange(1, len(metrics_val_df) + 1) 

      return metrics_train_df, metrics_val_df

    def save_model(self, epoch_ndx, stamp):
        """
        Save model state and hyperparameters
        """
        state = {
            'model_state': self.model.state_dict(),  # Model's state
            'optimizer_step': self.optimizer.state_dict(),
            'app_parameters': self.app_parameters,
            'hyperparameters': self.hyperparameters,
            'design': self.design,
            'transformations': self.transformations,
            'metrics': {'train': self.metrics_train, 'val': self.metrics_val},
            'epoch': epoch_ndx,
            'file_path': self.file_path,
            'elapsed_time': self.elapsed_time,
            'best_accuracy': self.best_accuracy
        }
                      
        if self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()
                      
        torch.save(state, self.file_path + stamp + '.state')


    def load_state(self, path):
        state = torch.load(path)
        # Load model, optimizer and scheduler states
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_step'])
        if 'scheduler_state' in state.keys() and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler_state'])
        # Load metrics
        self.metrics_train = state['metrics']['train']
        self.metrics_val = state['metrics']['val']
        # To continue training from checkpoint correctly
        self.best_accuracy = state['best_accuracy']
        self.start_epoch = state['epoch']
        self.elapsed_time = state['elapsed_time']
        self.file_path = state['file_path']
        
