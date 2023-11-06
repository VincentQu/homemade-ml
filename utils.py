import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# batch_size = 32
# n_folds = 5
# device = 'cpu'

def train_batch(model, X_batch, Y_batch, loss_func, optimizer):
    # Set model to training mode
    model.train()
    # Make prediction
    batch_pred = model(X_batch)
    # Calculate loss
    batch_loss = loss_func(batch_pred, Y_batch)
    # Reset gradients to 0
    optimizer.zero_grad()
    # Calculate gradients
    batch_loss.backward()
    # Backpropagation: adjust parameters
    optimizer.step()

    # Calculate training accuracy
    batch_pred_classes = batch_pred.argmax(1)
    correct = (batch_pred_classes == Y_batch).sum().item()
    total = len(Y_batch)

    return batch_loss.item(), correct, total


def train_epoch(model, dataloader, loss_func, optimizer, device):
    # Initialise loss, correct, and total values
    loss = correct = total = 0

    # Iterate over batches in dataloader
    for i, (X, y) in enumerate(dataloader):
        # Move batches to device
        X = X.to(device)
        y = y.to(device)
        # Train model on batch
        batch_loss, batch_correct, batch_total = train_batch(model, X, y, loss_func, optimizer)

        loss += batch_loss
        correct += batch_correct
        total += batch_total
        # Divide loss by number of batches (to make it comparable for different batch sizes)
    loss /= i + 1
    accuracy = correct / total

    return loss, accuracy


def eval_batch(model, X_batch, Y_batch, loss_func):
    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        batch_pred = model(X_batch)
        batch_loss = loss_func(batch_pred, Y_batch)

        # Calculate training accuracy
        batch_pred_classes = batch_pred.argmax(1)
        correct = (batch_pred_classes == Y_batch).sum().item()
        total = len(Y_batch)

        return batch_loss.item(), correct, total


def eval_epoch(model, dataloader, loss_func, device):
    # Initialise loss value
    loss = correct = total = 0

    # Iterate over batches in dataloader
    for i, (X, y) in enumerate(dataloader):
        # Move batches to device
        X = X.to(device)
        y = y.to(device)
        # Train model on batch
        batch_loss, batch_correct, batch_total = eval_batch(model, X, y, loss_func)

        loss += batch_loss
        correct += batch_correct
        total += batch_total

    # Divide loss by number of batches (to make it comparable for different batch sizes)
    loss /= i + 1
    accuracy = correct / total
    return loss, accuracy


def cross_validate_model(model_class, dataset, loss_func_class, optimiser_class, batch_size, n_folds,
                         epochs, start_lr, device, debug_print=True):
    # Create folds for cross-validation
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=65)
    m = len(dataset)
    results = pd.DataFrame(columns=['epoch', 'fold', 'dataset', 'metric', 'value'])

    # Loop over folds
    for fold, (train_idx, val_idx) in enumerate(folds.split(range(m))):
        print(f'Fold: {fold}')

        # Create samplers for training and validation dataset
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Create dataloaders for training and validation dataset
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,
                                num_workers=2, pin_memory=True)

        model = model_class()
        model.to(device)

        loss_fn = loss_func_class()
        optim = optimiser_class(model.parameters(), lr=start_lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5) # choice task
        for e in range(epochs):

            train_loss, train_accuracy = train_epoch(model, train_loader, loss_fn, optim, device)
            val_loss, val_accuracy = eval_epoch(model, val_loader, loss_fn, device)

            if debug_print:
                print(f"Epoch {e+1:>2}/{epochs} - Learning rate: {scheduler.get_last_lr()}")
                print(f"{'Training loss:':>18} {train_loss:.4f} - Accuracy: {train_accuracy:.3f}")
                print(f"{'Validation loss:':>18} {val_loss:.4f} - Accuracy: {val_accuracy:.3f}")

            train_result = pd.DataFrame(data={'epoch': e, 'fold': fold, 'dataset': 'training', 'metric': ['loss', 'accuracy'], 'value': [train_loss, train_accuracy]})
            val_result = pd.DataFrame(data={'epoch': e, 'fold': fold, 'dataset': 'validation', 'metric': ['loss', 'accuracy'], 'value': [val_loss, val_accuracy]})

            results = pd.concat([results, train_result, val_result], axis=0)
            scheduler.step()

    # Save model to file
    path = f'models/{model.__class__.__name__}.pt'
    torch.save(model.state_dict(), path)

    return model, results.reset_index()

"""
def train_model(model, train_loader, valid_loader, loss_func, optimiser, batch_size,
                         epochs, device, scheduler=None, debug_print=True):
    results = pd.DataFrame(columns=['epoch', 'dataset', 'metric', 'value'])

    # model = model_class()
    # model.to(device)
    #
    # loss_fn = loss_func_class()
    # optim = optimiser_class(model.parameters(), lr=start_lr)

    for e in range(epochs):

        train_loss, train_accuracy = train_epoch(model, train_loader, loss_func, optimiser, device)
        valid_loss, valid_accuracy = eval_epoch(model, valid_loader, loss_func, device)

        if debug_print:
            current_lr = [p['lr'] for p in optimiser.param_groups]
            print(f"Epoch {e+1:>2}/{epochs} - Learning rate: {current_lr}")
            print(f"{'Training loss:':>18} {train_loss:.4f} - Accuracy: {train_accuracy:.3f}")
            print(f"{'Test loss:':>18} {valid_loss:.4f} - Accuracy: {valid_accuracy:.3f}")

        train_result = pd.DataFrame(data={'epoch': e, 'dataset': 'training', 'metric': ['loss', 'accuracy'], 'value': [train_loss, train_accuracy]})
        test_result = pd.DataFrame(data={'epoch': e, 'dataset': 'test', 'metric': ['loss', 'accuracy'], 'value': [valid_loss, valid_accuracy]})

        results = pd.concat([results, train_result, test_result], axis=0)

        if scheduler is not None:
            scheduler.step(valid_loss)

    #TODO: add save_filepath or so as function parameter

    # Save model to file
    # path = f'models/{model.__class__.__name__}_final.pt'
    # torch.save(model.state_dict(), path)

    return model, results
"""
# """

def train_model(model_class, train_loader, valid_loader, loss_func_class, optimiser_class, batch_size,
                         epochs, start_lr, device, debug_print=True):
    results = pd.DataFrame(columns=['epoch', 'dataset', 'metric', 'value'])

    model = model_class()
    model.to(device)

    loss_fn = loss_func_class()
    optim = optimiser_class(model.parameters(), lr=start_lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    for e in range(epochs):

        train_loss, train_accuracy = train_epoch(model, train_loader, loss_fn, optim, device)
        valid_loss, valid_accuracy = eval_epoch(model, valid_loader, loss_fn, device)

        if debug_print:
            # print(f"Epoch {e+1:>2}/{epochs} - Learning rate: {scheduler.get_last_lr()}")
            print(f"Epoch {e+1:>2}/{epochs}")
            print(f"{'Training loss:':>18} {train_loss:.4f} - Accuracy: {train_accuracy:.3f}")
            print(f"{'Test loss:':>18} {valid_loss:.4f} - Accuracy: {valid_accuracy:.3f}")

        train_result = pd.DataFrame(data={'epoch': e, 'dataset': 'training', 'metric': ['loss', 'accuracy'], 'value': [train_loss, train_accuracy]})
        test_result = pd.DataFrame(data={'epoch': e, 'dataset': 'test', 'metric': ['loss', 'accuracy'], 'value': [valid_loss, valid_accuracy]})

        results = pd.concat([results, train_result, test_result], axis=0)
        # scheduler.step()

    # Save model to file
    # path = f'models/{model.__class__.__name__}_final.pt'
    # torch.save(model.state_dict(), path)

    return model, results
# """


def test_model(model_class, train_loader, test_loader, loss_func_class, optimiser_class, batch_size,
                         epochs, start_lr, device, debug_print=True):
    results = pd.DataFrame(columns=['epoch', 'dataset', 'metric', 'value'])

    model = model_class()
    model.to(device)

    loss_fn = loss_func_class()
    optim = optimiser_class(model.parameters(), lr=start_lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5) # choice task

    for e in range(epochs):

        train_loss, train_accuracy = train_epoch(model, train_loader, loss_fn, optim, device)
        test_loss, test_accuracy = eval_epoch(model, test_loader, loss_fn, device)

        if debug_print:
            print(f"Epoch {e+1:>2}/{epochs} - Learning rate: {scheduler.get_last_lr()}")
            print(f"{'Training loss:':>18} {train_loss:.4f} - Accuracy: {train_accuracy:.3f}")
            print(f"{'Test loss:':>18} {test_loss:.4f} - Accuracy: {test_accuracy:.3f}")

        train_result = pd.DataFrame(data={'epoch': e, 'dataset': 'training', 'metric': ['loss', 'accuracy'], 'value': [train_loss, train_accuracy]})
        test_result = pd.DataFrame(data={'epoch': e, 'dataset': 'test', 'metric': ['loss', 'accuracy'], 'value': [test_loss, test_accuracy]})

        results = pd.concat([results, train_result, test_result], axis=0)
        scheduler.step()

    # Save model to file
    path = f'models/{model.__class__.__name__}_final.pt'
    torch.save(model.state_dict(), path)

    return model, results


def plot_loss_accuracy(results):
    avgs = results.groupby(['epoch', 'dataset', 'metric'])['value'].mean().reset_index()
    plot = sns.relplot(data=avgs, x='epoch', y='value', hue='dataset', col='metric', kind='line', facet_kws={'sharey': False})
    plot.set_titles('{col_name}')
    return plot


def print_loss_accuracy(results):
    avgs = results.groupby(['epoch', 'dataset', 'metric'])['value'].mean().reset_index()
    last_epoch = avgs.loc[avgs.epoch == avgs.epoch.max(), ['dataset', 'metric', 'value']]
    results = last_epoch.to_records(index=False)

    for dataset, metric, score in results:
        print(f"{dataset.capitalize() + ' ' + metric:<20} - {round(score, 4)}")


def plot_confusion_matrix(model, dataloader, labels, device):
    predicted = np.array([])
    correct = np.array([])

    for x, y in dataloader:
        x = x.to(device)

        batch_pred = model(x)
        predicted_class = batch_pred.argmax(1).cpu().numpy()
        predicted = np.append(predicted, predicted_class)
        correct = np.append(correct, y.numpy())

    cm = confusion_matrix(y_true=correct, y_pred=predicted)

    cm_df = pd.DataFrame(cm, columns=labels, index=labels)
    cm_df = cm_df.div(cm_df.sum(axis=1), axis=0)
    cm_df = cm_df.rename_axis(index='True label', columns='Correct label')

    plot = sns.heatmap(cm_df, annot=True)
    return plot
