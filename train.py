import logging
import os
import albumentations
import albumentations.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from network import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,roc_auc_score
from load_data import *
from utils import *


def train_net(model, device, epochs=100, batch_size=2, lr=0.0001, save_cp=True):
    train_transform = albumentations.Compose(
        [albumentations.Resize(256, 256), albumentations.pytorch.ToTensor()])

    # dataset = MammoDataset(train=True,transform=train_transform)
    dataset = UltrasoundDataset(train=True, transform=train_transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Checkpoints:     {save_cp}
            Device:          {device.type}
        ''')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        y_preds = np.array([])
        labels = np.array([])
        y_probs = np.array([])

        acc = 0
        epoch_loss = 0

        for batch_index, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)

            output = model(image)
            prob = torch.softmax(output, 1)[:, 1].detach().cpu().numpy()

            _, preds = torch.max(output.data, 1)
            y_preds = np.hstack([y_preds, preds.cpu().numpy()])
            labels = np.hstack([labels, label.cpu().numpy()])
            y_probs = np.hstack([y_probs, prob])

            acc += torch.sum(preds == label.data)

            loss = criterion(output, label)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_index % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_index * len(image),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_index / len(train_loader),
                                                                               loss.item()))
        auc_score = roc_auc_score(labels, y_probs)
        tn, fp, fn, tp = confusion_matrix(labels, y_preds).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        dir_checkpoint = 'Save path'
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        print(
            'Epoch result - Loss : {:.4f}, AUC : {:.4f}, Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}, learning rate : {:.8f}'.format(
                epoch_loss / len(train_loader), auc_score, acc.item() / len(train_loader.dataset),
                sensitivity,
                specificity, get_lr(optimizer)))

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args,_ = get_args()
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.cuda.set_device(device)
    logging.info(f'Using device {device}')

    model = ConvNet().to(device)

    train_net(model=model, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, device=device)