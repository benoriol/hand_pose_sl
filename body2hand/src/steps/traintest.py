
import torch
import torch.nn as nn

import numpy as np
import time

from .utils import AverageMeter, ProgressSaver, print_epoch


def train(model, train_loader, val_loader, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)

    progress_saver = ProgressSaver(args.exp)
    start_time = time.time()

    best_epoch, best_val_loss = 0, np.inf

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    train_loss_meter = AverageMeter()


    for n in range(args.num_epochs):

        train_loss_meter.reset()

        for batch in train_loader:

            prediction = model(batch["body_kp"].to(device))

            loss = criterion(prediction, batch["right_hand_kp"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        validation_loss = validate(model, val_loader, criterion)
        total_time = time.time() - start_time

        progress_saver.update_epoch_progess(n, train_loss_meter.get_average(),
                                            validation_loss, best_epoch,
                                            best_val_loss, total_time)

        print_epoch(n, train_loss_meter.get_average(),
                    validation_loss, total_time)




def validate(model, val_loader, criterion):

    with torch.no_grad():

        loss_average = AverageMeter()

        for batch in val_loader:
            prediction = model(batch["body_kp"])
            loss = criterion(prediction, batch["right_hand_kp"])
            loss_average.update(loss.item())

        return loss_average.get_average()












