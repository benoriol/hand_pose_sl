
import torch
import torch.nn as nn

import numpy as np
import time

from .utils import AverageMeter, ProgressSaver, print_epoch, MSE2Pixels, adjust_learning_rate, \
    mask_output

def train(model, train_loader, val_loader, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)

    progress_saver = ProgressSaver(args.exp)
    start_time = time.time()

    loss_interpreter = MSE2Pixels(21, 1280)

    best_epoch, best_val_loss = 0, np.inf
    lr = args.lr

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.Adam(model.parameters())

    train_loss_meter = AverageMeter()

    n_iter = 0


    for n in range(args.num_epochs):

        train_loss_meter.reset()

        if args.lr_decay > 0:
            lr = adjust_learning_rate(args.lr, args.lr_decay, optimizer, n)


        for batch in train_loader:

            #print("Start loader loop")
            batch["body_kp"] = batch["body_kp"].to(device)
            batch["right_hand_kp"] = batch["right_hand_kp"].to(device)
            input_lengths = batch["n_frames"]

            if args.model == "Conv":
                prediction = model(batch["body_kp"])


            elif args.model == "TransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)

            elif args.model == "ConvTransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)

            else:
                raise ValueError()

            prediction = mask_output(prediction, input_lengths)
            batch["right_hand_kp"] = mask_output(batch["right_hand_kp"], input_lengths)

            loss = criterion(prediction, batch["right_hand_kp"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

            if n_iter % args.print_every == 0:
                print("iteration: " + str(n_iter))
                # os.system("nvidia-smi")
            n_iter += 1


        validation_loss = validate(model, val_loader, criterion, device, args)
        total_time = time.time() - start_time

        valid_pix_dist=loss_interpreter(validation_loss)
        train_loss = train_loss_meter.get_average()
        train_pix_dist = loss_interpreter(train_loss)

        if best_val_loss > validation_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), args.exp + "/models/best_model.pth")
        torch.save(model.state_dict(), args.exp + "/models/last_model.pth")
        torch.save(optimizer.state_dict(), args.exp + "/models/last_optim.pth")

        epoch_data = {
            "epoch": n,
            "train_loss": train_loss,
            "train_pix_dist": train_pix_dist,
            "val_loss": validation_loss,
            "val_pix_dist": valid_pix_dist,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "time": total_time,
            "lr": lr

        }

        progress_saver.update_epoch_progess(epoch_data)

        print_epoch(n, train_loss, train_pix_dist,
                    validation_loss, valid_pix_dist, total_time)




def validate(model, val_loader, criterion, device, args):

    with torch.no_grad():

        loss_average = AverageMeter()

        for batch in val_loader:

            batch["body_kp"] = batch["body_kp"].to(device)
            batch["right_hand_kp"] = batch["right_hand_kp"].to(device)
            input_lengths = batch["n_frames"]

            if args.model == "Conv":
                prediction = model(batch["body_kp"])
            elif args.model == "TransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)

            elif args.model == "ConvTransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)

            prediction = mask_output(prediction, input_lengths)
            batch["right_hand_kp"] = mask_output(batch["right_hand_kp"],
                                                 input_lengths)

            loss = criterion(prediction, batch["right_hand_kp"])
            loss_average.update(loss.item())


        return loss_average.get_average()



def infer_utterance(model, loader, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    loss_interpreter = MSE2Pixels(21, 1280)


    for batch in loader:
        batch["body_kp"] = batch["body_kp"].to(device)
        batch["right_hand_kp"] = batch["right_hand_kp"].to(device)
        input_lengths = batch["n_frames"]

        if args.model == "Conv":
            prediction = model(batch["body_kp"])
        elif args.model == "TransformerEncoder":
            prediction = model(batch["body_kp"], input_lengths)

        elif args.model == "ConvTransformerEncoder":
            prediction = model(batch["body_kp"], input_lengths)

        prediction = mask_output(prediction, input_lengths)
        batch["right_hand_kp"] = mask_output(batch["right_hand_kp"],
                                             input_lengths)

        loss = criterion(prediction, batch["right_hand_kp"])

        pix_dist=loss_interpreter(loss)

        print(prediction.shape)

        for i in range(prediction.shape[0]):


