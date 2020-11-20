
import torch
import torch.nn as nn

import numpy as np
import time
import json
import h5py

from .utils import AverageMeter, ProgressSaver, print_epoch, MSE2Pixels, adjust_learning_rate, \
    mask_output, array2open_pose, array2open_pose_index, array2open_pose_3finger, \
    L12Pixels, maskedPoseL1, poderatedPoseL1

def train(model, train_loader, val_loader, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    progress_saver = ProgressSaver(args.exp)

    if args.predict == "right_index":
        loss_interpreter = L12Pixels(4, 1280)

    elif args.predict == "right_3fingers":
        loss_interpreter = L12Pixels(12, 1280)

    elif args.predict == "right_hand":
        loss_interpreter = L12Pixels(21, 1280)

    else:
        print(args.predict)
        raise ValueError

    best_epoch, best_val_loss = 0, np.inf
    lr = args.lr
    if args.loss == "MSE":
        criterion = nn.MSELoss()
    elif args.loss == "L1":
        #criterion = nn.L1Loss()
        criterion = maskedPoseL1()
    elif args.loss == "confL1":
        criterion = poderatedPoseL1()
    elif args.loss == "huber":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss_meter = AverageMeter()

    n_iter = 0
    init_epoch = 0
    start_time = time.time()

    if args.resume:
        progress_saver.load_progress()
        init_epoch, best_val_loss, start_time = progress_saver.get_resume_stats()

        start_time = time.time() - start_time

        model.load_state_dict(torch.load(args.exp + "/models/last_model.pth"))


        optimizer.load_state_dict(
            torch.load(args.exp + "/models/last_optim.pth"))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % init_epoch)

        init_epoch += 1

    model.to(device)



    for n in range(init_epoch, args.num_epochs):
        model.train()
        train_loss_meter.reset()

        if args.lr_decay > 0:
            lr = adjust_learning_rate(args.lr, args.lr_decay, optimizer, n)

        t = time.time()
        for batch in train_loader:
            #print("Start loader loop")
            batch["input_kp"] = batch["input_kp"].to(device)
            batch["target_kp"] = batch["target_kp"].to(device)
            input_lengths = batch["n_frames"]
            t = time.time()
            if args.model == "Conv":
                prediction = model(batch["body_kp"])

            elif args.model == "TransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)

            elif args.model == "ConvTransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)

            elif args.model == "TransformerEnc":
                prediction = model(batch["body_kp"])

            elif args.model == "TextPoseTransformer":
                batch["text_tokens"] = batch["text_tokens"].to(device)
                prediction = model(batch["text_tokens"], batch["input_kp"])
            else:
                raise ValueError()

            prediction = mask_output(prediction, input_lengths)
            #batch["right_hand_kp"] = mask_output(batch["right_hand_kp"], input_lengths)

            if args.loss == "L1":
                loss = criterion(prediction, batch["target_kp"], input_lengths)
            elif args.loss == "confL1":
                loss = criterion(prediction, batch["target_kp"], input_lengths, batch["target_conf"])
            #loss = criterion(prediction, batch["right_hand_kp"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

            if n_iter % args.print_every == 0:
                print("iteration: " + str(n_iter))
                # os.system("nvidia-smi")
            n_iter += 1

            t = time.time()

            # if n == 17:
            #     quit()


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

    model.eval()

    with torch.no_grad():

        loss_average = AverageMeter()

        for batch in val_loader:

            batch["input_kp"] = batch["input_kp"].to(device)
            batch["target_kp"] = batch["target_kp"].to(device)
            input_lengths = batch["n_frames"]

            if args.model == "Conv":
                prediction = model(batch["body_kp"])
            elif args.model == "TransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)

            elif args.model == "ConvTransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)

            elif args.model == "TransformerEnc":
                prediction = model(batch["body_kp"])

            elif args.model == "TextPoseTransformer":
                batch["text_tokens"] = batch["text_tokens"].to(device)
                prediction = model(batch["text_tokens"], batch["input_kp"])

            else:
                raise ValueError()

            prediction = mask_output(prediction, input_lengths)
            # batch["right_hand_kp"] = mask_output(batch["right_hand_kp"],
            #                                      input_lengths)

            if args.loss == "L1":
                loss = criterion(prediction, batch["target_kp"], input_lengths)
            elif args.loss == "confL1":
                loss = criterion(prediction, batch["target_kp"], input_lengths, batch["target_conf"])
            #loss = criterion(prediction, batch["right_hand_kp"])
            loss_average.update(loss.item())

        return loss_average.get_average()


def infer_utterance(model, loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.loss == "MSE":
        criterion = nn.MSELoss()
    elif args.loss == "L1":
        # criterion = nn.L1Loss()
        criterion = maskedPoseL1()
    elif args.loss == "confL1":
        criterion = poderatedPoseL1()
    elif args.loss == "huber":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError()
    # loss_interpreter = MSE2Pixels(21, 1280)

    model.eval()
    with torch.no_grad():

        for batch in loader:
            batch["input_kp"] = batch["input_kp"].to(device)
            batch["target_kp"] = batch["target_kp"].to(device)
            input_lengths = batch["n_frames"]
            if args.model == "Conv":
                prediction = model(batch["body_kp"])
            elif args.model == "TransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)
            elif args.model == "ConvTransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)
            elif args.model == "TransformerEnc":
                prediction = model(batch["body_kp"])

            elif args.model == "TextPoseTransformer":
                batch["text_tokens"] = batch["text_tokens"].to(device)
                prediction = model(batch["text_tokens"], batch["input_kp"])

            else:
                raise ValueError()

            # prediction = mask_output(prediction, input_lengths)
            # batch["right_hand_kp"] = mask_output(batch["right_hand_kp"],
            #                                      input_lengths)

            if args.loss == "L1":
                loss = criterion(prediction, batch["target_kp"], input_lengths)
            elif args.loss == "confL1":
                loss = criterion(prediction, batch["target_kp"], input_lengths, batch["target_conf"])

            # pix_dist=loss_interpreter(loss)
            # print("Loss:", loss.item, "\tPix distance:", pix_dist)

        utt_id = args.data.replace(".json", "")

        prediction = prediction[0]
        # Scale back to the image size.
        # TODO: Don't hardcode upsample factor
        if args.normalize:
            prediction *= 1280

        prediction = prediction.cpu().numpy()
        for i, json_path in enumerate(batch["json_paths"][0]):
            # For now, if is loaded directly from the big json, we don't have access to the
            # specific json file, so just make up one.
            # add wrist position in case it has been train differentially to
            # this keypoint
            json_path = json_path
            json_data = json.load(open(json_path))

            if args.predict == "right_index":
                json_data["people"][0]["hand_right_keypoints_2d"] = \
                    array2open_pose_index(json_data["people"][0]["hand_right_keypoints_2d"],
                                          prediction[i])

            elif args.predict == "right_3fingers":
                json_data["people"][0]["hand_right_keypoints_2d"] = \
                    array2open_pose_3finger(
                        json_data["people"][0]["hand_right_keypoints_2d"],
                        prediction[i])
            elif args.predict == "right_hand":
                json_data["people"][0]["hand_right_keypoints_2d"] = array2open_pose(prediction[i])



            json_id = json_path.split("/")[-1]

            with open(args.output_folder + "/" + json_id, "w") as f:
                json.dump(json_data, f)

def order_and_reshape_toh5(frame_prediction, conf=None):

    if conf is not None:
        raise NotImplementedError()
    n_frames, n_joints, dim = frame_prediction.shape

    frame_prediction = frame_prediction.detach().cpu().numpy()
    # print(frame_prediction[0])
    frame_prediction = np.pad(frame_prediction, ((0, 0), (0, 0), (0, 1)))
    # print(frame_prediction[0])
    frame_prediction = frame_prediction.transpose(0, 2, 1)
    # print(frame_prediction[0])
    frame_prediction = frame_prediction.reshape(n_frames, -1)
    # print(frame_prediction[0])
    # quit()
    return frame_prediction

def prediction_to_h5(prediction, batch, h5_file):
    for i, utt_id in enumerate(batch["utt_id"]):
        right_hand_prediction = order_and_reshape_toh5(prediction[i])
        # right_hand_prediction = order_and_reshape_toh5(batch["right_hand_kp"][i])
        left_hand = order_and_reshape_toh5(batch["left_hand_kp"][i])
        body = order_and_reshape_toh5(batch["body_kp"][i])

        combined = np.concatenate((body, left_hand, right_hand_prediction), axis=1)
        print(body.shape, left_hand.shape, right_hand_prediction.shape, combined.shape)
        quit()
        h5_file.create_dataset(utt_id, data=combined, dtype="float32")


def infer_utterance_h5(model, loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.loss == "MSE":
        criterion = nn.MSELoss()
    elif args.loss == "L1":
        # criterion = nn.L1Loss()
        criterion = maskedPoseL1()
    elif args.loss == "confL1":
        criterion = poderatedPoseL1()
    elif args.loss == "huber":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError()
    # loss_interpreter = MSE2Pixels(21, 1280)

    out_file = h5py.File(args.out_h5data, "w")

    model.eval()
    with torch.no_grad():

        for batch in loader:
            batch["input_kp"] = batch["input_kp"].to(device)
            batch["target_kp"] = batch["target_kp"].to(device)
            input_lengths = batch["n_frames"]
            if args.model == "Conv":
                prediction = model(batch["body_kp"])
            elif args.model == "TransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)
            elif args.model == "ConvTransformerEncoder":
                prediction = model(batch["body_kp"], input_lengths)
            elif args.model == "TransformerEnc":
                prediction = model(batch["body_kp"])

            elif args.model == "TextPoseTransformer":
                batch["text_tokens"] = batch["text_tokens"].to(device)
                prediction = model(batch["text_tokens"], batch["input_kp"])

            else:
                raise ValueError()

            # prediction = mask_output(prediction, input_lengths)
            # batch["right_hand_kp"] = mask_output(batch["right_hand_kp"],
            #                                      input_lengths)

            if args.loss == "L1":
                loss = criterion(prediction, batch["target_kp"], input_lengths)
            elif args.loss == "confL1":
                loss = criterion(prediction, batch["target_kp"], input_lengths, batch["target_conf"])

            # pix_dist=loss_interpreter(loss)
            # print("Loss:", loss.item, "\tPix distance:", pix_dist)

            # Scale back to the image size.
            # TODO: Don't hardcode upsample factor
            if args.normalize:
                prediction *= 1280


            prediction_to_h5(prediction, batch, out_file)


