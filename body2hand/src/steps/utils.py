import pickle


class AverageMeter():
    def __init__(self):
        self.total = 0
        self.count = 0

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def get_average(self):
        return  self.total / self.count

class ProgressSaver():

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.progress = {
            "epoch":[],
            "train_loss":[],
            "val_loss":[],
            "time":[],
            "best_epoch":[],
            "best_val_loss":[]
            }

    def update_epoch_progess(self, epoch, train_loss, val_loss, best_epoch, best_val_loss, time):
        self.progress["epoch"].append(epoch)
        self.progress["train_loss"].append(train_loss)
        self.progress["val_loss"].append(val_loss)
        self.progress["time"].append(time)
        self.progress["best_epoch"].append(best_epoch)
        self.progress["best_val_loss"].append(best_val_loss)

        with open("%s/progress.pkl" % self.exp_dir, "wb") as f:
            pickle.dump(self.progress, f)


def print_epoch(epoch, train_loss, val_loss, time):

    print("Epoch #" + str(epoch) + ":\tTrain loss " + str(train_loss) +
          "\tValid loss: " + str(val_loss) + "\tTime: " + str(time))
