import json
from tqdm import tqdm
from . import utils
def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)




class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}


def train_model(model, train_dl,val_dl, epochs, loss_G,loss_D,display_every=250):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
        loss_G.append(loss_meter_dict['loss_G'])
        loss_D.append(loss_meter_dict['loss_D'])
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                utils.log_results(loss_meter_dict) # function to print out the losses
                utils.visualize(model, data, save=True) # function displaying the model's outputs
    losses_to_save = {
        "loss_G": [float(x.avg) for x in loss_G],
        "loss_D": [float(x.avg) for x in loss_D],
    }
    # json.dump(losses_to_save, f, indent=4)


    with open("losses.json", "w") as f:
        json.dump(losses_to_save, f, indent=4)

