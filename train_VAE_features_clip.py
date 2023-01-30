from datetime import datetime
import enum
from statistics import mean
from turtle import color
from wsgiref import validate
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks
import wandb
import matplotlib.pyplot as plt
from  sklearn.manifold import TSNE
#from models.VAE import Encoder, Decoder, VAE
import pickle

# global variables among training functions
training_iterations = 0
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)


def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    # this is needed for multi-GPUs systems where you just want to use a predefined set of GPUs
    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    # wanbd logging configuration
    
    if args.wandb_name is not None:
        wandb.login(key='c87fa53083814af2a9d0ed46e5a562b9a5f8b3ec')
        wandb.init(project="test-project", entity="egovision-aml22")
        #wandb.run.name = args.name + "_" + args.shift.split("-")[0] + "_" + args.shift.split("-")[-1]
        wandb.run.name = f'{args.name}_{args.models.RGB.model}'



def main():
    global training_iterations, modalities
    init_operations()
    modalities = args.modality

    # recover valid paths, domains, classes
    # this will output the domain conversion (D1 -> 8, et cetera) and the label list
    num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
    # device where everything is run
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # these dictionaries are for more multi-modal training/testing, each key is a modality used
    models = {}
    logger.info("Instantiating models per modality")
    for m in modalities:
        logger.info('{} Net\tModality: {}'.format(args.models[m].model, m))
        # notice that here, the first parameter passed is the input dimension
        # In our case it represents the feature dimensionality which is equivalent to 1024 for I3D
        #print(getattr(model_list, args.models[m].model)())
        models[m] = getattr(model_list, args.models[m].model)(1024, 512, 1024)

    # the models are wrapped into the ActionRecognition task which manages all the training steps
    # action_classifier = tasks.ActionRecognition("action-classifier", models, args.batch_size,
    #                                             args.total_batch, args.models_dir, num_classes,
    #   
    if args.action == "train":
        # resume_from argument is adopted in case of restoring from a checkpoint
        # if args.resume_from is not None:
        #     action_classifier.load_last_model(args.resume_from)
        # i.e. number of batches passed
        # notice, here it is multiplied by tot_batch/batch_size since gradient accumulation technique is adopted
        # training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
        # all dataloaders are generated here
        train_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                     'val', args.dataset, None, None, None,
                                                                     None, load_feat=True),
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
       # train(action_classifier, train_loader, val_loader, device, num_classes)
        ae = train(models, train_loader, device)
        # plot_latent(ae, train_loader, device)
        reconstruct(ae, train_loader, device)

def reconstruct(autoencoder, datalaoder, device):
    features = []
    for m in modalities:
        autoencoder[m].load_on(device)
    with torch.no_grad():
        for i, (data, label) in enumerate(datalaoder):
            for m in modalities:
                data[m] = data[m].permute(1,0,2)
            clips = []
            for i_c in range(args.test.num_clips):
                for m in modalities:
                    clip = data[m][i_c].to(device)
                    x_hat = autoencoder[m](clip)
                    clips.append(x_hat)
            clips = torch.stack(clips, dim=0)
            features.append(clips)
            print(features.shape)
        with open("reconstructed_features.pkl", "wb") as file:
            pickle.dump(features, file)

def train(autoencoder, train_dataloader, device, epochs=200):
    for m in modalities:
        autoencoder[m].load_on(device)
    opt = torch.optim.SGD(autoencoder['RGB'].parameters(), lr=0.0001, weight_decay=10e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=10e-4)
    losses = []
    autoencoder['RGB'].train(True)
    for epoch in range(epochs):
        for i, (data, label) in enumerate(train_dataloader):
            for m in modalities:
                data[m] = data[m].permute(1, 0, 2)
                # print(f"Data after permutation: {data[m].size()}")
            
            for i_c in range(args.test.num_clips):
                for m in modalities:
                    opt.zero_grad()
                    # extract the clip related to the modality
                    clip = data[m][i_c].to(device)
                    x_hat = autoencoder[m](clip)
                   # print(f"From autoencoder: {x_hat.size()}")
                    loss = ((clip - x_hat)**2).sum() + autoencoder[m].encoder.kl
                    losses.append(((clip - x_hat)**2).sum().detach().numpy())
                    loss.backward()
                    opt.step()
        scheduler.step()
        losses = np.mean(np.array(losses))
        wandb.log({"Reconstruction loss": losses, 'lr': scheduler.get_last_lr()[0]})
        losses = []

    return autoencoder

def plot_latent(autoencoder, dataloader, device, num_batches=100, loaded = False):
    if not loaded:
        output = []
        labels = []
        with torch.no_grad():
            for i, (data, label) in enumerate(dataloader):
                for m in modalities:
                    data[m] = data[m].permute(1, 0, 2)
                    for i_c in range(args.test.num_clips):
                        clip = data[m][i_c].to(device)
                        z = autoencoder[m].encoder(clip)
                        z = z.to('cpu').detach()
                        for j in range(len(label)):
                            labels.append(label[j])
                        output.append(z) 
        print(f"Len of output: {len(output)}")
        reconstruced_features = torch.stack(tuple(output), dim=0)
        print(f"Once stacked: {reconstruced_features.shape}")

        reconstruced_features = reconstruced_features.reshape(-1, 512)
        print(f"After reshape: {reconstruced_features.shape}")
        print(f'labels {len(labels)}')

        reduced = TSNE().fit_transform(reconstruced_features)
                # if i > num_batches:
                #     plt.colorbar()
                #     break
        #plt.show()
        # filtered = {}
        x_l = reduced[:, 0]
        y_l = reduced[:, 1]
        import pickle
        with open("./latent.pkl", "wb") as file:
            pickle.dump({'x': x_l, 'y': y_l, 'labels': labels}, file)
    else:
        import pandas as pd
        diz = pd.read_pickle("reconstructed_features.pkl")

    # colors= ['green', 'red', 'yellow', 'grey', 'green', 'blu', 'black', 'purple']
    # # for x, y, l in zip(x_l, y_l, labels):
    # #     print(colors[l])
    # plt.scatter(x_l, y_l, c=colors, label=labels)    
    # plt.legend()
    # plt.savefig("./img_VAE.png")
    # plt.show()

if __name__ == '__main__':
    main()
