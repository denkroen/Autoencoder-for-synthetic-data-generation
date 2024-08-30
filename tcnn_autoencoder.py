import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
from nn_tcnn_convolutions import TCNNConvBlock
from nn_tcnn_classificator import ClassificationHead
import numpy as np
import os
import pickle

from tcnn_autoencoder_module import TCNN_Autoencoder

from utils import efficient_distance, compute_feature_map_size_tcnn, reader_att_rep

class TCNN_Autoencoder_module(pl.LightningModule):
    def __init__(self, learning_rate, num_filters, filter_size, mode, num_attributes, num_classes, window_length, sensor_channels, path_attributes):
        super().__init__()
        #TODO: move some functions to utils.py, use nn_tcnn.py as network
        self.lr = learning_rate #def schedule

        latent_size = compute_feature_map_size_tcnn(0,window_length,sensor_channels,filter_size) 

        self.mode = mode

        self.synt_generation = False
        self.synt_data_folder = ""

        if self.mode == "attribute":
            self.loss_classification = nn.BCELoss()
            output_neurons = num_attributes

            # load attribute mapping
            self.attr = reader_att_rep(path_attributes) 
            for attr_idx in range(self.attr.shape[0]):
                self.attr[attr_idx, 1:] = self.attr[attr_idx, 1:] / np.linalg.norm(self.attr[attr_idx, 1:])

            self.atts = torch.from_numpy(self.attr).type(dtype=torch.FloatTensor)
            self.atts = self.atts.type(dtype=torch.cuda.FloatTensor)

        elif self.mode == "classification":
            self.loss_classification = nn.CrossEntropyLoss()
            output_neurons = num_classes


        self.loss_reconstruction = nn.MSELoss()


        self.autoencoder = TCNN_Autoencoder(1,window_length,sensor_channels,filter_size,output_neurons,num_filters)

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes
        )



    def forward(self, x):
        
        feat_rep, pred, reconst = self.autoencoder.forward(x)

        #no classification head
        #if self.mode == "attribute":
        #    pred = F.sigmoid(pred)
        #elif self.mode == "classification":
        #    pred = F.softmax(pred)

        return feat_rep, pred, reconst

    def training_step(self, batch, batch_idx):
        loss, prediction, label = self._common_step(batch, batch_idx)

        train_acc = 0#self.accuracy(prediction, label)
        train_f1 = 0#self.f1(prediction, label)

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": train_acc,
                "train_f1": train_f1
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        

        return loss

    def validation_step(self, batch, batch_idx):
        loss, prediction, label = self._common_step(batch, batch_idx)

        #calc_metrics
        val_acc = 0#self.accuracy(prediction, label)
        val_f1 = 0#self.f1(prediction, label)


        self.log_dict(
            {
                "validation_loss": loss,
                "validation_acc": val_acc,
                "validation_f1": val_f1
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        return loss

    def test_step(self, batch, batch_idx):
        loss, prediction, label = self._common_step(batch, batch_idx)

        test_acc = 0#self.accuracy(prediction, label)
        test_f1 = 0#self.f1(prediction, label)

        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": test_acc,
                "test_f1": test_f1

            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def _common_step(self, batch, batch_idx):
        
        input_tensor = batch["data"]
        label_reconstruction = batch["data"]

        if self.mode == "attribute":
            label_classification = batch["label_attr"][1:]
        else:
            label_classification = batch["label_class"][0]

        label_metrics = batch["label_class"][0]

        feat_rep, pred, reconst= self.forward(input_tensor)



        if self.synt_generation:

            if self.mode == "attribute":
                obj = {"data": reconst , "label": batch["label_class"], "labels": batch["label_attr"]}

            else:
                obj = {"data": reconst , "label": batch["label_class"], "labels": 0}


            f = open(os.path.join(self.synt_data_folder, 'seq_' + str(self.sequence_count) + ".pkl"), 'wb')
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            self.sequence_count = self.sequence_count + 1




        loss = self.loss_classification(pred, label_classification) + self.loss_reconstruction(reconst, label_reconstruction)

        pred2 = pred.detach().clone() #otherwise we have inlines
        if self.mode == "attribute":
            pred2 = efficient_distance(self.attr, self.atts, pred2) #distances
            pred2 = self.atts[torch.argmin(pred2, dim=1), 0] #classes

        return loss, pred2, label_metrics
    
    def enable_synt_generation(self, bool, path):
        self.synt_generation = bool
        self.autoencoder.enable_synt_generation(bool)
        self.synt_data_folder = path
        self.sequence_count = 0
    
    

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=self.lr)