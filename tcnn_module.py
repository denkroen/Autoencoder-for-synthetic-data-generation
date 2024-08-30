import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
from nn_tcnn_convolutions import TCNNConvBlock
from nn_tcnn_classificator import ClassificationHead
import numpy as np
from nn_tcnn import TCNN

from utils import efficient_distance, compute_feature_map_size_tcnn, reader_att_rep

class TCNN_module(pl.LightningModule):
    def __init__(self, learning_rate, num_filters, filter_size, mode, num_attributes, num_classes, window_length, sensor_channels, path_attributes):
        super().__init__()
        #TODO: move some functions to utils.py, use nn_tcnn.py as network
        self.lr = learning_rate #def schedule

        latent_size = compute_feature_map_size_tcnn(0,window_length,sensor_channels,filter_size) 

        self.mode = mode

        if self.mode == "attribute":
            self.loss = nn.BCELoss()
            output_neurons = num_attributes

            # load attribute mapping
            self.attr = reader_att_rep(path_attributes) 
            for attr_idx in range(self.attr.shape[0]):
                self.attr[attr_idx, 1:] = self.attr[attr_idx, 1:] / np.linalg.norm(self.attr[attr_idx, 1:])

            self.atts = torch.from_numpy(self.attr).type(dtype=torch.FloatTensor)
            self.atts = self.atts.type(dtype=torch.cuda.FloatTensor)

        elif self.mode == "classification":
            self.loss = nn.CrossEntropyLoss()
            output_neurons = num_classes


        self.tcnn = TCNN(1,window_length,sensor_channels,filter_size,output_neurons,num_filters)
        #self.conv = TCNNConvBlock(1, num_filters, filter_size)
        #self.classificator = ClassificationHead(latent_size, output_neurons, [128,128])

        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes
        )



    def forward(self, x):
        feat_rep, x, reconst = self.tcnn.forward(x)
        #x = self.conv.forward(x)
        #x = x.view(x.size()[0], x.size()[1], x.size()[2])
        #x = self.classificator.forward(x)

        if self.mode == "attribute":
            x = F.sigmoid(x)
        elif self.mode == "classification":
            
            x = F.softmax(x)

        return feat_rep, x, reconst

    def training_step(self, batch, batch_idx):
        loss, prediction, label = self._common_step(batch, batch_idx)

        train_acc = self.accuracy(prediction, label)
        train_f1 = self.f1(prediction, label)

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
        val_acc = self.accuracy(prediction, label)
        val_f1 = self.f1(prediction, label)


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

        test_acc = self.accuracy(prediction, label)
        test_f1 = self.f1(prediction, label)

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
        label_classification = batch["label_attr"]
        label_metrics = batch["label_class"]

        feat_rep, pred, reconst = self.forward(input_tensor)
        if self.mode == "classification":
            loss = self.loss(pred, label_metrics) 
        elif self.mode == "attribute":
            loss = self.loss(pred, label_classification) 




        pred2 = pred.detach().clone() #otherwise we have inlines

        if self.mode == "attribute":

            pred2 = efficient_distance(self.attr, self.atts, pred2) #distances
            pred2 = self.atts[torch.argmin(pred2, dim=1), 0] #classes

        return loss, pred2, label_metrics
    
    

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=self.lr)