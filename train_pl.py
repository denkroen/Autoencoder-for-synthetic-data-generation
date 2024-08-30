import os
import torch
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

import config_files.config_autoencoder_mbientlab as config

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

from tcnn_module import TCNN_module
from LARA_data_module import LARADataModule
from tcnn_autoencoder import TCNN_Autoencoder_module

import numpy as np


torch.set_float32_matmul_precision("medium") # to make lightning happy

if __name__ == "__main__":
    #seeding
    seed_everything(42, workers=True)
    #torch.autograd.set_detect_anomaly(True)

    DATA_FOLDER = "/data/dkroen/"
    MONITOR_LOSS = "validation_loss"

    #checkpoints
    CHECKPOINT_PATH = DATA_FOLDER + "pl_results/" + config.DATASET_NAME + "/" + config.EXPERIMENT_TYPE + "/"
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor=MONITOR_LOSS,
        mode="min",
        dirpath=CHECKPOINT_PATH,
        filename="checkpoint-Autoencoder-{epoch:02d}-{validation_loss:.2f}",
    )    


    TENSORBOARD_PATH = DATA_FOLDER + "logs/" + config.DATASET_NAME + "/" + config.EXPERIMENT_TYPE + "/"
    TENSORBOARD_FILE = config.EXPERIMENT_TYPE
    logger = TensorBoardLogger(TENSORBOARD_PATH + "tb_logs", name= TENSORBOARD_FILE + "-logs")



    if config.EXPERIMENT_TYPE == "tcnn":

        model = TCNN_module(
            learning_rate=config.LEARNING_RATE,
            num_filters=config.NUM_FILTERS,
            filter_size=config.FILTER_SIZE,
            mode=config.MODE,
            num_attributes=config.NUM_ATTRIBUTES,
            num_classes=config.NUM_CLASSES,
            window_length=config.WINDOW_LENGTH,
            sensor_channels=config.NUM_SENSORS,
            path_attributes=config.PATH_ATTRIBUTES
        )

    elif config.EXPERIMENT_TYPE == "autoencoder":

        if config.GENERATE_SYNTHETIC:
            model = TCNN_Autoencoder_module.load_from_checkpoint(config.CHECKPOINT_OF_GENERATOR)
        else:
            model = TCNN_Autoencoder_module(
                learning_rate=config.LEARNING_RATE,
                num_filters=config.NUM_FILTERS,
                filter_size=config.FILTER_SIZE,
                mode=config.MODE,
                num_attributes=config.NUM_ATTRIBUTES,
                num_classes=config.NUM_CLASSES,
                window_length=config.WINDOW_LENGTH,
                sensor_channels=config.NUM_SENSORS,
                path_attributes=config.PATH_ATTRIBUTES
            )
        
    


    data_module = LARADataModule(
        datadir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        synt_generation=config.GENERATE_SYNTHETIC,
        train_on_synthetic=config.TRAIN_ON_SYNTHETIC,
    )
    
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices= [0],
        logger=logger,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        accumulate_grad_batches=4,
        val_check_interval=config.VAL_BATCHES,
        callbacks=[EarlyStopping(monitor=MONITOR_LOSS),checkpoint_callback],
        deterministic=True
    )

    if config.GENERATE_SYNTHETIC:
        model.enable_synt_generation(config.GENERATE_SYNTHETIC, config.SYNTHETIC_DATA_FOLDER)
        trainer.test(model, data_module)

        #generate csv for synthetic data
        f = []
        for dirpath, dirnames, filenames in os.walk(config.SYNTHETIC_DATA_FOLDER):
            for n in range(len(filenames)):
                f.append(config.SYNTHETIC_DATA_FOLDER + 'seq_' + str(n) +'.pkl')
                print(n)

        np.savetxt(config.DATA_DIR, f, delimiter="\n", fmt='%s')

    else:
        trainer.fit(model, data_module)
        trainer.validate(model, data_module)
        trainer.test(model, data_module)