import os

DATASET_NAME = "mbientlab" #name of the dataset
EXPERIMENT_TYPE = "autoencoder" #model to be trained ("autoencoder"/"tcnn")


#synthetic data generation flags

TRAIN_ON_SYNTHETIC = False # if synthetic data should be used for training
GENERATE_SYNTHETIC = False # if synthetic data should be generated with a trained autoencoder
CHECKPOINT_OF_GENERATOR = "" #checkpoint of trained autoencoder that should be used for synthetic data generation
SYNTHETIC_DATA_FOLDER = "" #folder to save synthetic data

# Training hyperparameters
VAL_BATCHES = 100 # after VAL_BATCHES amount of batches in training do a validation step
BATCH_SIZE = 100 # batchsize
NUM_EPOCHS = 10 # epochs
LEARNING_RATE = 0.0001 #learning rate

# Dataset parameters
DATA_DIR = "/data/dkroen/dataset/mbientlab/" #add Path to Folder containing the dataset (root_dir)
NUM_WORKERS = os.cpu_count()-2

WINDOW_LENGTH = 100 # length of windows of the preprocessed sequences
WINDOW_STEP = 12 # Step between windows (not used in the code)

NUM_SENSORS = 30 # number of sensor channels in a window
NUM_CLASSES = 7 # number of diifferent classes in the dataset

# tcnn backbone parameters
NUM_FILTERS = 64 # amount of featuremaps
FILTER_SIZE = 5  # temporal kernel size = (1,FILTER_SIZE)

# Compute related parameters
ACCELERATOR = "gpu" # gpu / cpu training
DEVICE = "0"        # gpu number
PRECISION = 32      # floating point precision


# predicition related
# attribute vectors can be predicted for the motionminers and mbientlab dataset
# classification has to be chosen for all other datasets
MODE = "attribute" # "attribute" / "classification"


#if attribute vector is used as label
NUM_ATTRIBUTES = 19  #amount of attributes in the attribute vector label (Default = 19)
PATH_ATTRIBUTES = "atts_statistics_revised.txt" # file that maps attribute vectors to classes
