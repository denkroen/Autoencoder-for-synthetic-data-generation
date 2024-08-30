## General
This repository contains the code for the synthetic data generation with an autoencoder as described in the paper ""

The complete code can be adapted to different experiments by modifying the config file in the config_files folder. An example config file with all needed flags can be found in the folder.

# Setup

1. The used datasets have to be preprocessed before running the code in order to train the networks. The preprocessing files can be found in xxx.

2. Packages have to be installed

# Steps to run the code

1. Adapt the config file dependent on the dataset and the experiment.

2. Run the create_balanced_csv.py script in order to create a csv file with a balanced amount of data in each class for the dataset. This file is needed to generate the synthetic data with the autoencoder.

3. Train the autoencoder model on a dataset by running the train_pl.py file.

4. Activate the GENERATE_SYNTHETIC flag and specify the path to a trained model and the folder where the synthetic data should be saved.

5. Deactivate the GENERATE_SYNTHETIC flag and Activate the TRAIN_ON_SYNTHETIC flag in order to use the synthetic data as training data.


