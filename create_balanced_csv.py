import pandas as pd
import os
import pickle
import numpy as np
from config_files import config_autoencoder_mbientlab


def balance(root_dir, csv_path, num_classes):

    harwindows = pd.read_csv(csv_path)
    data = []

    for i in range(0,num_classes):
        data.append([])

    for i in range(0, len(harwindows)):
        window_name = os.path.join(root_dir, harwindows.iloc[i, 0])
        f = open(window_name, 'rb')
        data = pickle.load(f, encoding='bytes')
        f.close()
        class_data = data["label"][0]

        data[class_data].append(window_name)

    max_count = 0

    for i in range(0,len(data)):
        if len(data[i]) > max_count:
            max_count = len(data[i])

    for i in range(0,len(data)):
        data[i] = fill_up(data[i], max_count)

    result = []
    for i in range(0, len(data)):
        for j in range(0,data[j]):
            result.append(data[j])

    np.savetxt(root_dir + 'synt.csv', result, delimiter="\n", fmt='%s')



    
    


def fill_up(array, count):

    while len(array) < count:
        iterations = min( (count-len(array)), len(array)) 
        for i in range(0,iterations):
            array.append(array[i])
        #print(len(array))

    print("done")

    return array


if __name__ == '__main__':
    root_dir = config_autoencoder_mbientlab.DATA_DIR
    path_csv_file = root_dir + 'train_final.csv'
    num_classes = 10
    
    balance(root_dir, path_csv_file, num_classes)
