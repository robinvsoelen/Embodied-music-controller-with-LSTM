"""
Script used to test the classifier.
It loads the data stored in the "DATA_PATH" path
(as specified in config file) and tests the pretrained
model available at "MODEL_PATH".
The script plots a confusion matrix and prints the
performance of the classifier.
"""

import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle

from visualize import drawConfusionMatrix
from preprocessing import get_convoluted_data, one_hot_to_label, softmax_to_one_hot
from config import *
import socket
import time



def get_prediction(model, data, label_set):
    """
    Function to test the keras model on a dataset.
    Take as an input a keras model and a (pandas) test dataframe and
    perform a prediction. Return two numpy arrays: predicted and true
    labels.
    """

    X_test, y_test = get_convoluted_data(data, LABEL_SET=label_set)

    # Make predictions
    y_predicted = model.predict(X_test)
    y_predicted = np.asarray([softmax_to_one_hot(y) for y in y_predicted])
    

    predictions = []
    for actual, predicted in zip(y_test, y_predicted):
        predictions.append(one_hot_to_label(predicted, label_set))
    
    #print(max(set(predictions), key=predictions.count), glove)

    return max(set(predictions), key=predictions.count)

def collect_data_realtime(sequence_length, serversocket, glove):
    """
    Take as input:
    - an activity for which data will be collected,
    - data collection time (how many samples will be collected),
    default value: DATA_COLLECTION_TIME (set in the config).
    - boolean (visualize), indicating whether the data should be visualized
    (interactive visualization is discouraged as it introduces serious lag).
    The function returns a (pandas) dataframe (dictionary) with
    collected data.
    """


    ax_readings = []
    ay_readings = []
    az_readings = []
    gx_readings = []
    gy_readings = []
    gz_readings = []
    mx_readings = []
    my_readings = []
    mz_readings = []
    sensor = []



    activity_list = []

    while (len(ax_readings) < sequence_length):

        # Scale to the same range as WISDM dataset
        ax = ax/SCALE_FACTOR
        ay = ay/SCALE_FACTOR
        az = az/SCALE_FACTOR

        gx = gx/SCALE_FACTOR
        gy = gy/SCALE_FACTOR
        gz = gz/SCALE_FACTOR

        mx = mx/SCALE_FACTOR
        my = my/SCALE_FACTOR
        mz = mz/SCALE_FACTOR



        #print("Acceleration x, y, z: ", ax, ay, az)
        # print("Gyroscope x, y, z: ", gx, gy, gz)
        # print("Magnetometer x, y, z: ", mx, my, mz)
        ax_readings.append(ax)
        ay_readings.append(ay)
        az_readings.append(az)
        gx_readings.append(gx)
        gy_readings.append(gy)
        gz_readings.append(gz)
        mx_readings.append(gx)
        my_readings.append(gy)
        mz_readings.append(gz)
        sensor.append(glove)


    
    while (len(ax_readings) > sequence_length):
        ax_readings.pop(0)
        ay_readings.pop(0)
        az_readings.pop(0)
        gx_readings.pop(0)
        gy_readings.pop(0)
        gz_readings.pop(0)
        mx_readings.pop(0)
        my_readings.pop(0)
        mz_readings.pop(0)
        sensor.pop(0)
    
     
    data_dict = {
                COLUMN_NAMES[0]: LABELS_NAMES[0], COLUMN_NAMES[1]: ax_readings,
                COLUMN_NAMES[2]: ay_readings, COLUMN_NAMES[3]: az_readings, \
                COLUMN_NAMES[4]: gx_readings, COLUMN_NAMES[5]: gy_readings, \
                COLUMN_NAMES[6]: gz_readings, COLUMN_NAMES[7]: mx_readings, COLUMN_NAMES[8]: my_readings, \
                COLUMN_NAMES[9]: mz_readings, COLUMN_NAMES[10]: glove
                }
    data_frame = pd.DataFrame(data=data_dict)

    ax_readings = []
    ay_readings = []
    az_readings = []
    gx_readings = []
    gy_readings = []
    gz_readings = []
    mx_readings = []
    my_readings = []
    mz_readings = []
    sensor = []
    return data_frame

def strikeDetector(readingList, last_strike_time):
    amplitude_thresh = 4
    ddamplitude_thresh = 2
    time_thresh = 1
    derivative = readingList[len(readingList)-1] - readingList[len(readingList)] 
    prev_derivative =  readingList[len(readingList)-2] - readingList[len(readingList)-1]
    double_derivative = prev_derivative - derivative

    if (prev_derivative > 0 and derivative<0):
        if (readingList[len(readingList)] > amplitude_thresh):
            if (double_derivative>ddamplitude_thresh):
                if (time.perf_counter() - last_strike_time> time_thresh):
                    last_strike_time = time.perf_counter()
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False

def get_raw_data_realtime(serversocket):

    bytesAddressPair = serversocket.recvfrom(50)
    content = bytesAddressPair[0]
    address = bytesAddressPair[1]

    clientMsg = "Message from Client:{}".format(content)
    clientIP  = "Client IP Address:{}".format(address)

    glove = ""
    if address[0] == IP_GLOVE_LEFT: 
        glove = "LEFT"
    elif address[0] == IP_GLOVE_RIGHT: 
        glove = "RIGHT"
    elif address[0] == IP_BODY: 
        glove = "BODY"
    else:
        print("maybe an IP address changed?")
        print(address)


    axis = 0
    value = ""
    i=0
    mx = 0
    my = 0
    for c in content:
        i += 1
        if chr(c) == 'x' or chr(c) == 'y' or chr(c) =='z' or chr(c) =='a' or chr(c) =='s' or chr(c) =='d' or chr(c) =='b' or chr(c) =='n' or chr(c) =='m':    
            if (axis == 1):
                x = int(value) -128
            if (axis == 2): 
                y = int(value) -128
            if (axis == 3):
                z = int(value) -128
            if (axis == 4):
                gx = int(value) -128
            if (axis == 5):
                gy = int(value) -128
            if (axis == 6):
                gz = int(value) -128
            if (axis == 7):
                mx = int(value) -128
            if (axis == 8):
                my = int(value) -128


            if chr(c)  == 'x':
                axis = 1
            if chr(c)  == 'y':
                axis = 2
            if chr(c)  == 'z':
                axis = 3
            if chr(c)  == 'a': #gx
                axis = 4
            if chr(c)  == 's': #gy
                axis = 5
            if chr(c)  == 'd': #gz
                axis = 6
            if chr(c)  == 'b': #mx
                axis = 7
            if chr(c)  == 'n': #my
                axis = 8
            if chr(c)  == 'm': #mz
                axis = 9
                        
            value = ""

        else:
            value += chr(c)
            if i == len(content):
                if (axis == 9):
                    mz = int(value) - 128
                if (axis == 6):
                    gz = int(value) - 128
            else:
                mz = 0

    return x,y,z,gx,gy,gz,mx, my,mz, glove


if __name__ == '__main__':


    serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    host = IP_ADDRESS
    port = 8888
    serversocket.bind((host, port))
    print ('server started and listening')

    ax_readings = []
    ay_readings = []
    az_readings = []
    gx_readings = []
    gy_readings = []
    gz_readings = []
    sensor = []

    sequence_length = 300

    iteration = 0
    test_frequency = 1

    model_left = load_model(MODEL_PATH_LEFT)
    model_right = load_model(MODEL_PATH_RIGHT)

    while 1:
        iteration += 1
        
        data = collect_data_realtime(sequence_length, serversocket, "LEFT")


        data_right = data.loc[data['glove'] == "RIGHT"]
        data_left = data.loc[data['glove'] == "LEFT"]

        data_right = data.drop(['glove'], axis=1)
        data_left = data.drop(['glove'], axis=1)

        if iteration == test_frequency:
            if (len(data.index) == sequence_length):
        
                y_predicted, y_test = get_prediction(model_left, data_left, "LEFT")
                #cy_predicted, y_test = get_prediction(model_right, data_right, "RIGHT")

            iteration = 0
    





