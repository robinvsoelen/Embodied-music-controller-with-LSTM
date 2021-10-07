"""
Library for Bluetooth Low Energy data transfer between the BLE
device and the client. The service used for communication is gatt.
Communication is established via pexpect python library.
"""

# Run with
# sudo /home/tomasz/anaconda3/bin/python ble_gatt.py
# since sudo uses different python version (see "$ sudo which python")

import pexpect
import requests
import signal
import struct
import time
import sys
import glob

import socket

from time import sleep
import pandas as pd

#import visualize as vis
from config import * # Global variables



def getNorthOffset(serversocket):

    ax, ay, az, gx, gy, gz, mx, my, mz,  glove = get_raw_data(serversocket)

    return mx, my,mz

def get_raw_data(serversocket):
    
    bytesAddressPair = serversocket.recvfrom(40)


    content = bytesAddressPair[0]
    address = bytesAddressPair[1]


    clientMsg = "Message from Client:{}".format(content)
    clientIP  = "Client IP Address:{}".format(address)

    glove = ""
    

    axis = 0
    value = ""
    i=0
    mx = 0
    my = 0
    mz = 0
    gz = 0

    buttonPressed = False

    for c in content:
        i += 1
        if chr(c) == 'x' or chr(c) == 'y' or chr(c) =='z' or chr(c) =='a' or chr(c) =='s' or chr(c) =='d' or chr(c) =='b' or chr(c) =='n' or chr(c) =='m' or chr(c) =='u':    
            if (axis == 1):
                x = int(value) - 128
            if (axis == 2): 
                y = int(value) - 128
            if (axis == 3):
                z = int(value) - 128
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
            if (axis == 9):
                mz = int(value) -128

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
            if chr(c)  == 'u': #mz
                axis = 10
                        
            value = ""

        else:
            value += chr(c)
            if i == len(content):
                if chr(c) == 'L': 
                    glove = "LEFT"
                    value = value[0:(len(value)-1)]
                    if (value == 'l'):
                        buttonPressed = False
                    elif (value == 'h'):
                        buttonPressed = True
                elif chr(c) == 'R':
                    value = value[0:(len(value)-1)]
                    glove = "RIGHT"
                    if (value == 'l'):
                        buttonPressed = False
                    elif (value == 'h'):
                        buttonPressed = True                
                elif chr(c) == 'B':   
                    value = value[0:(len(value)-1)]
                    glove = "BODY"

                    if (value == 'l'):
                        buttonPressed = False
                    elif (value == 'h'):
                        buttonPressed = True                    
                else:
                    print("Not sure where I got this from")
                    print(address)


    return x,y,z,gx,gy,gz,mx, my,mz, buttonPressed, glove


prev_derivative = 0
last_strike_time = time.perf_counter()

def strikeTesting(ax, prev_ax, gx):
    global prev_derivative
    global last_strike_time
    amplitude_thresh = 0.05
    ddamplitude_thresh = 0.01
    ax_thresh = -0.2
    ax_der_amplitude_thresh = -0.07

    time_thresh = 1
    derivative = ax - prev_ax
    double_derivative = prev_derivative - derivative

    #print(double_derivative)

    if (prev_derivative > 0 and derivative<0):
        if (gx > amplitude_thresh):
            if (derivative < ax_der_amplitude_thresh):
                if (double_derivative>ddamplitude_thresh):
                    if (ax > ax_thresh):
                        new_strike_time = time.perf_counter()
                        print(new_strike_time - last_strike_time)
                        if (new_strike_time - last_strike_time> time_thresh):
                            last_strike_time = time.perf_counter()
                            print("HIT")
                            prev_derivative = derivative
                            return True
                        else:
                            prev_derivative = derivative
                            return False                        
                    else:
                        prev_derivative = derivative
                        return False       
                else:
                    prev_derivative = derivative
                    return False
            else:
                prev_derivative = derivative
                return False
        else:
            prev_derivative = derivative
            return False
    else:
        prev_derivative = derivative
        return False


ax_readings = []
ay_readings = []
az_readings = []
gx_readings = []
gy_readings = []
gz_readings = []
mx_readings = []
my_readings = []
mz_readings = []
def collect_data(activity, serversocket, trigger,
                 data_collection_time=DATA_COLLECTION_TIME,
                 visualize=False):
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

    prev_mx = 0
    prev_my = 0
    prev_mz = 0

    activity_list = []
    inner_loop_counter = 0

    button = True
    while button:
        ax, ay, az, gx, gy, gz, mx, my, mz, button, glove = get_raw_data(serversocket=serversocket)

    #time.sleep(2)
    #mx_offset, my_offset, mz_offset = getNorthOffset(serversocket)

    prev_left_array = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    prev_right_array = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    prev_body_array = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    while(inner_loop_counter < data_collection_time): # pragma: no cover
        ax, ay, az, gx, gy, gz, mx, my, mz, button, glove = get_raw_data(serversocket=serversocket)

        #if trigger gesture and button is pressed or no trigger gesture
        if ((trigger and button) or not trigger):

            dmx = mx - prev_mx
            dmy = my - prev_my
            dmz = mz - prev_mz


            #print(str(prev_left_array[0]))
            #print("RIGHT: " + str(prev_right_array[0]))
            # Scale to the same range as WISDM dataset
            ax = ax/SCALE_FACTOR
            ay = ay/SCALE_FACTOR
            az = az/SCALE_FACTOR

            gx = gx/SCALE_FACTOR
            gy = gy/SCALE_FACTOR
            gz = gz/SCALE_FACTOR

            mx = dmx/SCALE_FACTOR
            my = dmy/SCALE_FACTOR
            mz = dmz/SCALE_FACTOR

            fine_to_save = False

            if glove == "LEFT":
                this_left_array = [ax,ay,az,gx,gy,gz,mx,my,mz]
            if glove == "RIGHT":
                this_right_array = [ax,ay,az,gx,gy,gz,mx,my,mz]
            if glove == "BODY":
                this_body_array = [ax,ay,az,gx,gy,gz,mx,my,mz]

            if (glove == "LEFT" and this_left_array != prev_left_array):
                fine_to_save = True
            if (glove == "RIGHT" and this_right_array != prev_right_array):
                fine_to_save = True
            if (glove == "BODY" and this_body_array != prev_body_array):
                fine_to_save = True


            if fine_to_save:                
                ax_readings.append(ax)
                ay_readings.append(ay)
                az_readings.append(az)
                gx_readings.append(gx)
                gy_readings.append(gy)
                gz_readings.append(gz)
                mx_readings.append(mx)
                my_readings.append(my)
                mz_readings.append(mz)
                sensor.append(glove)
                inner_loop_counter += 1
                print(ax, ay, az, gx, gy, gz, mx, my, mz, button, glove)


                #strikeTesting(ax, prev_left_array[0],  gx)

                if glove == "LEFT":
                    prev_left_array = [ax,ay,az,gx,gy,gz,mx,my,mz]
                if glove == "RIGHT":
                    prev_right_array = [ax,ay,az,gx,gy,gz,mx,my,mz]
                if glove == "BODY":
                    prev_body_array = [ax,ay,az,gx,gy,gz,mx,my,mz]



    activity_list += [activity for _ in range(data_collection_time)]
    data_dict = {
                COLUMN_NAMES[0]: activity_list, COLUMN_NAMES[1]: ax_readings,
                COLUMN_NAMES[2]: ay_readings, COLUMN_NAMES[3]: az_readings, \
                COLUMN_NAMES[4]: gx_readings, COLUMN_NAMES[5]: gy_readings, \
                COLUMN_NAMES[6]: gz_readings, COLUMN_NAMES[7]: mx_readings, COLUMN_NAMES[8]: my_readings, \
                COLUMN_NAMES[9]: mz_readings,  COLUMN_NAMES[10]: glove
                }
    data_frame = pd.DataFrame(data=data_dict)
    return data_frame


def save_activity_data(activity, serversocket, LABELS_SET, TEMP_DIR, trigger):
    """
    Interface function for the web client (data collection).
    Client (flask app) revokes this function with a particular
    activity as an input (i.e. "Pushup"). The function establishes
    contact with a BLE device, collects the data and saves in a .pckl
    format.
    The function does not return anything.
    """

    if(activity not in LABELS_SET):
        print("Error: Wrong activity")
        raise NameError
    print("Selected activity: ", activity)

    data_frame = collect_data(activity, serversocket, trigger)
    data_frame = data_frame.dropna()

        # Save sample
    num_files = len(glob.glob(TEMP_DIR + '*.pckl'))
    data_frame.to_pickle(TEMP_DIR + 'sample_{}_{}.pckl'.format(activity, num_files + 1))

    print("----- ACTIVITY SAVED ----\n" * 20)



if __name__ == '__main__':
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    host = IP_ADDRESS
    port = 8888

    serversocket.bind((host, port))

    print ('server started and listening')
    
    data = collect_data("test", serversocket, True)
    #data = collect_data("test", serversocketR)


    print(data)
    
    #while 1:
    #    ax,ay,az,gx,gy,gz,mx,my,mz, glove = get_raw_data(serversocket)
    #    print(ax)
