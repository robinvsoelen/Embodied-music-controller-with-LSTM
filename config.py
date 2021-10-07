"""
Config file.
All other scripts import variables from this config
to their global namespace.
"""

import os
##################################################
### GLOBAL VARIABLES
##################################################
COLUMN_NAMES = [
    'activity',
    'acc-x-axis',
    'acc-y-axis',
    'acc-z-axis',
    'gyro-x-axis',
    'gyro-y-axis',
    'gyro-z-axis',
    'mag-x-axis',
    'mag-y-axis',
    'mag-z-axis',
    'glove'
]

LABELS_NAMES = []
TRIGGER_NAMES = []

# open file and read the content in a list
with open('gestures.txt', 'r') as filehandle:
    LABELS_NAMES = [gesture.rstrip() for gesture in filehandle.readlines()]

    # open file and read the content in a list
with open('trigger_gestures.txt', 'r') as filehandle:
    TRIGGER_NAMES = [gesture.rstrip() for gesture in filehandle.readlines()]

GESTURE_COMBOS = []
# open file and read the content in a list
with open('GESTURE_COMBOS.txt', 'r') as filehandle:
    GESTURE_COMBOS = [gesture.rstrip() for gesture in filehandle.readlines()]

BODY_GESTURES = []
# open file and read the content in a list
with open('BODY_GESTURES.txt', 'r') as filehandle:
    BODY_GESTURES = [gesture.rstrip() for gesture in filehandle.readlines()]

CONFIGURATIONS = []
CONFIGURATIONS = os.listdir('configurations')
# Data directories
DATA_DIR = 'data/'
DATA_TEMP_DIR = 'data_temp/'
TRIGGER_DATA_TEMP_DIR = 'trigger_data_temp/'
DATA_PATH = 'data/data.pckl'
TRIGGER_DATA_PATH = 'data/trigger_data.pckl'
BODY_DATA_TEMP_DIR = 'body_data_temp/'

# Model directories
MODEL_PATH_LEFT = 'models/model_left.h5'
MODEL_PATH_TRIGGER_LEFT = 'models/model_trigger_left.h5'
MODEL_PATH_TRIGGER_RIGHT = 'models/model_trigger_right.h5'
MODEL_PATH_RIGHT = 'models/model_right.h5'
MODEL_PATH_BODY = 'models/model_body.h5'
MODEL_PATH_DIR = 'models/'

##################################################
### MODEL
##################################################
# Used for shuffling data
RANDOM_SEED = 13

# Model
N_CLASSES = len(LABELS_NAMES)
N_FEATURES = len(COLUMN_NAMES) - 2

# Hyperparameters
N_LSTM_LAYERS = 2
N_EPOCHS = 10
LEARNING_RATE = 0.0005
N_HIDDEN_NEURONS = 50
BATCH_SIZE = 30
DROPOUT_RATE = 0.5

##################################################
### DATA COLLECTION/PREPROCESSING
##################################################
IMU_MAC_ADDRESS = "FF:3C:8F:22:C9:C8"
UUID_DATA = "2d30c082-f39f-4ce6-923f-3484ea480596"
BLE_HANDLE = "0x0011"

#Noorderhagen
#IP_ADDRESS = "192.168.2.24"
#IP_GLOVE_LEFT = '192.168.2.33'
#IP_GLOVE_RIGHT = '192.168.2.32'
#IP_BODY = '192.168.2.18'

LEFT_PORT = 8888
RIGHT_PORT = 8889
BODY_PORT = 8890


#noorderhagen
#IP_ADDRESS = "192.168.2.22"

#phone hotspot
#IP_ADDRESS = "192.168.43.26"

#eduroam
IP_ADDRESS = "192.168.137.1"

#Koningsbergen
#IP_ADDRESS = '192.168.178.13'

#IP UTGUEST
#IP_ADDRESS = "10.53.29.149"

#Jasmin's place
#IP_ADDRESS = "192.168.0.158"

#Sint oedenrode
#IP_ADDRESS = "192.168.1.54"

# Timeout exception time
TIMEOUT_EXCEPTION_TIME = 5

# Frequency of requesting data from BLE device
BLE_DATA_COLLECTION_LATENCY = 0.35

# Data type sent from the device
DATA_TYPE = 'h' # Short integer
DATA_SIZE_BYTES = 2 # Size of short

# How many times to collect samples
DATA_COLLECTION_TIME = 1000
# Factor to scale the readings
SCALE_FACTOR = 128

# Data preprocessing
TIME_STEP = 5
SEGMENT_TIME_SIZE = 20

BUTTON_SEGMENT_TIME_SIZE = 20

# Train/test proportion
TEST_SIZE = 0.2

##################################################
### VISUALIZE
##################################################
# Interactive data visualization plot ranges
plotRange_x = 50
plotRange_y = 20

##################################################
### BACKEND REQUEST
##################################################
PROTOCOL = "http://"
PORT = ":5000"
IP_EXTERNAL = PROTOCOL + "104.40.158.95" + PORT
IP_LOCAL = "http://192.168.1.71:5000"
PAYLOAD_KEY = "payload_json"
