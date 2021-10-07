from preprocessing import label_position
from ble_gatt import collect_data, get_raw_data, save_activity_data
import tkinter as tk
from config import *
import socket
import pandas as pd
from functools import partial
from merge_data import merge_pckls
from model_train_keras import *
import os
from model_test_keras import *
import mido
import time
from mido import Message
from predict_realtime import collect_data_realtime, get_prediction, get_raw_data_realtime
import keyboard
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, shutil
import sys
import ctypes




outport = mido.open_output('thebestport 1')

#left_serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#body_serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#right_serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


host = IP_ADDRESS

#left_serversocket.bind((host, LEFT_PORT))
#body_serversocket.bind((host, BODY_PORT))
#right_serversocket.bind((host, RIGHT_PORT))

serversocket.bind((host, LEFT_PORT))



print ('server started and listening')

window = tk.Tk()
window.title("the learning my computer cool gestures and mapping those to midi notes-inator")

window.tk.call('tk', 'scaling', 0.9)

#spacing
space = tk.Label(text="  ")
space.grid(row=0, column=0)

gloveTrainRowStart = 0
gloveTrainColumnStart = 1

#Train LSTM
trainTitle = tk.Label(text="Train Hands")
trainTitle.config(font=("Courier", 10))
trainTitle.grid(row=gloveTrainRowStart, column=gloveTrainColumnStart)



def updateButtonPositions(rowStart):
    trainLEFTButton.grid(row=rowStart+ len(gestures)+4, column=gloveTrainColumnStart)
    trainRIGHTButton.grid(row=rowStart + len(gestures)+4, column=gloveTrainColumnStart + 1)

    evaluateLEFTButton.grid(row=rowStart + len(gestures)+5, column=gloveTrainColumnStart)
    evaluateRIGHTButton.grid(row=rowStart + len(gestures)+5, column=gloveTrainColumnStart + 1)

    mergeDataButton.grid(row=rowStart + len(gestures)+3, column=gloveTrainColumnStart)
    entry.grid(row= rowStart + len(gestures)+2, column=gloveTrainColumnStart)
    AddGestureButton.grid(row=rowStart + len(gestures)+2, column=gloveTrainColumnStart + 1)


class Gesture:
    def __init__(self, gesture):
        self.gesture = gesture
        self.label = tk.Label(text= self.gesture)
        self.removeButton = tk.Button(text="Remove", command=lambda: self.removeGesture())
        self.getDataButton = tk.Button(text="Get data", command=lambda: self.getData())
        self.amountOfSamples = tk.Label(text= "Samples: " + self.getAmountOfDataSamples())
        self.initial = True

    def removeGesture(self):
        LABELS_NAMES.remove(self.gesture)
        print(LABELS_NAMES)     
        saveLabelNames()
        self.label.destroy()
        self.removeButton.destroy()
        self.getLeftDataButton.destroy()
        self.getRightDataButton.destroy()
        self.amountOfSamples.destroy()
        self.removeDataSamples()
        updateLabels()
        updateButtonPositions(gloveTrainRowStart)

    def addLabel(self, i):
        self.label.grid(row=gloveTrainRowStart + i+1, column=gloveTrainColumnStart)
        self.removeButton.grid(row=gloveTrainRowStart + i+1, column=gloveTrainColumnStart + 1)
        self.getDataButton.grid(row=gloveTrainRowStart + i+1, column= gloveTrainColumnStart + 2)
        self.amountOfSamples.grid(row=gloveTrainRowStart + i+1, column=gloveTrainColumnStart + 3)
        window.update()


    def getData(self):
        save_activity_data(self.gesture, serversocket= serversocket, LABELS_SET=LABELS_NAMES, TEMP_DIR=DATA_TEMP_DIR, trigger=False)
        self.amountOfSamples = tk.Label(text= "samples: " + self.getAmountOfDataSamples())
        reload_everything()

    def getAmountOfDataSamples(self):
        list = os.listdir(DATA_TEMP_DIR)
        matching = [s for s in list if self.gesture in s]
        return str(len(matching))

    def removeDataSamples(self):
        list = os.listdir(DATA_TEMP_DIR)
        matching = [s for s in list if self.gesture in s]
        for match in matching:
            os.remove(DATA_TEMP_DIR + match)


def saveLabelNames():
    with open('gestures.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % gesture for gesture in LABELS_NAMES)

gestures = []

for gestureName in LABELS_NAMES:
    gestures.append(Gesture(gestureName))


def updateLabels():
    i=0
    for gesture in gestures:
        i += 1
        gesture.addLabel(i)

updateLabels()

entry = tk.Entry()
def appendGesture():
    LABELS_NAMES.append(entry.get())
    gestures.append(Gesture(entry.get()))
    gestures[len(gestures)-1].addLabel(len(gestures))
    updateLabels()
    updateButtonPositions(gloveTrainRowStart)
    saveLabelNames()

AddGestureButton = tk.Button(text="add gesture", command=appendGesture)

trainLEFTButton = tk.Button(text="train left hand", command=lambda: train("LEFT", False))
trainRIGHTButton = tk.Button(text="train right hand", command=lambda: train("RIGHT", False))
evaluateLEFTButton = tk.Button(text="evaluate left hand", command=lambda: evaluate("LEFT", False))
evaluateRIGHTButton = tk.Button(text="evaluate right hand", command=lambda: evaluate("RIGHT", False))
mergeDataButton = tk.Button(text="merge data", command= lambda: mergeData('LEFT', False))
updateButtonPositions(gloveTrainRowStart)


##################################### trigger gestures ######################################
gloveTriggerTrainRowStart = gloveTrainRowStart + len(gestures)+6

triggerLabel = tk.Label(text= "Gestures for triggering stuff")
triggerLabel.config(font=("Courier", 10))
triggerLabel.grid(row=gloveTriggerTrainRowStart, column=gloveTrainColumnStart)


class Trigger_Gesture:
    def __init__(self, gesture):
        self.gesture = gesture
        self.label = tk.Label(text= self.gesture)
        self.removeButton = tk.Button(text="Remove", command=lambda: self.removeGesture())
        self.getDataButton = tk.Button(text="Get data", command=lambda: self.getData())
        self.amountOfSamples = tk.Label(text= "Samples: " + self.getAmountOfDataSamples())
        self.initial = True

    def removeGesture(self):
        TRIGGER_NAMES.remove(self.gesture)
        print(LABELS_NAMES)     
        saveTriggerLabelNames()
        self.label.destroy()
        self.removeButton.destroy()
        self.getDataButton.destroy()
        self.amountOfSamples.destroy()
        self.removeDataSamples()
        updateTriggerLabels()
        updateTRIGGERButtonPositions(gloveTriggerTrainRowStart)

    def addLabel(self, i):
        self.label.grid(row=gloveTriggerTrainRowStart + i+1, column=gloveTrainColumnStart)
        self.removeButton.grid(row=gloveTriggerTrainRowStart + i+1, column=gloveTrainColumnStart + 1)
        self.getDataButton.grid(row=gloveTriggerTrainRowStart + i+1, column= gloveTrainColumnStart + 2)
        self.amountOfSamples.grid(row=gloveTriggerTrainRowStart + i+1, column=gloveTrainColumnStart + 4)
        window.update()

    def getData(self):
        save_activity_data(self.gesture, serversocket= serversocket, LABELS_SET=TRIGGER_NAMES, TEMP_DIR=TRIGGER_DATA_TEMP_DIR, trigger=True)
        self.amountOfSamples = tk.Label(text= "samples: " + self.getAmountOfDataSamples())
        reload_everything()

    def getAmountOfDataSamples(self):
        list = os.listdir(TRIGGER_DATA_TEMP_DIR)
        matching = [s for s in list if self.gesture in s]
        return str(len(matching))

    def removeDataSamples(self):
        list = os.listdir(TRIGGER_DATA_TEMP_DIR)
        matching = [s for s in list if self.gesture in s]
        for match in matching:
            os.remove(TRIGGER_DATA_TEMP_DIR + match)


def saveTriggerLabelNames():
    with open('trigger_gestures.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % gesture for gesture in TRIGGER_NAMES)

trigger_gestures = []

for gestureName in TRIGGER_NAMES:
    trigger_gestures.append(Trigger_Gesture(gestureName))

def updateTriggerLabels():
    i=0
    for gesture in trigger_gestures:
        i += 1
        gesture.addLabel(i)

updateTriggerLabels()


def updateTRIGGERButtonPositions(rowStart):
    trigger_trainLEFTButton.grid(row=rowStart+ len(trigger_gestures)+4, column=gloveTrainColumnStart)
    trigger_trainRIGHTButton.grid(row=rowStart + len(trigger_gestures)+4, column=gloveTrainColumnStart + 1)

    trigger_evaluateLEFTButton.grid(row=rowStart + len(trigger_gestures)+5, column=gloveTrainColumnStart)
    trigger_evaluateRIGHTButton.grid(row=rowStart + len(trigger_gestures)+5, column=gloveTrainColumnStart + 1)

    trigger_mergeDataButton.grid(row=rowStart + len(trigger_gestures)+3, column=gloveTrainColumnStart)
    trigger_entry.grid(row= rowStart + len(trigger_gestures)+2, column=gloveTrainColumnStart)
    AddTRIGGERGestureButton.grid(row=rowStart + len(trigger_gestures)+2, column=gloveTrainColumnStart + 1)


trigger_entry = tk.Entry()
def appendTriggerGesture():
    TRIGGER_NAMES.append(trigger_entry.get())
    trigger_gestures.append(Trigger_Gesture(trigger_entry.get()))
    trigger_gestures[len(trigger_gestures)-1].addLabel(len(trigger_gestures))
    updateTriggerLabels()
    updateTRIGGERButtonPositions(gloveTriggerTrainRowStart)
    saveTriggerLabelNames()

AddTRIGGERGestureButton = tk.Button(text="add gesture", command=appendTriggerGesture)



############training gloves############

def mergeData(glove, trigger):
    if (glove == 'LEFT' or glove== 'RIGHT'):
        if (trigger):
            try:
                df = merge_pckls(TRIGGER_DATA_TEMP_DIR)
                print("Final data shape: ", df.shape)
            except ValueError:
                raise ValueError
        if (not trigger):
            try:
                df = merge_pckls(DATA_TEMP_DIR)
                print("Final data shape: ", df.shape)
            except ValueError:
                raise ValueError

    if (glove == 'BODY'):
        try:
            df = merge_pckls(BODY_DATA_TEMP_DIR)
            print("Final data shape: ", df.shape)
        except ValueError:
            raise ValueError

def train(glove, trigger):
    # Load data
    if (trigger):
        data = pd.read_pickle(TRIGGER_DATA_PATH)
    else:
        data = pd.read_pickle(DATA_PATH)
    print(data)
    data = data.loc[data['glove'] == glove]
    data = data.drop(['glove'], axis=1)
    print(data)

    LABEL_SET = []
    if (glove == 'BODY'):
        LABEL_SET = BODY_GESTURES
    elif trigger:
        LABEL_SET = TRIGGER_NAMES
    else:
        LABEL_SET = LABELS_NAMES


    data_convoluted, labels = get_convoluted_data(data, LABEL_SET)
    X_train, X_val, y_train, y_val = train_test_split(data_convoluted,
                                                        labels, test_size=TEST_SIZE,
                                                        random_state=RANDOM_SEED,
                                                        shuffle=True)

    if (glove == 'BODY'):
    # Build a model
        model = createBidirectionalLSTM(SEGMENT_TIME_SIZE,
                                        LEARNING_RATE,
                                        N_HIDDEN_NEURONS,
                                        DROPOUT_RATE,
                                        N_EPOCHS,
                                        BATCH_SIZE,
                                        X_train, y_train,
                                        X_val, y_val, 6, len(LABEL_SET),
                                        visualize=False)
    else:                                    
        model = createBidirectionalLSTM(SEGMENT_TIME_SIZE,
                                    LEARNING_RATE,
                                    N_HIDDEN_NEURONS,
                                    DROPOUT_RATE,
                                    N_EPOCHS,
                                    BATCH_SIZE,
                                    X_train, y_train,
                                    X_val, y_val, 6, len(LABEL_SET),
                                    visualize=False)                                        
    if glove == "LEFT" and not trigger:
        model.save(MODEL_PATH_LEFT)
    if glove == "LEFT" and trigger:
        model.save(MODEL_PATH_TRIGGER_LEFT)
    if glove == "RIGHT" and not trigger:
        model.save(MODEL_PATH_RIGHT)
    if glove == "RIGHT" and trigger:
        model.save(MODEL_PATH_TRIGGER_RIGHT)
    if glove == "RIGHT" and trigger:
        model.save(MODEL_PATH_RIGHT)
    if glove == "BODY":
        model.save(MODEL_PATH_BODY)

def evaluate(glove, trigger):
    if (trigger):
        data = pd.read_pickle(TRIGGER_DATA_PATH)
    else:
        data = pd.read_pickle(DATA_PATH)

    data = data.loc[data['glove'] == glove]
    LABEL_SET = []
    if glove == "LEFT":
        if (not trigger):
            model = load_model(MODEL_PATH_LEFT)
            LABEL_SET = LABELS_NAMES
        if (trigger):
            model = load_model(MODEL_PATH_TRIGGER_LEFT)
            LABEL_SET = TRIGGER_NAMES
    if glove == "RIGHT":
        if (not trigger):
            model = load_model(MODEL_PATH_RIGHT)
            LABEL_SET = LABELS_NAMES
        if (trigger):
            model = load_model(MODEL_PATH_TRIGGER_RIGHT)
            LABEL_SET = TRIGGER_NAMES
    if glove == "BODY":
        LABEL_SET = BODY_GESTURES

    y_predicted, y_test = test_model(model, data, LABEL_SET)
    print("Final accuracy: ", accuracy_score(y_test, y_predicted))

    # Confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_predicted, axis=1))
    drawConfusionMatrix(cm)
       



trigger_trainLEFTButton = tk.Button(text="train left hand", command=lambda: train("LEFT", True))
trigger_trainRIGHTButton = tk.Button(text="train right hand", command=lambda: train("RIGHT", True))
trigger_evaluateLEFTButton = tk.Button(text="evaluate left hand", command=lambda: evaluate("LEFT", True))
trigger_evaluateRIGHTButton = tk.Button(text="evaluate right hand", command=lambda: evaluate("RIGHT", True))
trigger_mergeDataButton = tk.Button(text="merge data", command= lambda: mergeData('LEFT', True))
updateTRIGGERButtonPositions(gloveTriggerTrainRowStart)



########################################## train body sensor ####################################
bodyTrainStart = 1
bodyTrainRowStart = 100

#spacing
#space = tk.Label(text="             ")
#space.grid(row=bodyTrainRowStart, column=bodyTrainStart)

#Train LSTM
trainTitle = tk.Label(text="Train Body")
trainTitle.config(font=("Courier", 10))
trainTitle.grid(row=bodyTrainRowStart, column=bodyTrainStart)



def updateBODYButtonPositions():
    trainBODYButton.grid(row=bodyTrainRowStart+ len(bodyGestures)+4, column=bodyTrainStart)
    evaluateBODYButton.grid(row=bodyTrainRowStart + len(bodyGestures)+5, column=bodyTrainStart)
    mergeBODYDataButton.grid(row=bodyTrainRowStart + len(bodyGestures)+3, column=bodyTrainStart)
    bodyEntry.grid(row=bodyTrainRowStart + len(bodyGestures)+2, column=bodyTrainStart)
    bodyGestureAdd.grid(row=bodyTrainRowStart + len(bodyGestures)+2, column=bodyTrainStart+1)

class BodyGesture:
    def __init__(self, gesture):
        self.gesture = gesture
        self.label = tk.Label(text= self.gesture)
        self.removeButton = tk.Button(text="remove", command=lambda: self.removeGesture())
        self.getDataButton = tk.Button(text="get data", command=lambda: self.getData())
        self.amountOfSamples = tk.Label(text= "samples: " + self.getAmountOfDataSamples())
        self.initial = True

    def removeGesture(self):
        try:
            BODY_GESTURES.remove(self.gesture)
            print(BODY_GESTURES)    
        except:
            pass     
        saveBODYLabelNames()
        self.label.destroy()
        self.removeButton.destroy()
        self.getDataButton.destroy()
        self.amountOfSamples.destroy()
        self.removeDataSamples()
        updateLabels()
        updateBODYButtonPositions()

    def addLabel(self, i):
        self.label.grid(row=bodyTrainRowStart + i+1, column=bodyTrainStart)
        self.removeButton.grid(row=bodyTrainRowStart + i+1, column=bodyTrainStart +1)
        self.getDataButton.grid(row=bodyTrainRowStart + i+1, column=bodyTrainStart +2)
        self.amountOfSamples.grid(row=bodyTrainRowStart + i+1, column= bodyTrainStart +3)

    def getData(self):
        save_activity_data(self.gesture, serversocket=serversocket, LABELS_SET=BODY_GESTURES, TEMP_DIR=BODY_DATA_TEMP_DIR, trigger=False)
        self.amountOfSamples = tk.Label(text= "samples: " + self.getAmountOfDataSamples())
        updateBODYLabels()

    def getAmountOfDataSamples(self):
        list = os.listdir(BODY_DATA_TEMP_DIR)
        matching = [s for s in list if self.gesture in s]
        return str(len(matching))

    def removeDataSamples(self):
        list = os.listdir(BODY_DATA_TEMP_DIR)
        matching = [s for s in list if self.gesture in s]
        for match in matching:
            os.remove(BODY_DATA_TEMP_DIR + match)


def saveBODYLabelNames():
    with open('BODY_GESTURES.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % gesture for gesture in BODY_GESTURES)

bodyGestures = []

for gestureName in BODY_GESTURES:
    bodyGestures.append(BodyGesture(gestureName))

def updateBODYLabels():
    i=0
    for gesture in bodyGestures:
        i += 1
        gesture.addLabel(i)


updateBODYLabels()

bodyEntry= tk.Entry()
def appendBodyGesture():
    BODY_GESTURES.append(bodyEntry.get())
    bodyGestures.append(BodyGesture(bodyEntry.get()))
    bodyGestures[len(bodyGestures)-1].addLabel(len(bodyGestures))
    updateLabels()
    updateBODYButtonPositions()
    saveBODYLabelNames()

bodyGestureAdd = tk.Button(text="add gesture", command=appendBodyGesture)


trainBODYButton = tk.Button(text="train body sensor", command=lambda: train("BODY", False))
evaluateBODYButton = tk.Button(text="evaluate body sensor", command=lambda: evaluate("BODY", False))
mergeBODYDataButton = tk.Button(text="merge body data", command= lambda: mergeData('BODY', False))
updateBODYButtonPositions()



########################################### MIDI CONGIFIGUTEARIONK SDISDFSF #########################################################

comboColumnStart = 14
comboRowStart = 0

#spacing
space = tk.Label(text="                  ")
space.grid(row=comboRowStart, column=comboColumnStart-1)

#MIDI mapping
midiTitle = tk.Label(text="Map MIDI")
midiTitle.config(font=("Courier", 10))
midiTitle.grid(row=comboRowStart, column=comboColumnStart)


def sendControllerChange(control, value):
    controlChange = Message('control_change', control=control, value=value )
    outport.send(controlChange)

xControlChannelL = 10
yControlChannelL = 11
zControlChannelL = 12

xControlChannelR = 20
yControlChannelR = 21
zControlChannelR = 22

labelLeftHand = tk.Label(text="Left hand")
xButtonL = tk.Button(text="send controllerchange x-axis", command=lambda: sendControllerChange(xControlChannelL, 127))
yButtonL = tk.Button(text="send controllerchange y-axis", command=lambda: sendControllerChange(yControlChannelL, 127))
zButtonL = tk.Button(text="send controllerchange z-axis", command=lambda: sendControllerChange(zControlChannelL, 127))

labelRightHand = tk.Label(text="Right hand")
xButtonR = tk.Button(text="send controllerchange x-axis", command=lambda: sendControllerChange(xControlChannelR, 127))
yButtonR = tk.Button(text="send controllerchange y-axis", command=lambda: sendControllerChange(yControlChannelR, 127))
zButtonR = tk.Button(text="send controllerchange z-axis", command=lambda: sendControllerChange(zControlChannelR, 127))
labelBody = tk.Label(text="Body")


labelLeftHand.grid(row=comboRowStart+2, column=comboColumnStart)
labelBody.grid(row=comboRowStart+2, column=comboColumnStart+1)
labelRightHand.grid(row=comboRowStart+2, column=comboColumnStart+2)

'''
xButtonL.grid(row=3, column=comboColumnStart)
yButtonL.grid(row=4, column=comboColumnStart)
zButtonL.grid(row=5, column=comboColumnStart)

xButtonR.grid(row=3, column=comboColumnStart+1)
yButtonR.grid(row=4, column=comboColumnStart+1)
zButtonR.grid(row=5, column=comboColumnStart+1)
'''

#spacing
space = tk.Label(text="                  ")
space.grid(row=comboRowStart+1, column=comboColumnStart+2)

########################################## GESTURE COMBINATIONS ####################################
class GestureCombinations:
    def __init__(self, gestureLEFT, gestureBODY,  gestureRIGHT):
        self.button = tk.Button(text="Send MIDI", command= self.sendMIDI)
        self.remove = tk.Button(text="Remove", command= self.removeGesture)
        self.gestureLEFT = tk.Label(text=gestureLEFT)
        self.gestureRIGHT = tk.Label(text=gestureRIGHT)
        self.gestureBODY = tk.Label(text=gestureBODY)

        if "accelerometer" in gestureLEFT or "accelerometer" in gestureRIGHT or "accelerometer" in gestureBODY or "gyroscope" in gestureLEFT or "gyroscope" in gestureRIGHT or "gyroscope" in gestureBODY:
            self.isController = True
        else:
            self.isController = False

    def addLabel(self, i):
        self.button.grid(row=comboRowStart+i+3, column=comboColumnStart+3)
        self.gestureLEFT.grid(row=comboRowStart + i+3, column=comboColumnStart)
        self.gestureBODY.grid(row=comboRowStart + i+3, column=comboColumnStart+1)
        self.gestureRIGHT.grid(row= comboRowStart+ i+3, column=comboColumnStart+2)
        self.remove.grid(row=comboRowStart+ i+3, column=comboColumnStart+4)
        self.note = i+52
        self.i = i

    def removeGesture(self):
        GESTURE_COMBOS.remove(self.gestureLEFT.cget("text") + ';' + self.gestureBODY.cget("text") + ';' + self.gestureRIGHT.cget("text"))
        saveComboLabelNames()
        print(GESTURE_COMBOS)
        self.gestureLEFT.destroy()
        self.gestureBODY.destroy()
        self.gestureRIGHT.destroy()
        self.button.destroy()
        self.remove.destroy()
        updateComboLabels()
        updateDropdownPositions()

    def controllerChange(self, value):
        sendControllerChange(self.note, value)

    def sendMIDI(self):
        if self.isController:
            sendControllerChange(self.note, 127)
        else:
            msg_on = Message('note_on', note=self.note)
            msg_off = Message('note_off', note=self.note)
            outport.send(msg_on)
            outport.send(msg_off)


def saveComboLabelNames():
    with open('GESTURE_COMBOS.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % gestureCombo for gestureCombo in GESTURE_COMBOS)


def updateComboLabels():
    gestureCombos = []
    i=0

    for combo in GESTURE_COMBOS:
        combo = combo.split(';')       
        gestureCombos.append(GestureCombinations(combo[0], combo[1], combo[2]))


    for gestureCombo in gestureCombos:
        i += 1
        gestureCombo.addLabel(i)


    return gestureCombos

gestureCombos = updateComboLabels()

def addGestureCombination():
    global gestureCombos
    global GESTURE_COMBOS   
    gestureCombos.append(GestureCombinations(gestureVarL.get(), gestureVarB.get(),  gestureVarR.get()))
    GESTURE_COMBOS.append(gestureVarL.get() + ";" + gestureVarB.get() +  ";" + gestureVarR.get())
    saveComboLabelNames()
    updateComboLabels()
    updateDropdownPositions()

combinationExplanationLabel = tk.Label(text="Set:")
labelLeftHand = tk.Label(text="Left hand")
labelBody = tk.Label(text="Body")
labelRightHand = tk.Label(text="Right hand")


def appendOptions(optionList):
    for gesture in TRIGGER_NAMES:
        optionList.append(gesture)
    optionList.append("accelerometer x")
    optionList.append("accelerometer y")
    optionList.append("accelerometer z")
    optionList.append("gyroscope x")
    optionList.append("gyroscope y")
    optionList.append("gyroscope z")
    optionList.append("ignore me")
    optionList.append("strike")
    return optionList

GLOVE_OPTIONS = LABELS_NAMES.copy()
GLOVE_OPTIONS = appendOptions(GLOVE_OPTIONS)

BODY_OPTIONS = BODY_GESTURES.copy()
BODY_OPTIONS = appendOptions(BODY_OPTIONS)

gestureVarL = tk.StringVar(window)
gestureVarL.set(GLOVE_OPTIONS[0])

gestureVarB = tk.StringVar(window)
gestureVarB.set(BODY_OPTIONS[0])

gestureVarR = tk.StringVar(window)
gestureVarR.set(GLOVE_OPTIONS[0])

gestureDropR = tk.OptionMenu(window, gestureVarR, *GLOVE_OPTIONS)
gestureDropL = tk.OptionMenu(window, gestureVarL, *GLOVE_OPTIONS)
gestureDropB = tk.OptionMenu(window, gestureVarB, *BODY_OPTIONS)

addGestureComboButton = tk.Button(text="Add", command=addGestureCombination)

def updateDropdownPositions():
    combinationExplanationLabel.grid(row=comboRowStart+ 7 + len(GESTURE_COMBOS), column=comboColumnStart)
    labelLeftHand.grid(row=comboRowStart+ 8 + len(GESTURE_COMBOS), column=comboColumnStart)
    labelBody.grid(row=comboRowStart+ 8 + len(GESTURE_COMBOS), column=comboColumnStart +1)
    labelRightHand.grid(row=comboRowStart + 8 + len(GESTURE_COMBOS), column=comboColumnStart+2)
    gestureDropL.grid(row=comboRowStart + 9 + len(GESTURE_COMBOS), column=comboColumnStart)
    gestureDropB.grid(row= comboRowStart+ 9 + len(GESTURE_COMBOS), column=comboColumnStart+1)
    gestureDropR.grid(row= comboRowStart+ 9 + len(GESTURE_COMBOS), column=comboColumnStart+2)
    addGestureComboButton.grid(row=comboRowStart + 9 + len(GESTURE_COMBOS), column=comboColumnStart+3)

updateDropdownPositions()

def convertRange(OldValue, OldMin, OldMax,NewMin, NewMax):
    OldRange = (OldMax - OldMin)
    if (OldRange == 0):
        NewValue = NewMin
    else:
        NewRange = (NewMax - NewMin)  
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

    return int(NewValue)


############################## REALTIME GESTURE RECOGNITION #############################################

def strikeDetector(glove, acc_readingList, gyro_readingList):
    global last_strike_time
    gx_amplitude_thresh = 0.01
    ddamplitude_thresh = 0.05
    ax_thresh = -1.5
    ax_der_amplitude_thresh = -0.01
    time_thresh = 0.1
    derivative = acc_readingList[len(acc_readingList)-1] - acc_readingList[len(acc_readingList)-2] 
    prev_derivative =  acc_readingList[len(acc_readingList)-2] - acc_readingList[len(acc_readingList)-3]
    double_derivative = prev_derivative - derivative

    if (prev_derivative > 0 and derivative<0):
        if (gyro_readingList[len(gyro_readingList) -1] > gx_amplitude_thresh):
            if (derivative < ax_der_amplitude_thresh):
                if (acc_readingList[len(acc_readingList)-1] > ax_thresh):
                    if (double_derivative>ddamplitude_thresh):
                        if (time.perf_counter() - last_strike_time> time_thresh):
                            last_strike_time = time.perf_counter()
                            #print("HIT" + glove)
                            return True
                        else:
                            #print("something with time")
                            return False
                    else:
                        #print(double_derivative)
                        #print("double derivative")
                        return False
                else:
                    #print(ax_thresh)
                    #print("ax_thresh")
                    return False
            else:
                #print(derivative)
                #print("ax derivative")
                return False
        else:
            return False
    else:
        return False

def sendStrikeMIDI(gesture_left, gesture_body, gesture_right, glove, gx):
    for combo in gestureCombos:
        if (glove == "LEFT" and combo.gestureLEFT.cget("text") == "strike"):
            if combo.gestureRIGHT.cget("text") == gesture_right or combo.gestureRIGHT.cget("text") == "ignore me":
                if combo.gestureBODY.cget("text") == gesture_body or combo.gestureBODY.cget("text") == "ignore me":
                    combo.sendMIDI()

        if (glove == "RIGHT" and combo.gestureRIGHT.cget("text") == "strike"):
            if combo.gestureLEFT.cget("text") == gesture_left or combo.gestureLEFT.cget("text") == "ignore me":
                if combo.gestureBODY.cget("text") == gesture_body or combo.gestureBODY.cget("text") == "ignore me":
                    combo.sendMIDI()
    """
        if (glove == "BODY" and combo.gestureBODY.cget("text") == "strike"):
            if combo.gestureRIGHT.cget("text") == gesture_right or combo.gestureRIGHT.cget("text") == "ignore me":
                if combo.gestureLEFT.cget("text") == gesture_left or combo.gestureLEFT.cget("text") == "ignore me":
                    combo.sendMIDI()
                    """

def checkCombosSendMIDI(new_combo, gesture_left, gesture_body, gesture_right, trigger_gesture_left, trigger_gesture_right, glove, ax,ay,az,gx,gy,gz, mx, my,mz):
    ax = ax*SCALE_FACTOR
    ay = ay*SCALE_FACTOR
    
    rangeMin = -128
    rangeMax = 128

    for combo in gestureCombos:
            send_control = False
            parameter = ""

            if (new_combo):
                if combo.gestureLEFT.cget("text") == "ignore me" or combo.gestureLEFT.cget("text") == gesture_left or combo.gestureLEFT.cget("text") == trigger_gesture_left: 
                    if combo.gestureRIGHT.cget("text") == gesture_right or combo.gestureRIGHT.cget("text") == trigger_gesture_right or combo.gestureRIGHT.cget("text") == "ignore me":
                        if combo.gestureBODY.cget("text") == gesture_body or combo.gestureBODY.cget("text") == "ignore me":
                            combo.sendMIDI()
                            control_gestures = []

            if glove == "RIGHT":
                if "accelerometer" in combo.gestureRIGHT.cget("text") or "gyroscope" in combo.gestureRIGHT.cget("text") or "magneto" in combo.gestureRIGHT.cget("text"):
                    if combo.gestureLEFT.cget("text") == gesture_left or combo.gestureLEFT.cget("text") == trigger_gesture_left or combo.gestureLEFT.cget("text") == "ignore me":
                        if combo.gestureBODY.cget("text") == gesture_body or combo.gestureBODY.cget("text") == "ignore me":
                            send_control = True
                            parameter = combo.gestureRIGHT.cget("text")


            if glove == "LEFT":
                if "accelerometer" in combo.gestureLEFT.cget("text") or "gyroscope" in combo.gestureLEFT.cget("text") or "magneto" in combo.gestureLEFT.cget("text"):
                    if combo.gestureRIGHT.cget("text") == gesture_right or combo.gestureRIGHT.cget("text") == trigger_gesture_right or combo.gestureRIGHT.cget("text") == "ignore me":
                        if combo.gestureBODY.cget("text") == gesture_body or combo.gestureBODY.cget("text") == "ignore me":
                            send_control = True
                            parameter = combo.gestureLEFT.cget("text")
            
            if glove == "BODY":
                if "accelerometer" in combo.gestureBODY.cget("text") or "gyroscope" in combo.gestureBODY.cget("text") or "magneto" in combo.gestureBODY.cget("text"):
                    if combo.gestureLEFT.cget("text") == gesture_left or combo.gestureLEFT.cget("text") == trigger_gesture_left or combo.gestureLEFT.cget("text") == "ignore me":
                        if combo.gestureRIGHT.cget("text") == gesture_right or combo.gestureRIGHT.cget("text") == trigger_gesture_right or combo.gestureRIGHT.cget("text") == "ignore me":
                            send_control = True
                            parameter = "accelerometer x body"


            try:
                if (send_control):
                    if "accelerometer" in parameter:
                        if "x" in parameter:
                            if "body" in parameter:
                                if ax > 0:
                                    combo.controllerChange(convertRange(ax, 0, rangeMax, 0, 127))
                            combo.controllerChange(convertRange(ax, rangeMin, rangeMax, 0, 127))
                        if "y" in parameter:
                            combo.controllerChange(convertRange(ay, rangeMin, rangeMax, 0, 127))
                        if "z" in parameter:
                            combo.controllerChange(convertRange(az, rangeMin, rangeMax, 0, 127))
                    if "gyroscope" in parameter:
                        if "x" in parameter:
                            combo.controllerChange(convertRange(gx, rangeMin, rangeMax, 0, 127))
                        if "y" in parameter:
                            combo.controllerChange(convertRange(gy, rangeMin, rangeMax, 0, 127))
                        if "z" in parameter: 
                            combo.controllerChange(convertRange(gz, rangeMin, rangeMax, 0, 127))
                    if "magneto" in parameter:
                        if "x" in parameter:
                            combo.controllerChange(convertRange(mx, rangeMin, rangeMax, 0, 127))
                        if "y" in parameter:
                            combo.controllerChange(convertRange(my, rangeMin, rangeMax, 0, 127))
                        if "z" in parameter: 
                            combo.controllerChange(convertRange(mz, rangeMin, rangeMax, 0, 127))
                        
            except:
                pass

trigger_readings_left = [[],[],[],[],[],[],[],[],[]]
trigger_readings_right = [[],[],[],[],[],[],[],[],[]]
readings_left = [[],[],[],[],[],[],[],[],[]]
readings_body = [[],[],[],[],[],[],[],[],[]]
readings_right = [[],[],[],[],[],[],[],[],[]]

data_iteration = 0
initialized = False
gesture_lists = [[],[],[]]

last_strike_time = time.perf_counter()

send_control = False
control_gestures = []
prev_mx = 0
prev_my = 0
prev_mz = 0

gesture_left = ""
gesture_body = ""
gesture_right = ""
trigger_gesture_right = ""
trigger_gesture_left = ""

prev_average_gestures = ["","",""]
new_combo = False
prev_arrays = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]

def realtimeGestureDetection(gesture_array_size, gyro_thresh, checking_interval, sequence_length, model_left, model_right, model_body, model_trigger_left, model_trigger_right):
    global data_iteration
    global trigger_readings_left
    global trigger_readings_right
    global readings_left
    global readings_right
    global readings_body
    global initialized
    global gesture_lists
    global control_gestures
    global send_control
    global last_strike_time
    global prev_mx
    global prev_my
    global prev_mz
    global trigger_gesture_left
    global trigger_gesture_right
    global gesture_left
    global gesture_right
    global gesture_body
    global prev_average_gestures
    global new_combo

    def handle_readings(readings, ax,ay,az,gx, gy,gz,mx,my,mz):
        global prev_arrays

        dmx = mx - prev_mx
        dmy = my - prev_my
        dmz = mz - prev_mz
        
        checkCombosSendMIDI(new_combo=new_combo, gesture_left=gesture_left, gesture_body=gesture_body, gesture_right=gesture_right, trigger_gesture_left=trigger_gesture_left, trigger_gesture_right=trigger_gesture_right, glove= glove, ax=ax,ay=ay,az=az,gx=gx,gy=gy,gz=gz, mx=mx, my=my, mz=mz)
        

        fine_to_save = False

        this_array = [ax,ay,az,gx,gy,gz,mx,my,mz]

        if (glove == "LEFT" and this_array != prev_arrays[0]):
            fine_to_save = True
        if (glove == "RIGHT" and this_array != prev_arrays[1]):
            fine_to_save = True
        if (glove == "BODY" and this_array != prev_arrays[2]):
            fine_to_save = True

        if fine_to_save:
            readings[0].append(ax)
            readings[1].append(ay)
            readings[2].append(az)
            readings[3].append(gx)
            readings[4].append(gy)
            readings[5].append(gz)
            readings[6].append(mx)
            readings[7].append(my)
            readings[8].append(mz)
            if (glove == "LEFT"):
                prev_arrays[0] = [ax,ay,az,gx,gy,gz,mx,my,mz]
            if (glove == "RIGHT"):
                prev_arrays[1] = [ax,ay,az,gx,gy,gz,mx,my,mz]
            if (glove == "BODY"):
                prev_arrays[2] = [ax,ay,az,gx,gy,gz,mx,my,mz]
            if (len(readings[0])>sequence_length):
                readings[0].pop(0)
                readings[1].pop(0)
                readings[2].pop(0)
                readings[3].pop(0)
                readings[4].pop(0)
                readings[5].pop(0)
                readings[6].pop(0)
                readings[7].pop(0)
                readings[8].pop(0)
                
                if (glove != "BODY"):
                    if (strikeDetector(glove, readings[0], readings[3])):
                        sendStrikeMIDI(gesture_left=gesture_left, gesture_body=gesture_body, gesture_right=gesture_right, glove= glove, gx =readings[3][len(readings)-1])

        return readings

    def get_average_gesture(readings,glove, model):
        average_gesture = ""

        data_dict = {
                    COLUMN_NAMES[0]: LABELS_NAMES[0], COLUMN_NAMES[1]: readings[0],
                    COLUMN_NAMES[2]: readings[1], COLUMN_NAMES[3]: readings[2], \
                    COLUMN_NAMES[4]: readings[3], COLUMN_NAMES[5]: readings[4], \
                    COLUMN_NAMES[6]: readings[5], COLUMN_NAMES[7]: readings[6], COLUMN_NAMES[8]: readings[7], COLUMN_NAMES[9]: readings[8]
                    }
        data = pd.DataFrame(data=data_dict)

        if (data[COLUMN_NAMES[4]].mean() < gyro_thresh and data[COLUMN_NAMES[4]].mean() > -gyro_thresh and data[COLUMN_NAMES[5]].mean() < gyro_thresh  and data[COLUMN_NAMES[5]].mean() > -gyro_thresh and data[COLUMN_NAMES[6]].mean() < gyro_thresh and data[COLUMN_NAMES[6]].mean() > -gyro_thresh):
            gesture = "---"            
            average_gesture = "---"


        elif (glove == "LEFT"):
            gesture = get_prediction(model, data, LABELS_NAMES)
            gesture_lists[0].append(gesture)
            while len(gesture_lists[0]) > gesture_array_size:
                gesture_lists[0].pop(0)

            average_gesture = max(set(gesture_lists[0]), key=gesture_lists[0].count)


        elif (glove == "RIGHT"):
            gesture = get_prediction(model, data, LABELS_NAMES)
            gesture_lists[1].append(gesture)
            while len(gesture_lists[1]) > gesture_array_size:
                gesture_lists[1].pop(0)

            average_gesture = max(set(gesture_lists[1]), key=gesture_lists[1].count)

        if (glove == "BODY"):
            gesture = get_prediction(model, data, BODY_GESTURES)
            gesture_lists[2].append(gesture)
            while len(gesture_lists[2]) > gesture_array_size:
                gesture_lists[2].pop(0)
            
            average_gesture = max(set(gesture_lists[2]), key=gesture_lists[2].count)
        return average_gesture

    def get_trigger_gesture(readings, model, glove):
        data_dict = {
                    COLUMN_NAMES[0]: TRIGGER_NAMES[0], COLUMN_NAMES[1]: readings[0],
                    COLUMN_NAMES[2]: readings[1], COLUMN_NAMES[3]: readings[2], \
                    COLUMN_NAMES[4]: readings[3], COLUMN_NAMES[5]: readings[4], \
                    COLUMN_NAMES[6]: readings[5], COLUMN_NAMES[7]: readings[6], COLUMN_NAMES[8]: readings[7], COLUMN_NAMES[9]: readings[8]
                    }
        data = pd.DataFrame(data=data_dict)
        gesture = get_prediction(model, data, TRIGGER_NAMES)

        return gesture



    ax, ay, az, gx, gy, gz, mx, my, mz, button, glove = get_raw_data(serversocket)
    ax = ax/SCALE_FACTOR
    ay = ay/SCALE_FACTOR
    az = az/SCALE_FACTOR
    gx = gx/SCALE_FACTOR
    gy = gy/SCALE_FACTOR
    gz = gz/SCALE_FACTOR
    mx = mx/SCALE_FACTOR
    my = my/SCALE_FACTOR
    mz = mz/SCALE_FACTOR


    if (glove == "RIGHT"):
        if (right_check_var.get() == 1):
            if (button):
                trigger_readings_right[0].append(ax)
                trigger_readings_right[1].append(ay)
                trigger_readings_right[2].append(az)
                trigger_readings_right[3].append(gx)
                trigger_readings_right[4].append(gy)
                trigger_readings_right[5].append(gz)
                trigger_readings_right[6].append(mx)
                trigger_readings_right[7].append(my)
                trigger_readings_right[8].append(mz)
            if (not button):
                if (len(trigger_readings_right[0]) > BUTTON_SEGMENT_TIME_SIZE):
                    new_combo = True
                    trigger_gesture_right = get_trigger_gesture(trigger_readings_right, model_trigger_right, "RIGHT")
                    print("triggered right   " + trigger_gesture_right)
                    trigger_readings_right = [[],[],[],[],[],[],[],[],[]]
                readings_right = handle_readings(readings_right,ax, ay, az, gx, gy, gz, mx, my, mz) 
                new_combo = False
                trigger_gesture_right = ""
       

        if (right_check_var.get() == 0):
            return
    if (glove == "LEFT"):
        if (left_check_var.get() == 1):
            if (button):
                trigger_readings_left[0].append(ax)
                trigger_readings_left[1].append(ay)
                trigger_readings_left[2].append(az)
                trigger_readings_left[3].append(gx)
                trigger_readings_left[4].append(gy)
                trigger_readings_left[5].append(gz)
                trigger_readings_left[6].append(mx)
                trigger_readings_left[7].append(my)
                trigger_readings_left[8].append(mz)
            if (not button):
                if (len(trigger_readings_left[0]) > BUTTON_SEGMENT_TIME_SIZE):
                    new_combo = True
                    trigger_gesture_left = get_trigger_gesture(trigger_readings_left, model_trigger_left, "LEFT")
                    print("triggered left  " + trigger_gesture_left)
                    trigger_readings_left = [[],[],[],[],[],[],[],[],[]]
                readings_left = handle_readings(readings_left,ax, ay, az, gx, gy, gz, mx, my, mz)
                new_combo = False
                trigger_gesture_left = ""


            
        if (left_check_var.get() == 0):
            return

    if (glove == "BODY"):
        if (body_check_var.get() == 1):
            readings_body = handle_readings(readings_body,ax, ay, az, gx, gy, gz, mx, my, mz)
        if (body_check_var.get() == 0):
            return 


    data_iteration = data_iteration + 1

    if (data_iteration >= checking_interval):
        data_iteration=0

        if (left_check_var.get() == 1 and len(readings_left[0]) >= sequence_length):
            gesture_left = get_average_gesture(readings_left, "LEFT", model_left)
        if (body_check_var.get() == 1 and len(readings_body[0]) >= sequence_length):
            gesture_body = get_average_gesture(readings_body, "BODY", model_body)
        if (right_check_var.get() ==1  and len(readings_right[0]) >= sequence_length):
            gesture_right = get_average_gesture(readings_right, "RIGHT", model_right)
        
        if (gesture_left != prev_average_gestures[0] or gesture_right != prev_average_gestures[1] or gesture_body != prev_average_gestures[2]):
            print("LEFT HAND: " + gesture_left + "         " + "BODY: " + gesture_body + "         " + "RIGHT HAND: " + gesture_right)
            new_combo = True
        else:
            new_combo = False

        prev_average_gestures[0] = gesture_left
        prev_average_gestures[1] = gesture_right
        prev_average_gestures[2] = gesture_body


performState = False
def performance(model_left, model_right, model_body, model_trigger_left, model_trigger_right):

    #time.sleep(2)
    #mx_offset, my_offset, mz_offset = getNorthOffset()

     #no delay 
    while True:
        realtimeGestureDetection(gesture_array_size=5, gyro_thresh=0, checking_interval=40, sequence_length=SEGMENT_TIME_SIZE, model_left=model_left, model_right=model_right, model_body=model_body, model_trigger_left=model_trigger_left, model_trigger_right=model_trigger_right)
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                print("key is pressed")
                break
        except:
            break
    '''
    #1ms delay
    realtimeGestureDetection(gesture_array_size=5, gyro_thresh=8/SCALE_FACTOR, checking_interval=30, sequence_length=SEGMENT_TIME_SIZE, model_left=model_left, model_right=model_right)
    if (performState):
        window.after(1, lambda: performance(model_left, model_right)) 
    '''

def startPerformance():
    model_left = load_model(MODEL_PATH_LEFT)
    model_right = load_model(MODEL_PATH_RIGHT)
    model_body = load_model(MODEL_PATH_BODY)
    model_trigger_left = load_model(MODEL_PATH_TRIGGER_LEFT)
    model_trigger_right = load_model(MODEL_PATH_TRIGGER_RIGHT)

    global performState
    performState = True
    performance(model_left, model_right, model_body, model_trigger_left, model_trigger_right)

def stopPerformance():
    global performState
    performState = False

performanceColumnStart = 50

#spacing
space = tk.Label(text="                  ")
space.grid(row=1, column=performanceColumnStart-1)

left_check_var = tk.IntVar()
body_check_var = tk.IntVar()
right_check_var = tk.IntVar()

left_check_var.set(1)
body_check_var.set(1)
right_check_var.set(1)

glove_left_check = tk.Checkbutton(window, text='Left',variable=left_check_var, onvalue=1, offvalue=0)
body_check = tk.Checkbutton(window, text='Body',variable=body_check_var, onvalue=1, offvalue=0)
glove_right_check = tk.Checkbutton(window, text='Right',variable=right_check_var, onvalue=1, offvalue=0)

glove_left_check.grid(row=3, column=performanceColumnStart)
body_check.grid(row=4, column=performanceColumnStart)
glove_right_check.grid(row=5, column=performanceColumnStart)

performButton = tk.Button(text="perform", command=startPerformance)
performButton.grid(row=2,column=performanceColumnStart)

#stopButton = tk.Button(text="stop", command=stopPerformance)
#stopButton.grid(row=4,column=performanceColumnStart)









def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def add_items_to_folder(newpath):
    gesturesText = open(newpath + "/gestures.txt", "w")
    bodyGesturesText = open(newpath + "/BODY_GESTURES.txt", "w")
    GestureCombosText = open(newpath + "/GESTURE_COMBOS.txt", "w")
    triggerGesturesText = open(newpath + "/trigger_gestures.txt", "w")

    for line in LABELS_NAMES:
    # write line to output file
        gesturesText.write(line)
        gesturesText.write("\n")
    gesturesText.close()

    for line in BODY_GESTURES:
    # write line to output file
        bodyGesturesText.write(line)
        bodyGesturesText.write("\n")
    bodyGesturesText.close()

    for line in GESTURE_COMBOS:
    # write line to output file
        GestureCombosText.write(line)
        GestureCombosText
        GestureCombosText.write("\n")
    GestureCombosText.close()    

    for line in TRIGGER_NAMES:
    # write line to output file
        triggerGesturesText.write(line)
        triggerGesturesText
        triggerGesturesText.write("\n")
    triggerGesturesText.close()    


    if not os.path.exists(newpath + '/data_temp'):
        os.makedirs(newpath + '/data_temp')    
        copytree(DATA_TEMP_DIR ,os.path.abspath(newpath + '/data_temp'))

    if not os.path.exists(newpath + '/data'):
        os.makedirs(newpath + '/data')    
        copytree(DATA_DIR ,os.path.abspath(newpath + '/data'))

    if not os.path.exists(newpath + '/data_temp'):
        os.makedirs(newpath + '/data_temp')    
        copytree(TRIGGER_DATA_TEMP_DIR ,os.path.abspath(newpath + '/data_temp'))

    if not os.path.exists(newpath + '/body_data_temp'):
        os.makedirs(newpath + '/body_data_temp')    
        copytree(BODY_DATA_TEMP_DIR ,os.path.abspath(newpath + '/body_data_temp'))

    if not os.path.exists(newpath + '/models'):
        os.makedirs(newpath + '/models')    
        copytree(MODEL_PATH_DIR ,os.path.abspath(newpath + '/models'))


def save_configuration():
    newpath = 'configurations/' + newConfigurationEntry.get()
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    else:
        print("directory name already exists")

    add_items_to_folder(newpath)


newConfigurationEntry = tk.Entry()

newConfigurationEntry.grid(row=7, column=performanceColumnStart)
saveNewButton = tk.Button(text="save new configuration file", command=save_configuration)
saveNewButton.grid(row=7,column=performanceColumnStart + 1)


def delete_folder_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def reload_everything():
    global BODY_GESTURES
    global LABELS_NAMES
    global GESTURE_COMBOS
    global TRIGGER_NAMES
    global gestures
    global bodyGestures
    global gestureCombos
    global trigger_gestures

    with open('BODY_GESTURES.txt', 'r') as filehandle:
        BODY_GESTURES = [gesture.rstrip() for gesture in filehandle.readlines()]

    with open('gestures.txt', 'r') as filehandle:
        LABELS_NAMES = [gesture.rstrip() for gesture in filehandle.readlines()]

    with open('GESTURE_COMBOS.txt', 'r') as filehandle:
        GESTURE_COMBOS = [gesture.rstrip() for gesture in filehandle.readlines()]

    with open('trigger_gestures.txt', 'r') as filehandle:
        TRIGGER_NAMES = [gesture.rstrip() for gesture in filehandle.readlines()]


    gestures = []

    for gestureName in LABELS_NAMES:
        gestures.append(Gesture(gestureName))

    trigger_gestures = []

    for gestureName in TRIGGER_NAMES:
        trigger_gestures.append(Gesture(gestureName))

    bodyGestures = []

    for gestureName in BODY_GESTURES:
        bodyGestures.append(BodyGesture(gestureName))

    
    gestureCombos = updateComboLabels()
    updateLabels()
    updateBODYLabels()
    updateButtonPositions(gloveTrainRowStart)
    updateTRIGGERButtonPositions(gloveTriggerTrainRowStart)
    updateBODYButtonPositions()

def reset_EVERYTHING():
    global gestures
    global bodyGestures
    global gestureCombos
    global trigger_gestures

    delete_folder_contents(os.path.abspath(DATA_TEMP_DIR))
    delete_folder_contents(os.path.abspath(DATA_DIR))
    delete_folder_contents(os.path.abspath(BODY_DATA_TEMP_DIR))
    delete_folder_contents(os.path.abspath(TRIGGER_DATA_TEMP_DIR))

    gesturesText = open("gestures.txt", "w")
    bodyGesturesText = open("BODY_GESTURES.txt", "w")
    GestureCombosText = open("GESTURE_COMBOS.txt", "w")
    GestureCombosText = open("trigger_gestures.txt", "w")

    
    bodyGesturesText.write("---")
    bodyGesturesText.close()
    gesturesText.write("---")
    gesturesText.close()
    GestureCombosText.write("")
    gesturesText.close()

    for gesture in gestures:
        gesture.label.destroy()
        gesture.removeButton.destroy()
        gesture.getDataButton.destroy()
        gesture.amountOfSamples.destroy()

    for gesture in bodyGestures:
        gesture.label.destroy()
        gesture.removeButton.destroy()
        gesture.getDataButton.destroy()
        gesture.amountOfSamples.destroy()

    for gesture in trigger_gestures:
        gesture.label.destroy()
        gesture.removeButton.destroy()
        gesture.getDataButton.destroy()
        gesture.amountOfSamples.destroy()

    for gesture in gestureCombos:
        gesture.gestureLEFT.destroy()
        gesture.gestureBODY.destroy()
        gesture.gestureRIGHT.destroy()
        gesture.button.destroy()
        gesture.remove.destroy()

    reload_everything()

def load_configuration():
    delete_folder_contents(os.path.abspath(DATA_TEMP_DIR))
    delete_folder_contents(os.path.abspath(DATA_DIR))
    delete_folder_contents(os.path.abspath(BODY_DATA_TEMP_DIR))
    delete_folder_contents(os.path.abspath(TRIGGER_DATA_TEMP_DIR))
    delete_folder_contents(os.path.abspath(MODEL_PATH_DIR))

    newpath = "configurations/" + configurationOpts.get()
    
    copytree(os.path.abspath(newpath + '/data_temp'), DATA_TEMP_DIR)
    copytree(os.path.abspath(newpath + '/data'), DATA_DIR)  
    copytree(os.path.abspath(newpath + '/body_data_temp'), BODY_DATA_TEMP_DIR)   
    copytree(os.path.abspath(newpath + '/trigger_data_temp'), TRIGGER_DATA_TEMP_DIR) 
    copytree(os.path.abspath(newpath + '/models'), MODEL_PATH_DIR )

    gesturesTextToCopy = open(newpath + "/gestures.txt", "r")
    triggerGesturesTextToCopy = open(newpath + "/trigger_gestures.txt", "r")
    bodyGesturesTextToCopy = open(newpath + "/BODY_GESTURES.txt", "r")
    GestureCombosTextToCopy = open(newpath + "/GESTURE_COMBOS.txt", "r")

    gesturesText = open('gestures.txt', "w")
    triggerGesturesText = open('trigger_gestures.txt', "w")
    bodyGesturesText = open("BODY_GESTURES.txt", "w")
    GestureCombosText = open("GESTURE_COMBOS.txt", "w")

    for line in gesturesTextToCopy:
    # write line to output file
        gesturesText.write(line)
    gesturesText.close()

    for line in triggerGesturesTextToCopy:
    # write line to output file
        triggerGesturesText.write(line)
    triggerGesturesText.close()

    for line in bodyGesturesTextToCopy:
    # write line to output file
        bodyGesturesText.write(line)
    bodyGesturesText.close()

    for line in GestureCombosTextToCopy:
    # write line to output file
        GestureCombosText.write(line)
    GestureCombosText.close()   

    reload_everything()


loadConfigurationButton = tk.Button(text="load configuration", command=load_configuration)
loadConfigurationButton.grid(row=8,column=performanceColumnStart +1)

configurationOpts = tk.StringVar(window)
configurationOpts.set(CONFIGURATIONS[0])
configurationsDropdown = tk.OptionMenu(window, configurationOpts, *CONFIGURATIONS)
configurationsDropdown.grid(row=8,column=performanceColumnStart)

newConfigurationButton = tk.Button(text="Delete current session", command=reset_EVERYTHING)
newConfigurationButton.grid(row=10,column=performanceColumnStart )

#spacing
space = tk.Label(text="                  ")
space.grid(row=1, column=performanceColumnStart +2)

window.mainloop()    


