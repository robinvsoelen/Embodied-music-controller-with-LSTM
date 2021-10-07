import matplotlib.pyplot as plt
import matplotlib.animation as animation
from predict_realtime import collect_data_realtime, get_prediction, get_raw_data_realtime
import socket
from config import *
import time
import math
from collections import deque , defaultdict
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import threading
from random import randint
from statistics import *


serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
host = IP_ADDRESS
port = 8888
serversocket.bind((host, port))
print ('server started and listening')

class DataPlot:
    def __init__(self, max_entries = 200):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
        self.axis_y2 = deque(maxlen=max_entries)

        self.max_entries = max_entries

        self.buf1=deque(maxlen=5)
        self.buf2=deque(maxlen=5)

     
    def add(self, x, y,y2):
        self.axis_x.append(x)
        self.axis_y.append(y)
        self.axis_y2.append(y2)

class RealtimePlot:
    def __init__(self, axes):
        self.axes = axes

        self.lineplot, = axes.plot([], [], "ro-")
        self.lineplot2, = axes.plot([], [], "go-")

    def plot(self, dataPlot):
        self.lineplot.set_data(dataPlot.axis_x, dataPlot.axis_y)
        self.lineplot2.set_data(dataPlot.axis_x, dataPlot.axis_y2)

        self.axes.set_xlim(min(dataPlot.axis_x), max(dataPlot.axis_x))
        ymin = min([min(dataPlot.axis_y), min(dataPlot.axis_y2)])-10
        ymax = max([max(dataPlot.axis_y), max(dataPlot.axis_y2)])+10
        self.axes.set_ylim(ymin,ymax)
        self.axes.relim()

def main():
    fig, axes = plt.subplots()
    plt.title('Plotting Data')

    data = DataPlot()
    dataPlotting= RealtimePlot(axes)

    try:
        count=0
        while True:    
            ax, ay, az, gx, gy, gz, glove = get_raw_data_realtime(serversocket)
            count+=1
            data.add(count, gx ,gy)
            dataPlotting.plot(data)

            plt.pause(0.0000001)
    except KeyboardInterrupt:
        print('nnKeyboard exception received. Exiting.')
        plt.close()
        exit()

if __name__ == "__main__": main()