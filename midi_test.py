import mido
import time
from mido import Message


msg_on = Message('note_on', note=60)
msg_off = Message('note_off', note=60)


#outport = mido.get_output_names()
#print(outport)

outport = mido.open_output('thebestport 1')

#print(outport)

while 1:
    outport.send(msg_on)
    time.sleep(1)
    outport.send(msg_off)
    time.sleep(1)
