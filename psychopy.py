from psychopy import core, visual, event, data, gui, sound
import random

def demographics():
    #Ask for the participant's age, gender and subject number, record it, save it to a data file
    f = open('takehome_demographics.txt', 'a')
    info = {'Subject' : 00, 'Age' : 18, 'Gender' : ['Male', 'Female', 'Other']}
    dlg = gui.DlgFromDict(dictionary=info, title='BART')
    if dlg.OK:
        subject = info['Subject']
        f.write("{}\t{}\t{}\n".format(subject, info['Age'], info['Gender']))
    else:
        subject = 00
    f.close()
    return subject

def display_instruction(win, foto):
    #diplay the image containing the instructions on the screen, press space to proceed
    instructions = visual.ImageStim(win, image=foto, size = 1.5, units ="height")
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

def probabilitylist(p1,p2,p3):
    #generate a list that one probability for every trial
    #the three probability conditions are represented equally but in which trial they occur is random
    probabilities = []
    for i in range(5):
        probabilities += [p1, p2, p3]
    random.shuffle(probabilities)
    return probabilities



for trial in trials:
    presentBalloon()


def whentopop(probability):
    #generate a list of the length of the given probability
    #the number one marks the
    pop = [1]
    for i in range(probability-1):
        pop = pop + [0]
    random.shuffle(pop)
    return pop

def explode(win):
    #Start explode sequence: 
    #play explode sound, display the explode message, press space to proceed
    inflate = sound.Sound("Explode.wav")
    inflate.play()
    msg = visual.TextStim(win,"Oh no! \n \n \n The balloon exploded!! \n You receive 0 credits \n \n \n press space to proceed", color = "black")
    msg.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

def payout(win,amount):
    #Start payout sequene:
    #play the payout sound, display payout message, press space to proceed
    payout = sound.Sound("Payout.wav")
    payout.play()
    msg ="Payout: \n \n  " +  str(amount) + " credits will be added to your balance \n \n \n Press space to continue"
    payout = visual.TextStim(win,msg,color="black")
    payout.draw()
    win.flip()
    event.waitKeys(keyList=["space"])


def presentBalloon(win, foto, sizew,sizeh, probability):

    keys = {'bigger': 'b', 'payout': 'p'} 
    biggerReminder = visual.TextStim(win, '{} = bigger'.format(keys['bigger']), pos=(-0.7, -0.85),color="black")
    payoutReminder = visual.TextStim(win, '{} = payout'.format(keys['payout']), pos=(0.7, -0.85),color="black")
    biggerReminder.setAutoDraw(True)
    payoutReminder.setAutoDraw(True)
    
    balloon = visual.ImageStim(win, image=foto, size = (sizew,sizeh), units ="height", pos = (0,-0.2))
    balloon.draw()
    win.flip()
    inflate = sound.Sound("Inflate2.wav")

    respond = event.waitKeys(keyList=["b","p"])

    pumps = 0
    pop = whentopop(probability)

    for trial in pop:
        if respond == ["b"] and trial == 1:
            pumps = pumps + 1
            outcome = "explode"
            return outcome, pumps
        elif respond == ["b"] and trial == 0:
            pumps = pumps + 1
            inflate.play()
            balloon.size = (sizew + pumps*0.03, sizeh + pumps*0.05)
            balloon.draw()
            win.flip()
            core.wait(0.6)
            respond = event.waitKeys(keyList=["b","p"])
        else:
            outcome = "payout"
            return outcome, pumps
            
    biggerReminder.setAutoDraw(False)
    payoutReminder.setAutoDraw(False)
    
    return pumps, outcome


def writeData(dataList, file):
    output = ""
    for d in dataList[:-1]:
        output += "{}\t".format(d)
    output += "{}\n".format(dataList[-1])
        
    f = open("data.txt", 'a')
    f.write(output.format(dataList))
    f.close()


def BART(subject):
    win = visual.Window()
    win.color = (1,1,1)
    clock = core.Clock()

    instructions = ["welcome.jpg", "instructions1.jpg", "instructions2.jpg"]
    for instruction in instructions:
        display_instruction(win, instruction)


    conditions = []   
    probabilities = [128,32,8]
    for prob in probabilities:
        conditions.append({'name' : prob})
        trials = data.TrialHandler(conditions, 5) #create the trial handler
    #trials = probabilitylist(128,32,8)
    
    for trial in trials:
        outcome, pumps = presentBalloon(win,"ball.jpg",0.2,0.3,trial)
        if outcome == "explode":
            explode(win)
        if outcome == "payout":
            payout(win,pumps)
        dataList = [subject, pumps, outcome]
        writeData(dataList, 'BART_data.txt')

    win.close() 


def main():
    subject = demographics()
    BART(subject)

main()


