    
    
    
    
    conditions = []                                     # Import condition using Python
    for colorname in keys:
        for colorink in keys:
            conditions.append({'name' : colorname,
                               'ink' : colorink})



    #draw the reminders when starting the task
    blueReminder.setAutoDraw(True)
    redReminder.setAutoDraw(True)
    greenReminder.setAutoDraw(True)
    orangeReminder.setAutoDraw(True)

    trials = data.TrialHandler(conditions, 2) #create the trial handler
    print(trials)
    win.close() # THE END 



    for trial in trials:
        presentColorWord(win, stim, trial['name'], trial['ink'])
        
        #wait for a response and record latency
        clock.reset()
        respond = event.waitKeys(keyList=keys.values(), timeStamped=clock)
        response, latency = respond[0]
        correct = (response == keys[trial['ink']])  # check whether the key pressed matches the dictionary value of the inkcolor, as defined above ^^
     
        if not correct:
            errorFeedback(win)
            
        # write all the data
        dataList = [subject, trials.thisN, trial['ink'], trial['name'], response, latency, correct]
        writeData(dataList, 'data.txt')


def main():
    subject = demographics()
    stroopTask(subject)


main()
