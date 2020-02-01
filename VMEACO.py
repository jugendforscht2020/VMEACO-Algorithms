#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@author: K. and L. Scholz
@date: 22.11.2019 - 31.1.2020
@note:  This is an Vector Enhanced Ant Colony Optimization Algorithm. The idea is to find the fastest way between a
        couple of points printed in a coordinate system using the collective intelligence ant colonies
        usually work with combined with vector memory of previous problems. The algorithm is therefore able to learn
        solving similar problems faster.
        This program runs best in the Anaconda Spyder IDE.
'''
# import libraries:
import random as rn # used for choosing random numbers
import numpy as np # used for matricies and more dimensional arrays
import math # used for complex mathematic functions
import matplotlib.pyplot as plt # used for plotting the results graphically
import os
import re

# class: Vector Memory Enhanced Ant Colony Optimization Algorithm
class VMEACO():
    
    # function used to initialize public variables
    def __init__(self, nodes, iterations, learning_rate):
        self.nodes = nodes
        self.iterations = iterations
        self.learning_rate = learning_rate
    
    # function to initialize coordinates of the nodes
    def init_coordinates(self):
        # there are two coordinates for each node
        self.coordinates = np.arange(self.nodes*2)
        # choose random numbers for the coordinates between -25 and 25
        for i in self.coordinates:
            self.coordinates[i] = rn.randint(-25,25)
        # call function to make sure all nodes are different
        self.verify_coordinates()
        # convert the coordinate list into a two dimensional array
        # - columns : x and y coordinates
        # - lines: nodes
        self.coordinates = self.coordinates.reshape(self.nodes, 2)
        return self.coordinates
        
    # function to test if all nodes have different coordinates - prevent errors
    def verify_coordinates(self):
        coordinate_list = [] # list for coordinate sets
        # save every x,y coordinate pair in the new list
        # repeat number of node times
        for i in range(0, len(self.coordinates), 2): # jump to every second element
            coordinate_list.append(tuple([self.coordinates[i],self.coordinates[i+1]])) # add element
        # detect doubled element by comparing the original coordinates against their set
        # the set eliminates doubled elements and is therefore shorter
        if len(coordinate_list) != len(set(coordinate_list)):
            self.init_coordinates() # redo the coordinate creation task
    
    # function to create the distances and the pheromone matrices
    def distances_and_pheromone(self, new_coordinates):
        self.coordinates = new_coordinates
        # create an empty two dimensional array square the size of the nodes
        self.distances = np.zeros(self.nodes**2)
        # calculate the displacements between the nodes using pythagoras theorem
        counter = 0 # varable used for representing the index
        for i in range(self.nodes): # for the first node
            for n in range(self.nodes): # for the second node
                # calculate the change of the x-values with subtraction
                x_change = abs(self.coordinates[i][0]-self.coordinates[n][0])
                # calculate the change of the y-values with subtraction
                y_change = abs(self.coordinates[i][1]-self.coordinates[n][1])
                # calculate the distance between the nodes using pythagoras theorem
                # save every distance in the distances matrix
                self.distances[counter] = math.sqrt(x_change**2 + y_change**2)
                # continue with the next distance
                counter += 1
        # convert distance array into two dimensional array
        self.distances = self.distances.reshape(self.nodes, self.nodes)
        # create two dimensional array with the same size as the distances matrix
        self.pheromone =  np.ones(self.distances.shape) / len(self.distances)
        # delete every pheromone value referring to one node 
        for x in range(self.nodes):
            self.pheromone[x][x]=0

    # function to determine the chosen way between the nodes from the pheromone matrix
    def determine_chosen_path(self):
        # starting node is always node 0
        self.best_way = [0] # list for the chosen way between the nodes
        next_nodes = [] # list for next possible nodes
        ctn = 0 # current test node
        # list all possible next nodes
        counter = 0
        for i in range(self.nodes):
            next_nodes.append(counter)
            counter += 1
        # delete already visited nodes saved in best_way
        next_nodes = list(set(next_nodes).difference(self.best_way))
        # choose the next node
        for n in range(self.nodes-1): # this process just needs to run number of nodes -1 times
            # find the biggest value in the given pheromone map
            max_value = max(self.pheromone[ctn][next_nodes])
            # now find the position representing the node
            for i in range(self.nodes):
                if self.pheromone[ctn][i] == max_value:
                    ctn = i # current test node changes to the next node with the highest pheromone
                    break # make sure to leave this loop because ctn is changed now
            # add the next node to the list
            self.best_way.append(ctn)
            # delete already visited nodes saved in best_way
            # otherwise it will bounce between two nodes with high pheromone
            next_nodes = list(set(next_nodes).difference(self.best_way))
        #sns.heatmap(self.pheromone,annot=True,cbar=False)
        #plt.show() # show second graphic
        return self.best_way
    
    # function to calculate the total distance of the travelled way
    def calculate_travelled_distance(self):
        self.total_distance = 0 # the total distance travelled
        n1 = 0 # node
        for n2 in self.best_way[1:]:
            self.total_distance += self.distances[n1][n2] # add the chosen distance connection
            n1=n2
        self.total_distance += self.distances[n2][0]
        return self.total_distance
    
    def create_vector(self, ctn, n, vector_memory):
        # create empty vector variable
        vector = None
        # calculate x and y differences between the following nodes
        # every quadrant receives and extra vector score for the 360 degree direction
        x_change = self.coordinates[n][0] - self.coordinates[ctn][0]
        y_change = self.coordinates[n][1] - self.coordinates[ctn][1]
        # calculate specific vector for every coordinate quadrant
        if x_change < 0 and y_change < 0: # quadrant 3
            vector = math.degrees(math.atan(abs(x_change/y_change))) + 180
        if x_change > 0 and y_change < 0: # quadrant 4
            vector = math.degrees(math.atan(abs(y_change/x_change))) + 90
        if x_change < 0 and y_change > 0: # quadrant 2
            vector = math.degrees(math.atan(abs(y_change/x_change))) + 270
        if x_change > 0 and y_change > 0: # quadrant 1
            vector = math.degrees(math.atan(abs(x_change/y_change)))
        # take care of the exceptions if the x or y difference is zero
        if x_change == 0 and y_change == 0:
            vector = None # there is no vector for twice the same point in space
        if x_change == 0 and y_change < 0:
            vector = 180 # the next point is just below
        if x_change == 0 and y_change > 0:
            vector = 0 # the next point is just above
        if x_change < 0 and y_change == 0:
            vector = 270 # the next point is to the left
        if x_change > 0 and y_change == 0:
            vector = 90 # the next point is to the right
        # calculate difference between the new vector and the memorized vector
        vector_difference = abs(vector - vector_memory)# + 0.001 # make sure it isnt zero ????
        # make sure the difference is below 180 degrees
        if vector_difference > 180:
            vector_difference = abs(vector_difference - 360)
            vector_score = 1/vector_difference # calculate the vector score
        if vector_difference == 0:
            vector_score = 100 # calculate the vector score
        else:
            vector_score =  1/vector_difference # calculate the vector score
        return vector_score
    
    # main function for vector training - train and execute algorithm using vector memory
    def vector_train(self, vector_memory, best_path, iterations):
        repeat = 0
        # train the algorithm
        # every repetition conforms to one agent
        while repeat < iterations:
            # optional parameter for weighting the decision process
            # if alpha, beta = 1 : no weighting
            alpha = 1
            beta = 5
            # always start at node 0
            # only include vector memory on the first quarter of the iterations
            if repeat < round(iterations/4):   
                ctn = best_path[0] # current test node
            # if the whole training iterations are quite low, always use vector memory
            elif iterations <= 5:
                ctn = best_path[0] # current test node
            else:
                ctn = rn.randint(0,9) # ignore vector memory
            node_hist = [ctn] # list for visited nodes         
            # run through the network as long as unused nodes are existing
            while len(node_hist) <= self.nodes-1:
                counter = 0 # counter variable
                score_list = [] # list to save scores between the nodes
                prob_list = [] # list to save probabilities of node connections
                next_nodes = [] # list to save possible next nodes
                # eliminate connections from nodes to themselfes
                for i in self.distances[ctn]: # repeat for every other node
                    # if connection exists add the refering node to the next possible nodes
                    if i > 0:
                        next_nodes.append(counter)
                    # jump to the next node
                    counter += 1
                # delete already visited nodes from possible next nodes
                next_nodes = list(set(next_nodes).difference(node_hist))
                #print("next nodes:", next_nodes)
                # calculate connection scores
                for n in next_nodes: # repeat for every possible connection
                    # the first time the algorithm runs use memory to choose the next node
                    vector_score = self.create_vector(ctn, n, vector_memory[len(node_hist)-1])
                    # only include vector memory on the first quarter of the iterations
                    if repeat < round(iterations/4):
                        influence = 1.5 # 1.5 is a quite good value found by testing
                    # if the whole training iterations are quite low, always use vector memory
                    elif iterations <= 5:
                        influence = 1.5 # 1.5 is a quite good value found by testing
                    else:
                        influence = 0 # ignore vector memory
                    # calculate the node to node score
                    score = (vector_score**influence)*(((self.pheromone[ctn][n])**alpha) / (self.distances[ctn][n]**beta))
                    score_list.append(score) # add score to score list
                # convert connection scores to connection probabilities
                for i in score_list: # repeat for every score
                    # using probability formula: probability = score(i) / sum_of_all_scores
                    prob_list.append(i/sum(score_list))
                # choose the one next node randomly appropriate to their probabilities
                move = int(np.random.choice(next_nodes, 1 ,p=prob_list))
                # update pheromone map after which way was chosen
                self.pheromone[ctn][move] += 1/(self.distances[ctn][move]) # fist matrix element
                self.pheromone[move][ctn] += 1/(self.distances[ctn][move]) # mirror of first element
                # old pheromone decays slowly
                self.pheromone = self.pheromone * self.learning_rate
                # determine the next node
                ctn = move
                # add this node to the visited nodes
                node_hist.append(ctn)
            repeat += 1

     # main function - train and execute algorithm
    def train(self, iterations):
        repeat = 0  # counting variable
        # train the algorithm
        # every repetition conforms to one agent
        while repeat < iterations:
            # optional parameter for weighting the decision process
            # if alpha, beta = 1 : no weighting
            alpha = 1
            beta = 5
            # always start at a random node in the network
            ctn = rn.randint(0,9)# # current test node
            node_hist = [ctn] # list for visited nodes         
            # run through the network as long as unused nodes are existing
            while len(node_hist) <= self.nodes-1:
                counter = 0 # counter variable
                score_list = [] # list to save scores between the nodes
                prob_list = [] # list to save probabilities of node connections
                next_nodes = [] # list to save possible next nodes
                # eliminate connections from nodes to itself
                for i in self.distances[ctn]: # repeat for every other node
                    # if connection exists add the refering node to the next possible nodes
                    if i > 0: 
                        next_nodes.append(counter)
                    # jump to the next node
                    counter += 1
                # delete already visited nodes from possible next nodes
                next_nodes = list(set(next_nodes).difference(node_hist))
                # calculate connection scores
                for n in next_nodes: # repeat for every possible connection
                    # calculate the node to node score
                    score = ((self.pheromone[ctn][n])**alpha) / (self.distances[ctn][n]**beta)
                    score_list.append(score) # add score to score list
                # convert connection scores to connection probabilities
                for i in score_list: # repeat for every score
                    # using probability formula: probability = score(i) / sum_of_all_scores
                    prob_list.append(i/sum(score_list))
                # choose the one next node randomly appropriate to their probabilities
                move = int(np.random.choice(next_nodes, 1 ,p=prob_list))
                # update pheromone map after which way was chosen
                self.pheromone[ctn][move] += 1/(self.distances[ctn][move]) # fist matrix element
                self.pheromone[move][ctn] += 1/(self.distances[ctn][move]) # mirror of first element
                # old pheromone decays slowly
                self.pheromone = self.pheromone * self.learning_rate
                # determine the next node
                ctn = move
                # add this node to the visited nodes
                node_hist.append(ctn)
            repeat += 1 # increment the training iterations
    
    # function to calculate vector directions between the final nodes
    def vector_memory(self):
        vector_memory = [] # list to save vector directions
        n1 = self.best_way[0] # choose first node of solution list
        for n2 in self.best_way[1:]:
            # create empty vector variable
            vector = None
            # calculate x and y differences between the following nodes
            # every quadrant receives and extra vector score for the 360 degree direction
            x_change = self.coordinates[n2][0] - self.coordinates[n1][0]
            y_change = self.coordinates[n2][1] - self.coordinates[n1][1]
            # calculate specific vector for every coordinate quadrant
            # because the degrees go clockwise:
            # 2nd and 4th quadrant are calculates y/x change
            # 1st and 3rd quadrant are calculates x/y change
            if x_change < 0 and y_change < 0: # quadrant 3
                vector = math.degrees(math.atan(abs(x_change/y_change))) + 180
            if x_change > 0 and y_change < 0: # quadrant 4
                vector = math.degrees(math.atan(abs(y_change/x_change))) + 90
            if x_change < 0 and y_change > 0: # quadrant 2
                vector = math.degrees(math.atan(abs(y_change/x_change))) + 270
            if x_change > 0 and y_change > 0: # quadrant 1
                vector = math.degrees(math.atan(abs(x_change/y_change)))
            # take care of the exceptions if the x or y difference is zero
            if x_change == 0 and y_change == 0:
                vector = None # there is no vector for twice the same point in space
            if x_change == 0 and y_change < 0:
                vector = 180 # the next point is just below
            if x_change == 0 and y_change > 0:
                vector = 0 # the next point is just above
            if x_change < 0 and y_change == 0:
                vector = 270 # the next point is to the left
            if x_change > 0 and y_change == 0:
                vector = 90 # the next point is to the right
            # add the vector to the memory
            vector_memory.append(vector)# = [vector] + vector_memory
            # jump to the next node to node connection
            n1=n2
        return vector_memory

# function to print a progress bar while execution
def progress_bar (current_iteration, total_iterations, length):
    decimals = 1
    printEnd = "\r"
    prefix = "Progress:"
    suffix = "Complete"
    fill = '█'
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current_iteration / float(total_iterations)))
    filledLength = int(length * current_iteration // total_iterations)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)

# function to read data from a file
def read_data(data_path, test_data_number):
    path_of_file = data_path + test_data_number
    # open the file
    file = open(path_of_file, "r")
    # read data from file
    file_total_distance = file.readline() # fist line is the solution
    # use regex to seperate the number from the rest of the line
    file_total_distance = re.findall('[0-9]+.[0-9]+', file_total_distance)
    # convert regex string to float
    file_total_distance = float(file_total_distance[0])
    # convert coordinate list from string to integer
    file_coordinates = [] # list for saving the coordinates from the file
    file_coordinates = re.findall('.[0-9]+', file.read())
    for element in range(len(file_coordinates)):
        file_coordinates[element] = int(file_coordinates[element])
    # change the dimensions to two dimensional array
    file_coordinates = np.array(file_coordinates).reshape(nodes, 2)
    file.close() # close file after finished reading
    return file_coordinates, file_total_distance

print("Program started...")
# important variables
total_iterations = 100 # how often should the program run
nodes = 10 # number of nodes in the network
learning_rate = 0.995 # rate at wich the pheromone decays per round
training_iterations = 200 # training rate of the ACO
data_path = "C:/ENTER_DATA_PATH_HERE" # choose a test data set


# create a new vector memory enhanced aco instance
vmeaco = VMEACO(nodes, training_iterations, learning_rate)
coordinates, total_distance1 = read_data(data_path, "data1.txt")
# call preparation functions
vmeaco.distances_and_pheromone(coordinates) # calculate the distances and the pheromone of the network
# train the algorithm and complete
vmeaco.train(training_iterations)
best_path1 = vmeaco.determine_chosen_path()
total_distance2 = vmeaco.calculate_travelled_distance()
# create the vector memory
vector_memory = vmeaco.vector_memory()
if total_distance1 == total_distance2:
    print("Vector Memory successfully created")
else:
    print("Failure creating Vector Memory")

# Initial call to print 0% progress
progress = 0
progress_bar(progress, total_iterations, 50)

# create score counting variables
score_list_vmeaco = []
score_list_aco = []
repetitions_list = []

while progress < total_iterations:
    score_vmeaco = 0
    score_aco = 0
    # repeat the program x times
    for file in os.listdir(data_path):
        # read the data from the file
        coordinates, total_distance1 = read_data(data_path, file)
        # calculate the distances and the pheromone of the network
        
        # reset the pheromone map
        vmeaco.distances_and_pheromone(coordinates)
        # train the algorithm using the vector memory and complete
        vmeaco.vector_train(vector_memory, best_path1, 1) # last parameter: training iterations
        best_path2 = vmeaco.determine_chosen_path()
        total_distance2 = vmeaco.calculate_travelled_distance()
        if total_distance2 == total_distance1:
            score_vmeaco +=1
        
        # reset the pheromone map
        vmeaco.distances_and_pheromone(coordinates)
        # train the algorithm and complete with a given number of training iterations
        vmeaco.train(1)
        best_path3 = vmeaco.determine_chosen_path()
        total_distance3 = vmeaco.calculate_travelled_distance()
        if total_distance3 == total_distance1:
            score_aco +=1
    # update the scores
    score_list_aco.append(score_aco)
    score_list_vmeaco.append(score_vmeaco)
    repetitions_list.append(progress)
    # continue printing the progress bar
    progress_bar(progress+1, total_iterations, 50)
    progress += 1

# plot a diagram to compare the two algorithms
print()
print(data_path)
print("score aco    :", sum(score_list_aco), "/", total_iterations*100)
print("score vmeaco :", sum(score_list_vmeaco), "/", total_iterations*100)
fig, ax = plt.subplots(figsize=(7,7))
plt.plot(repetitions_list, score_list_vmeaco,  color = "green")
plt.plot(repetitions_list, score_list_aco, color = "red")
plt.xlabel("training iterations")
plt.ylabel("score")
plt.show()

print("Program finished")
