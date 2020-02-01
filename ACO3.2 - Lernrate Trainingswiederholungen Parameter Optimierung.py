#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@author: K. and L. Scholz
@date: 22.11.2019 - 31.1.2020
@note:  This is an Ant Colony Optimization Algorithm. It finds the best relationship 
        between the parameter for training the algorithm using a three dimensional data plotting heatmap.
        This program runs best in the Anaconda Spyder IDE.
'''

# import libraries:
import random as rn # used for choosing random numbers
import numpy as np # used for matricies and more dimensional arrays
import math # used for complex mathematic functions
import matplotlib.pyplot as plt # used for plotting the results graphically
import seaborn as sns # used for printing the heatmap
import re
import os

# main class: Ant Colony Optimization Algorithm
class ACO():
    
    # function used to initialize public variables
    def __init__(self, nodes, iterations, learning_rate, file_coordinates):
        self.nodes = nodes
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.coordinates = file_coordinates

    # function to create the distances and the pheromone matrices
    def distances_and_pheromone(self):
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
    
    # function to calculate the total distance of the travelled way
    def calculate_travelled_distance(self):
        self.total_distance = 0 # the total distance travelled
        n1 = 0 # node
        for n2 in self.best_way[1:]:
            self.total_distance += self.distances[n1][n2] # add the chosen distance connection
            n1=n2
        # do not forget the last connection to the start node
        self.total_distance += self.distances[n2][0]
        return self.total_distance
        
    # main function - train and execute algorithm
    def train(self):
        repeat = 0 # counting variable
        # train the algorithm
        # set parameters
        alpha = 1
        beta = 5
        # every repetition conforms to one agent
        while repeat < self.iterations:
            # always start at a random node in the network
            ctn = rn.randint(0,9) # ctn stands for current test node
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

# function to read data from the file
def read_data(data_path, test_data_number):
    # generate the path of every file individually
    path_of_file = data_path + "data" + str(test_data_number) + ".txt"
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
    return file_total_distance, file_coordinates

print("Program started...")
# set important variables
nodes = 10 # number of nodes in the network
data_path = "C:/ENTER_DATA_PATH_HERE" # path where the data is saved in text files

# create two dimensional array later used to represent relationship between pheromone and iterations
scores = np.zeros((10, 10))

run = 0 # variable to count the runs

# prepare axis numeration for matplotlib
decay_rate_list = []
iterations_list = []
for i in range(5):
    decay_rate_list.append(99 + i*0.2)
    iterations_list.append(i*150)

# train the ACO using different parameters alpha and beta each time
for l_rate in range(10):
    learning_rate = 0.99 + l_rate*0.1 # set learning rate parameter value
    for iterations in range(10):
        training_iterations = iterations * 75 # set iterations parameter value
        test_data_number = 1 # indicates number of the test data file
        score = 0 # score used to compare test data with ACOs results
        # repeat for every test data file found in the directory
        # compare ACOs result with the saved one
        for files in os.listdir(data_path):
            # read the data from the file
            file_total_distance, file_coordinates = read_data(data_path, test_data_number)
            # create new instance of the ACO with specified parameters
            aco = ACO(nodes, training_iterations, learning_rate, file_coordinates)
            # calculate distances and pheromone according to the given coordinates
            aco.distances_and_pheromone()
            # train the algorithm - pheromone map changes
            aco.train()
            # calculate results - optimal path
            aco.determine_chosen_path()
            # calculate ACOs solution
            total_distance = aco.calculate_travelled_distance()
            # check if ACO object calculated the right distance
            if total_distance == file_total_distance:
                score += 1 # if yes, increment the score
            # jump to the next test data set
            test_data_number += 1
        # saved the result in the corresponding score position
        scores[l_rate][iterations]=score
        # plot the heatmap of the scores after each row is done
        fig, ax = plt.subplots(figsize=(7,7))
        sns.heatmap(scores, annot=True, cbar=True, vmin=0, vmax=100)
        plt.ylabel("learning rate")
        plt.xlabel("training iterations")
        plt.xticks([0,2,4,6,8,10], decay_rate_list)
        plt.yticks([0,2,4,6,8,10], iterations_list)
        plt.show()
        run += 1
        print(run, "/100 done")

# show final heatmap of the scores
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(scores, annot=True, cbar=True, vmin=0, vmax=100)
plt.ylabel("learning rate")
plt.xlabel("training iterations")
plt.xticks([0,2,4,6,8,10], decay_rate_list)
plt.yticks([0,2,4,6,8,10], iterations_list)
plt.show() # show graphic
print("Program finished.")
