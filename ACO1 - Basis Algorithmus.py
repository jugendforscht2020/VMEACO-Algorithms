#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@author: Karl Scholz and Lewin Scholz
@date: 22.11.2019 - 31.1.2020
@note:  This is an Ant Colony Optimization Algorithm. The idea is to find the fastest way between a
        couple of points printed in a coordinate system using the collective intelligence ant colonies
        usually work with. This is the basic algorithm which is enhanced and developed farther in
        different programs.
        This program runs best in the Anaconda Spyder IDE.
'''

# import libraries:
import random as rn # used for choosing random numbers
import numpy as np # used for matricies and more dimensional arrays
import math # used for complex mathematic functions
import matplotlib.pyplot as plt # used for plotting the results graphically
import seaborn as sns # used for printing the heatmap

# main class: Ant Colony Optimization Algorithm
class ACO():
    
    # function used to initialize public variables
    def __init__(self, nodes, iterations, learning_rate):
        self.nodes = nodes
        self.iterations = iterations
        self.learning_rate = learning_rate

    # function for printing the results using switch parameter
    def show(self, switch):
        # parameter if program started the training
        if switch == "train":
            print("Begin training...")
        # parameter if program finished the training
        if switch == "done":
            print("Done training...")
        # parameter if network statistics are questioned
        if switch == "stats":
            print("Network statistics:")
            print("The network consists of", self.nodes, "nodes.")
            print("Coordinates of the nodes:")
            print(self.coordinates)
            # save data in two new variables so it is not affected by the rounding
            show_distances = self.distances
            show_pheromone = self.pheromone
            # round the distances between the nodes to two digits behind the comma
            for i in range(self.nodes):
                for n in range(self.nodes):
                    show_distances[i][n] = format(show_distances[i][n], '.2f')
            print("Distances: \n", show_distances)
            # round the pheromone trails between the nodes to two digits behind the comma
            for i in range(self.nodes):
                for n in range(self.nodes):
                    show_pheromone[i][n] = format(show_pheromone[i][n], '.2f')
            print("Pheromone trails: \n", show_pheromone)
        # parameter for plotting diagrams and graphic output
        if switch == "plot":
            # start function to prepare data for plotting
            self.prepare_graphic_plot()
            # print out a matplotlib.pyplot graphic 
            # with a cartesian coordinate system, the nodes and their connections
            plt.plot(self.line_x_coordinates, self.line_y_coordinates, marker='o', color='grey')
            plt.axis([-25, 25, -25, 25]) # define the scale
            plt.grid(True) # show a grid for better optical measurements
            plt.show() # show first graphic
            # print out a heatmap of the pheromone matrix
            sns.heatmap(self.pheromone, annot=True, cbar=False)
            plt.show() # show second graphic
        # parameter if program should finish
        if switch == "result":
            # output the algorithms solution for the most effective way
            print("The best way is: ", self.best_way)
            print("The total distance travelled is:", format(self.total_distance, '.2f'), ", ", self.total_distance)

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
    
    # function to process coordinates for printing them with matplotlib
    def prepare_graphic_plot(self):
        # uses complex mathematic pattern to calculate the values for printing the diagram
        # hint: its not important to understand the following process completely since its
        #       just a mathematic conversion of the coordinates to plot them in the heatmap
        # general process: - create two lists with the x and y coordinates for the lines connecting the nodes
        #                  - format the coordinate matrix into two arrays for matplotlib
        x=[]
        y=[]
        for i in range(self.nodes):
            x.append(self.coordinates[i][0])
            y.append(self.coordinates[i][1])
        lx = []
        ly = []
        lx2 = []
        ly2 = []
        counter = self.nodes-1
        for i in range(counter):
            for n in range(counter):
                lx.append(x[i])
                ly.append(y[i])
            counter = counter-1
        counter = self.nodes-1
        for i in range(1,counter):
            lx2.extend(x[i:])
            ly2.extend(y[i:])
        lx2.append(x[-1])
        ly2.append(y[-1])
        self.line_x_coordinates = [None]*(len(lx)+len(lx2))
        self.line_y_coordinates = [None]*(len(ly)+len(ly2))
        self.line_x_coordinates[1::2]=lx2
        self.line_x_coordinates[::2]=lx
        self.line_y_coordinates[1::2]=ly2
        self.line_y_coordinates[::2]=ly
    
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
        self.show("train")
        repeat = 0 # counting variable
        # train the algorithm
        # every repetition conforms to one agent
        while repeat < self.iterations:
            # optional parameter for weighting the decision process
            # if alpha, beta = 1 : no weighting
            alpha = 1
            beta = 5
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
        self.show("done")
    

# the program starts here
print("Program started...") # show that the program started as expected
# repeat the program a number of times
for i in range(1):
    # set important variables
    nodes = 10 # number of nodes in the network
    learning_rate = 0.995 # the decay rate of the training process for the pheromone
    training_iterations = 300 # how often does the algorithm has to train
    # create new instance of the ACO
    aco = ACO(nodes, training_iterations, learning_rate)
    # call preparation functions
    aco.init_coordinates() # initiate a random network
    aco.distances_and_pheromone() # calculate its distances and pheromone values
    # train the algorithm
    aco.train()
    # calculate the results
    aco.determine_chosen_path() # find the most traveled path
    total_distance = aco.calculate_travelled_distance() # determine its length
    # show what the program did
    aco.show("stats")
    aco.show("plot")
    aco.show("result")
print("Program finished.")