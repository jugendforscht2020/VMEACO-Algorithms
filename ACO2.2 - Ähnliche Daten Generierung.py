#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@author: K. and L. Scholz
@date: 22.11.2019 - 31.1.2020
@note:  This is an Ant Colony Optimization Algorithm program. It generates and solves similar networks
        and saves them in a directory in order for other programs to work with it.
        This program runs best in the Anaconda Spyder IDE.
'''

# import libraries:
import random as rn # used for choosing random numbers
import numpy as np # used for matricies and more dimensional arrays
import math # used for complex mathematic functions
import os # used to create/remove directories
import glob # used for file managing


# main class: Ant Colony Optimization Algorithm
class ACO():
    
    # function used to initialize public variables using switch parameter
    def __init__(self, nodes, iterations, learning_rate):
        self.nodes = nodes
        self.iterations = iterations
        self.learning_rate = learning_rate

    # function to initialize coordinates of the nodes
    def init_coordinates(self):
        # there are two coordinates for each node
        # use function to generate an original version of the network
        self.original_coordinates = np.arange(self.nodes*2)
        # choose random numbers for the coordinates between -25 and 25
        for i in self.original_coordinates:
            self.original_coordinates[i] = rn.randint(-25,25)
        # call function to make sure all nodes are different 
        # give parameter so it knows what to continue afterwards
        self.verify_coordinates("ic", self.original_coordinates)
        # convert the coordinate list into a two dimensional array
        # - columns : x and y coordinates
        # - lines: nodes
        self.original_coordinates = self.original_coordinates.reshape(self.nodes, 2)
          
    # function to test if all nodes have different coordinates - prevent errors
    def verify_coordinates(self, switch, coordinates):
        coordinate_list = [] # list for coordinate sets
        # save every x,y coordinate pair in the new list
        # repeat number of node times
        for i in range(0, len(coordinates), 2): # jump to every second element
            coordinate_list.append(tuple([coordinates[i],coordinates[i+1]])) # add element
        # detect doubled element by comparing the original coordinates against their new set
        # the set eliminates doubled elements and is therefore shorter
        if len(coordinate_list) != len(set(coordinate_list)):
            if switch == "ic":
                self.init_coordinates() # redo the coordinate creation task
            elif switch == "rn":
                self.randomize_network() # redo the randomization task
    
    # function to create a similar network by modifying the old one
    def randomize_network(self):
        # create an empty coordinate array
        self.coordinates = np.arange(self.nodes*2).reshape(self.nodes, 2)
        # modifiy the old coordinates step by step
        for i in range(self.nodes):
            # change x coordinates
            randomizer = rn.randint(-2,2) # create and add a small random number
            self.coordinates[i][0] = self.original_coordinates[i][0] + randomizer
            # change y coordinates
            randomizer = rn.randint(-2,2) # create and add a small random number
            self.coordinates[i][1] = self.original_coordinates[i][1] + randomizer
        # convert the dimensions of the coordinates array to test them on duplicity
        self.coordinates = self.coordinates.reshape(self.nodes*2)
        # call function to make sure all nodes are different 
        # give parameter so it knows what to continue afterwards
        self.verify_coordinates("rn", self.coordinates)
        # convert the coordinate list into a two dimensional array
        # - columns : x and y coordinates
        # - lines: nodes
        self.coordinates = self.coordinates.reshape(self.nodes, 2)
        return self.coordinates
            
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
            repeat += 1  # increment the training iterations
    
# function to delete old data stored in the directory
def clear_data(data_path):
    # delete every file already existing in the folder
    files = glob.glob(data_path+"/*")
    for f in files:
        os.remove(f)

# function to save the data
def save_data(data_path, total_distance, coordinates, walkthrough):
    # create file name by adding the strings: path + "data" + number + format
    file_name = data_path + "data" + str(walkthrough) + ".txt"
    file = open(file_name, "a+") # open this file and "a+" -> if not existing create it
    file.write(str(total_distance)) # save the distance of the optimal path
    file.write("\n") # leave one line space
    file.write(str(coordinates)) # save the layout of the network
    file.close() # close the file

# function to print a progress bar while the program runs
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

print("Program started...")
# set important variables
nodes = 10 # number of nodes in the network
learning_rate = 0.999 # the decay rate of the training process for the pheromone
training_iterations = 1000 # how often does the algorithm has to train
data_path = "C:/ENTER_PATH_HERE" # path where the data is saved in text files
amount_of_data_files = 10 # number of text files to be created

# clear old data
clear_data(data_path)
# create new instance of the ACO
aco = ACO(nodes, training_iterations, learning_rate)
aco.init_coordinates() # create a random network

# Initial call to print 0% progress
progress = 0
progress_bar(progress, amount_of_data_files, 50)
saved_files = 0 # variable to remember how many files are created
while progress < amount_of_data_files:
    # call preparation functions
    coordinates = aco.randomize_network() # save the randomly created network in a variable
    total_distance = None # variable representing the travelled distance
    check = 0 # variable to test if the three ACOs calculated the same result
    # let the ACO calculate the same problem 3 times to be sure it found the best way
    for repeat in range(3):
        # reset the pheromone map
        aco.distances_and_pheromone() # calculate its distances and pheromone values
        # train the algorithm - pheromone map changes
        aco.train()
        # calculate results
        aco.determine_chosen_path()
        # if the travelled distance is smaller than before save it
        if  total_distance == None or total_distance > aco.calculate_travelled_distance():
            total_distance = aco.calculate_travelled_distance()
            check += 1
    # save the data if all three ACOs calculated the same result
    if check == 1:
        # save the data
        save_data(data_path, total_distance, coordinates, saved_files+1)
        saved_files += 1
    # continue picturing the progress bar
    progress_bar(progress+1, amount_of_data_files, 50)
    progress += 1

print(saved_files,"of", amount_of_data_files, "successfully saved.")
print("Program finished.")
