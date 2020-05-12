
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''
import numpy as np
from numpy import empty
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv
import matplotlib.pyplot as plt
from sklearn import metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    return [(9796134, 'Vaibhav', 'Vachhani')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    
        
    ##         "INSERT YOUR CODE HERE" 
    
    #to get size of the dataset (rows and columns)
    def get_size(fname):
        f = open(fname, 'r')
        line = f.readline()
        arr = line.split(',')
        num_cols = len(arr)
        with open(fname) as f:
                for i, l in enumerate(f):
                        pass
        return i + 1,num_cols
    rows,columns= get_size(dataset_path)
    f = open(dataset_path, 'r')
    if(f.mode=='r'):
        numberOfAttributes = columns - 2 #ignore ID and Y label columns for learners (first 2)
        numberOfObservations = rows
        #create empty numpy arrays
        X = np.zeros([numberOfObservations,numberOfAttributes])
        y = np.zeros(numberOfObservations)
        y = y.astype(np.uint8)
        #read each line and store it in the array
        for i in range(0,numberOfObservations):
            contents = f.readline()
            data = contents.strip('\n')
            arr = data.split(',')
            #if result is 'M' we set Y label to 1 otherwise leave it as 0
            if((arr[1])=='M'):
                y[i]=1
            arr = np.asarray(arr[2:])
            X[i]=arr
            
        X = np.asarray(X)
        y = np.asarray(y)
        return X,y #return X and y
        
        
            
            
   
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training, depth):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    dt_clf = DecisionTreeClassifier(max_depth=depth, random_state = 9)# (random_state=1 to avoid randomness in results)
    #dt_clf = DecisionTreeClassifier(max_depth=depth)
    dt_clf = dt_clf.fit(X_training, y_training)
    
    return dt_clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training, numberOfNeighbors):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    knn_clf = KNeighborsClassifier(n_neighbors = numberOfNeighbors)
    knn_clf = knn_clf.fit(X_training, y_training)
    return knn_clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training, parameter_C):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    svm_clf = svm.SVC(C=parameter_C,gamma='scale', random_state = 9)
    svm_clf = svm_clf.fit(X_training, y_training)
    return svm_clf
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training, numberOfNeurons):
    '''  
    Build a Neural Network with two dense hidden layers classifier 
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    clf = MLPClassifier(hidden_layer_sizes=(numberOfNeurons,numberOfNeurons), activation='relu',
                        max_iter=30, learning_rate = 'constant', learning_rate_init= 0.0001, alpha = 0.0001,
                        solver='sgd', verbose=True, random_state = 9)
    clf = clf.fit(X_training, y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
def knn_Errors(X_data, y_data, optimal_K):
    '''
    Determine the accuracy of predicting X_data with a K Nereast Neighbor Classifier
    trained with most optimal number of neighbors

    @param 
	X_data: Data to be predicted
	y_training: Target class values
    optimal_K: Most optimal value of K

    @return
	accuracyScore : number representing the accuracy of the trainer on X_data, y_data
    '''
    knn_clf = build_NearrestNeighbours_classifier(X_train,y_train,numberOfNeighbors=optimal_K) #build a KNN Classifier with optimal_K
    y_hat = knn_clf.predict(X_data) # predict X_data using that classifier
    accuracyScore = metrics.accuracy_score(y_data, y_hat) #accuracy score calculated from difference between target values and predicted values
    return accuracyScore #return accuracy score



def decisionTreeClassifier_Errors(X_data, y_data, optimal_Depth):
    '''
    Determine the accuracy of predicting X_data with a Decison Tree Classifier
    trained with most optimal depth value.

    @param 
	X_data: Data to be predicted
	y_training: Target class values
    optimal_K: Most optimal depth value

    @return
	accuracyScore : number representing the accuracy of the trainer on X_data, y_data
    '''
    decisionTree_clf = build_DecisionTree_classifier(X_train,y_train,depth=optimal_Depth)  #build a Decision Tree Classifier with optimal depth value
    y_hat = decisionTree_clf.predict(X_data) # predict X_data using that classifier
    accuracyScore = metrics.accuracy_score(y_data, y_hat) #accuracy score calculated from difference between target values and predicted values
    return accuracyScore  #return accuracy score
 
def SVM_Errors(X_data, y_data, C):
    '''
    Determine the accuracy of predicting X_data with a Support Vector Machine Classifier
    trained with most optimal C value.

    @param 
	X_data: Data to be predicted
	y_training: Target class values
    C: Most optimal C Value

    @return
	accuracyScore : number representing the accuracy of the trainer on X_data, y_data
    '''
    svm_clf = build_SupportVectorMachine_classifier(X_train,y_train,parameter_C=C) #build a SVM Classifier with optimal depth value
    y_hat = svm_clf.predict(X_data) # predict X_data using that classifier
    accuracyScore =  metrics.accuracy_score(y_data, y_hat) #accuracy score calculated from difference between target values and predicted values
    return accuracyScore #return accuracy score

def NeuralNetowrk_Errors(X_data, y_data, optimalNumberOfNeurons):
    '''
    Determine the accuracy of predicting X_data with a Support Vector Machine Classifier
    trained with most optimal C value.

    @param 
	X_data: Data to be predicted
	y_training: Target class values
    C: Most optimal C Value

    @return
	accuracyScore : number representing the accuracy of the trainer on X_data, y_data
    '''
    svm_clf = build_NeuralNetwork_classifier(X_train,y_train,optimalNumberOfNeurons) #build a MLP Classifier with optimal number of neurons
    y_hat = svm_clf.predict(X_data) # predict X_data using that classifier
    accuracyScore =  metrics.accuracy_score(y_data, y_hat) #accuracy score calculated from difference between target values and predicted values
    return accuracyScore #return accuracy score
       
def knn_Tests(X_data,y_data, neighbors):
    '''
    Determine the most optimal number of neighbors for K Nearest Neighbor Classifier
    which classifies the training data with least Mean Sqaured Error (MSE)

    @param 
	X_data: 30 Features data for each record 
	y_training: Target label for each record

    @return
	optimal_K : number representing the most optimal value for number of neighbors
    '''
    MSE = [] #initialse empty array to store Mean Squared Errors
    # loop over given array to calcukate Cross Val Score and MSE for each trained model with different number of neighbors
    for k in neighbors:
        knn = build_NearrestNeighbours_classifier(X_train,y_train,numberOfNeighbors=k) #build KNN classifier
        scores = cross_val_score(knn, X_data, y_data, cv=5, scoring='accuracy') # calculate Cross Val Score
        MSE.append(1-scores.mean()) #Calculate MSE from Cross Val Score
    minIndexMSE = MSE.index(min(MSE)) #find where MSE is minimum
    optimal_K = neighbors[minIndexMSE] #find corresponding Neighbor value
    print ("The optimal number of neighbors is %d" % optimal_K)
    neighbors = list(neighbors)
    #Plot MSE vs Neighbor
    plt.plot(neighbors, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()
    return optimal_K # return Optimal number of Neighbors

def decisionTreeClassifier_Tests(X_data,y_data, depth):
    '''
    Determine the best depth value for Decision Tree Classifier
    which classifies the training data with least Mean Sqaured Error (MSE)

    @param 
	X_data: 30 Features data for each record 
	y_training: Target label for each record

    @return
	optimal_Depth : number representing the most optimal Depth value
    
    '''
    MSE = [] #initialse empty array to store Mean Squared Errors
    # loop over given array to calcukate Cross Val Score and MSE for each trained model with different depth value
    for i in depth:
        clf = build_DecisionTree_classifier(X_train,y_train,depth=i) #build Decision tree classifier
        scores = cross_val_score(clf, X_data, y_data,cv=5) # calculate Cross Val Score
        MSE.append(1- scores.mean()) #Calculate MSE from Cross Val Score
    minIndexMSE = MSE.index(min(MSE)) #find where MSE is minimum
    optimal_Depth = depth[minIndexMSE] #find corresponding depth value
    print ("The optimal depth for training data is %d" % optimal_Depth)
    plt.plot(depth,MSE)
    plt.xlabel('Max Depth of Decision Tree Classifier')
    plt.ylabel('Misclassification Error')
    plt.show()
    
    return optimal_Depth # return Optimal depth value
    
def SVM_Tests(X_data, y_data, C_values):
    '''
    Determine the most optimal value of C for Support Vector Machine Classifier
    which classifies the training data with least Mean Sqaured Error (MSE)

    @param 
	X_data: 30 Features data for each record 
	y_training: Target label for each record

    @return
	optimal_C : number representing the most optimal C Value
    '''
    MSE = [] #initialse empty array to store Mean Squared Errors
    # loop over given array to calcukate Cross Val Score and MSE for each trained model with different C value
    for i in C_values:
        clf = build_SupportVectorMachine_classifier(X_train,y_train,parameter_C=i) #build SVM classifier
        scores = cross_val_score(clf, X_data, y_data, cv=5)  # calculate Cross Val Score
        MSE.append(1- scores.mean()) #Calculate MSE from Cross Val Score
    minIndexMSE = MSE.index(min(MSE)) #find where MSE is minimum
    optimal_C = C_values[minIndexMSE] #find corresponding C value
    print ("The optimal value of C for training data is %d" % optimal_C)
    plt.plot(C_values,MSE)
    plt.xlabel('Parameter C Value')
    plt.ylabel('Misclassification Error')
    plt.show()
    
    return optimal_C # return Optimal C value

def NeuralNetowrk_Tests(X_data, y_data, neruon_values):
    '''
    Determine the most optimal number if Neurons for MLP Classifier
    which classifies the training data with least Mean Sqaured Error (MSE)

    @param 
	X_data: 30 Features data for each record 
	y_training: Target label for each record

    @return
	optimalNumberOfNeurons : number representing the most optimal number of neurons
                             from variable 'neurons'
    '''
    error = []  #initialse empty array to store Mean Squared Errors
    # loop over given array to calcukate Cross Val Score and MSE for each trained model with different number of neurons
    for i in neruon_values:
        clf = build_NeuralNetwork_classifier(X_train,y_train,i) #build MLP classifier
        scores = cross_val_score(clf, X_data, y_data, cv=5) # calculate Cross Val Score
        error.append(1-np.min(scores)) #Calculate MSE from Cross Val Score
    minIndexMSE = error.index(min(error)) #find where MSE is minimum
    optimalNumberOfNeurons = neurons[minIndexMSE] #find corresponding number of neighbors value
    print ("The optimal number of neurons for training data is %d" % optimalNumberOfNeurons)
    plt.plot(neurons,error)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Misclassification Error')
    plt.show()
    
    return optimalNumberOfNeurons # return Optimal number of neighbors


def NearestNeighbors():
    '''
    Performs tests for Nearest Neighbors Classifier to evaluate best number of neighbors
    and report errors and accuracy on it.
    '''
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("Training Data Graph for KNN Classifier")
    optimalNumberOfNeighbors = knn_Tests(X_train,y_train, neighbors)
    print("Accuracy on Training Data with optimal number of neighbors: " + str(knn_Errors(X_train,y_train,optimalNumberOfNeighbors)*100) + "%")
    print("Accuracy on Test Data with optimal number of neighbors: " + str(knn_Errors(X_test,y_test,optimalNumberOfNeighbors)*100) + "%")
    print("Accuracy on Validation Data with optimal number of neighbors: " + str(knn_Errors(X_val,y_val,optimalNumberOfNeighbors)*100) + "%")
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("Error on Training Data with optimal number of neighbors: " + str(100 - knn_Errors(X_train,y_train,optimalNumberOfNeighbors)*100) + "%")
    print("Error on Test Data with optimal number of neighbors: " + str(100 - knn_Errors(X_test,y_test,optimalNumberOfNeighbors)*100) + "%")
    print("Error on Validation Data with optimal number of neighbors: " + str(100 - knn_Errors(X_val,y_val,optimalNumberOfNeighbors)*100) + "%")
    print("")
    print("")
    print("")
    
def DecisionTree():
    '''
    Performs tests for Decision Tree Classifier to evaluate best value for depth
    and report errors and accuracy on it.
    '''
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("Training Data Graph for Decision Tree Classifier")
    optimalDepth = decisionTreeClassifier_Tests(X_train,y_train, depth)
    print("Accuracy on Training Data with optimal Depth: " + str(decisionTreeClassifier_Errors(X_train,y_train,optimalDepth)*100) + "%")
    print("Accuracy on Test Data with optimal Depth: " + str(decisionTreeClassifier_Errors(X_test,y_test,optimalDepth)*100) + "%")
    print("Accuracy on Validation Data with optimal Depth: " + str(decisionTreeClassifier_Errors(X_val,y_val,optimalDepth)*100) + "%")
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("Error on Training Data with optimal Depth: " + str(100 - decisionTreeClassifier_Errors(X_train,y_train,optimalDepth)*100) + "%")
    print("Error on Test Data with optimal Depth: " + str(100 - decisionTreeClassifier_Errors(X_test,y_test,optimalDepth)*100) + "%")
    print("Error on Validation Data with optimal Depth: " + str(100 -decisionTreeClassifier_Errors(X_val,y_val,optimalDepth)*100) + "%")
    
    print("")
    print("")
    print("")
    
def SupportVectorMachine():
    '''
    Performs tests for SVM Classifier to evaluate best value of C
    and report errors and accuracy on it.
    '''
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("Training Data Graph for SVM CLassifier")
    optimal_c = SVM_Tests(X_train,y_train, C_values)
    print("Accuracy on Training Data with optimal Depth: " + str(SVM_Errors(X_train,y_train,optimal_c)*100) + "%")
    print("Accuracy on Test Data with optimal Depth: " + str(SVM_Errors(X_test,y_test,optimal_c)*100) + "%")
    print("Accuracy on Validation Data with optimal Depth: " + str(SVM_Errors(X_val,y_val,optimal_c)*100) + "%")
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("Error on Training Data with optimal Depth: " + str(100- SVM_Errors(X_train,y_train,optimal_c)*100) + "%")
    print("Error on Test Data with optimal Depth: " + str(100- SVM_Errors(X_test,y_test,optimal_c)*100) + "%")
    print("Error on Validation Data with optimal Depth: " + str(100- SVM_Errors(X_val,y_val,optimal_c)*100) + "%")
    print("")
    print("")
    print("")
    
def NeuralNetworks():
    '''
    Performs tests for Neural Network Classifier to evaluate best number of neurons 
    and report errors and accuracy on it.
    '''
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("Training Data Graph for MLP CLassifier")
    optimal_NumberOfNeurons = NeuralNetowrk_Tests(X_train,y_train, neurons)
    print("Accuracy on Training Data with optimal number of neurons: " + str(NeuralNetowrk_Errors(X_train,y_train,optimal_NumberOfNeurons)*100) + "%")
    print("Accuracy on Test Data with optimal number of neurons: " + str(NeuralNetowrk_Errors(X_test,y_test,optimal_NumberOfNeurons)*100) + "%")
    print("Accuracy on Validation Data with optimal number of neurons: " + str(NeuralNetowrk_Errors(X_val,y_val,optimal_NumberOfNeurons)*100) + "%")
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("Error on Training Data with optimal number of neurons: " + str(100- NeuralNetowrk_Errors(X_train,y_train,optimal_NumberOfNeurons)*100) + "%")
    print("Error on Test Data with optimal number of neurons: " + str(100- NeuralNetowrk_Errors(X_test,y_test,optimal_NumberOfNeurons)*100) + "%")
    print("Error on Validation Data with optimal number of neurons: " + str(100- NeuralNetowrk_Errors(X_val,y_val,optimal_NumberOfNeurons)*100) + "%")
    print("")
    print("")
    print("")
    
    
if __name__ == "__main__":
    
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    
    #Splitting the data into 3 parts (Training, Testing & Validation)
    X, y = prepare_dataset("medical_records1.data")
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2,shuffle = False, random_state=1)
    X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.2,shuffle = False, random_state=1)
    
    
    #records considered for each dataset
    
    print ("Number of records for training : " + str(X_train.shape[0]))
    print()
    print()
    print ("Number of records for testing : " + str(X_test.shape[0]))
    print()
    print()
    print ("Number of records for validation : " + str(X_val.shape[0]))
    print()
    print()
    
    
    
    
    #hyper parameters for each classifiers declared here
    neighbors = [1,3,5,7,9,13,15,17,19,21,23,25,27,29,31,33,35,43,55] # neighbors array for KNN Classifier
    depth = range(3,20) # dpeth list for Decision Tree Classifier
    C_values = np.arange(1, 50, 0.5) # C Values for SVM CLassifier
    neurons  = range(55,200,10) # number of neurons
    
    #neighbors = [39,40,41,42,43,44,45,59,60,65,69,71,73,75,85,86,87,89,90] # neighbors array for KNN Classifier
    #depth = range(15,35) # dpeth list for Decision Tree Classifier
    #C_values = np.arange(5, 15, 0.03) # C Values for SVM CLassifier
    #neurons  = range(10,500,50) # number of neurons
    
    #neighbors = range(20,40) # neighbors array for KNN Classifier
    #depth = range(35,45) # dpeth list for Decision Tree Classifier
    #C_values = np.arange(10, 20, 0.07) # C Values for SVM CLassifier
    #neurons  = range(5,100,5) # number of neurons
    
    
    '''
    Uncomment one of the lines below to run tests and evaluate best hyperparameter of any Classifier
    Please test one at a time.
    '''
    #NearestNeighbors()
    #DecisionTree()
    #SupportVectorMachine()
    #NeuralNetworks()
    


