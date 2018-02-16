from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from scipy import stats
import numpy as np
import argparse
import sys
import os
import csv
import random

max_iter = 1000

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    total_sum = 0
    diag_sum = 0
    for i in range(len(C)):
        total_sum += sum(C[i,:])
        diag_sum += C[i,i]
    if total_sum == 0: 
        return 0
    return diag_sum/total_sum
 
def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recall_list = [0,0,0,0]
    for i in range(len(C)):
        class_sum = sum(C[i,:])
        if class_sum == 0:
            recall_list[i] = 0
        else:
            recall_list[i] = C[i,i]/class_sum
    return recall_list
 
def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precision_list = [0,0,0,0]
    for i in range(len(C)):
        class_sum = sum(C[:,i])
        if class_sum == 0:
            precision_list[i] = 0
        else:
            precision_list[i] = C[i,i]/class_sum
    return precision_list
 
def class31(filename):
    ''' This function performs experiment 3.1
     
    Parameters
       filename : string, the name of the npz file from Task 2
 
    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    # Extract feats (from question 2) from the contents of the compressed npy file
    data = np.load(filename)
    feats_array = data.f.arr_0
    
    features = feats_array[:,:173]
    labels = feats_array[:,173]
    
    # Split the data into a random 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=42)       

    # Create a result array, where the first row is accuracy, second is recall, third is precision, and each of the columns is a classifier
    # Use this to build the csv
    acc_array = np.zeros((1, 5))
    rec_array = np.zeros((5, 4))
    prec_array = np.zeros((5, 4))
    conf_array = []
    
    iBest = 0

    # Run through each classifier
    for i in range(5):
        if i+1 == 1:
            #1. SVC:  support vector machine with a linear kernel.
            #print("[3.1] Training classifier1: SVC - support vector machine with a linear kernel")
            classifier = SVC(kernel='linear', max_iter=max_iter) 
        elif i+1 == 2:
            #2. SVC:  support vector machine with a radial basis function (γ= 2) kernel.
            #print("[3.1] Training classifier2: SVC - support vector machine with a radial basis function (γ= 2) kernel")
            classifier = SVC(kernel='rbf', gamma=2, max_iter=max_iter) 
        elif i+1 == 3:
            #3. RandomForestClassifier:  with a maximum depth of 5, and 10 estimators.
            #print("[3.1] Training classifier3: RandomForestClassifier - with a maximum depth of 5, and 10 estimators")
            classifier = RandomForestClassifier(max_depth=5, n_estimators=10)
        elif i+1 == 4:
            #4. MLPClassifier:  A feed-forward neural network, with α = 0.05.
            #print("[3.1] Training classifier4: MLPClassifier - A feed-forward neural network, with α = 0.05")
            classifier = MLPClassifier(alpha=0.05)
        elif i+1 == 5:
            #5. AdaBoostClassifier:  with the default hyper-parameters. 
            #print("[3.1] Training classifier5: AdaBoostClassifier - with the default hyper-parameters")
            classifier = AdaBoostClassifier()

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        conf_array.append([conf_mat])
        acc_array[0,i] = accuracy(conf_mat)
        rec_array[i,:] = recall(conf_mat)
        prec_array[i,:] = precision(conf_mat)
        
    # Build the csv file string
    # While doing this, check to find the highest accuracy, use that to determine iBest
    output_filename = 'a1_3.1.csv'
    out_str = ""
    highest_accuracy = 0
    line_array = np.zeros((5,26))
    for i in range(5):
        line_array[i][:2] = [str(i+1), acc_array[0,i]]
        if acc_array[0,i] > highest_accuracy:
            highest_accuracy = acc_array[0,i]
            iBest = i + 1
        line_array[i][2:6] = rec_array[i]
        line_array[i][6:10] = prec_array[i]
        for j in range(4):
            C = conf_array[i][0]
            line_array[i][10+4*j:14+4*j] = C[j,:]
        for k in line_array[i]:
            out_str += str(k) + ', '
        out_str = out_str.rstrip(', ') + '\n'
    
    # Comments on observations and analysis
    comments = "The best classifier was determined to be RandomForestClassifier. This was computed based on highest accuracy at runtime. However upon further inspection it was noted that as well as a higher average RandomForestClassifier also had a higher recall and precision - further reinforcing its preference. The second best was MLPClassifier followed by AdaBoostClassifier. SVC was the least accurate which was partly due to a max iteration of 10000 being selected. While this did not allow SVC to train as accurately as it could otherwise - the values to which it was converging were still less accurate than the other classifiers. As such limiting iterations was done for more time-efficient runtime in a classifier whose accuracy did not demonstrate very accurate results anyways. One other note is that many features extracted from a1_extractFeatures.py were zeros far more often than not because of the aggressive StopWords removal from preprocessing. In essentially removing several features it would explain why RandomForestClassifier was so fast and possibly why it was so accurate: querying across fewer options in categorization could be easy to work with with fewer features - less divides needed"

    # Write to the csv    
    out_str += comments
    f = open(output_filename, "w") 
    f.write(out_str)
    f.close()

    
    return (X_train, X_test, y_train, y_test,iBest)
 
 
def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
     
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
 
    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    '''

    X_1k = 0
    y_1k = 0
    # Iterate through all the data increments, running the most accurate classifier from part 1 on all of them
    data_increments = [1000, 5000, 10000, 15000, 20000]
    acc_array = np.zeros((1,len(data_increments)))  
    for data_increment in range(len(data_increments)):
    
        # Get a list of random indices spanning data_increments[data_increment] in length
        indices = list(range(data_increments[data_increment]))
        random.shuffle([indices])
        index_array = np.array(indices)

        # Index the random index array from the full training data
        X_inc = X_train[[index_array],:][0]
        y_inc = y_train[[index_array]]
        
        # Store the 1K training data, to be returned
        if data_increments[data_increment] == 1000:
            X_1k = X_inc
            y_1k = y_inc
    
        if iBest == 1:
            #1. SVC:  support vector machine with a linear kernel.
            #print("[3.2] Training classifier1: SVC - support vector machine with a linear kernel on sample data size of {}".format(data_increments[data_increment]))
            classifier = SVC(kernel='linear', max_iter=max_iter) 
        elif iBest == 2:
            #2. SVC:  support vector machine with a radial basis function (γ= 2) kernel.
            #print("[3.2] Training classifier2: SVC - support vector machine with a radial basis function (γ= 2) kernel on sample data size of {}".format(data_increments[data_increment]))
            classifier = SVC(kernel='rbf', gamma=2, max_iter=max_iter) 
        elif iBest == 3:
            #3. RandomForestClassifier:  with a maximum depth of 5, and 10 estimators.
            #print("[3.2] Training classifier3: RandomForestClassifier - with a maximum depth of 5, and 10 estimators on sample data size of {}".format(data_increments[data_increment]))
            classifier = RandomForestClassifier(max_depth=5, n_estimators=10)
        elif iBest == 4:
            #4. MLPClassifier:  A feed-forward neural network, with α = 0.05.
            #print("[3.2] Training classifier4: MLPClassifier - A feed-forward neural network, with α = 0.05 on sample data size of {}".format(data_increments[data_increment]))
            classifier = MLPClassifier(alpha=0.05)
        elif iBest == 5:
            #5. AdaBoostClassifier:  with the default hyper-parameters. 
            #print("[3.2] Training classifier5: AdaBoostClassifier - with the default hyper-parameters on sample data size of {}".format(data_increments[data_increment]))
            classifier = AdaBoostClassifier()
        
        classifier.fit(X_inc, y_inc)
        y_pred = classifier.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        acc_array[0][data_increment] = accuracy(conf_mat)
    
    #print(acc_array[0])
    
    # Comment on the changes to accuracy as the number of training samples increases, including at least two sentences on a possible explanation
    # Is there an expected trend?  Do you see such a trend?  Hypothesize as to why or why not.
    comments = "Iterating through the different data sizes generally yielded an increase in the accuracy of the RandomForestClassifier classifier (selected from part 3.1) over the course of all training data sizes. The gap was often largest from 5K to 10K and usually had very minor improvement from 10K to 15K to 20K. Running the function 500 times (re-randomizing indices of the selected X K data samples each time) gave the following average accuracties for [5K 10K 15K 20K] respectively: [ 0.64383125  0.66960525  0.67131575  0.67298975  0.67338575]. This implies that the average number of samples needed to train the model to converged to a maxiumal accuracy was greater than 20K. An explanation for this would be that with more samples the RandomForestClassifier would be able to train more extensively (using more data) and develop a model that was more fine-tuned to classify data. However by inspection there were often cases where using 20K training data samples resulted in a slight decrease in accuracy compared to the 15K (but still approximately around the 10K result). This accuracy decrease is a clear case of overfitting where the model became too fine-tuned to the training data such that the test data became more like outliers to the model. This only happened in some iterations when the model was able to converge faster. For the most part though the model was still getting better after 20K samples."
    
    # Build the csv file
    output_filename = 'a1_3.2.csv'
    out_str = ""
    for k in acc_array[0]:
        out_str += str(k) + ', '
    out_str = out_str.rstrip(', ') + '\n' + comments
    f = open(output_filename, "w") 
    f.write(out_str)
    f.close()
    
    return (X_1k, y_1k)
     
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
     
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    
    k_vals = [5, 10, 20, 30, 40, 50]
    sorted_min_5 = [0,0,0,0,0]
    out_str = ""
    for k in k_vals:
        selector = SelectKBest(f_classif, k)
        X_new = selector.fit_transform(X_train, y_train)
        pp = selector.pvalues_
        pp_np = np.array(pp)
        sorted_min = pp_np.argsort()[:k]
        if (k == 5):
            sorted_min_5 = sorted_min
        out_str += str(k) + ', ' + ",".join(str(i) for i in pp) + '\n'
        
    if i+1 == 1:
        #1. SVC:  support vector machine with a linear kernel.
        #print("[3.3] Training classifier1: SVC - support vector machine with a linear kernel")
        classifier1 = SVC(kernel='linear', max_iter=max_iter) 
        classifier2 = SVC(kernel='linear', max_iter=max_iter)
    elif i+1 == 2:
        #2. SVC:  support vector machine with a radial basis function (γ= 2) kernel.
        #print("[3.3] Training classifier2: SVC - support vector machine with a radial basis function (γ= 2) kernel")
        classifier1 = SVC(kernel='rbf', gamma=2, max_iter=max_iter) 
        classifier2 = SVC(kernel='rbf', gamma=2, max_iter=max_iter) 
    elif i+1 == 3:
        #3. RandomForestClassifier:  with a maximum depth of 5, and 10 estimators.
        #print("[3.3] Training classifier3: RandomForestClassifier - with a maximum depth of 5, and 10 estimators")
        classifier1 = RandomForestClassifier(max_depth=5, n_estimators=10)
        classifier2 = RandomForestClassifier(max_depth=5, n_estimators=10)
    elif i+1 == 4:
        #4. MLPClassifier:  A feed-forward neural network, with α = 0.05.
        #print("[3.3] Training classifier4: MLPClassifier - A feed-forward neural network, with α = 0.05")
        classifier1 = MLPClassifier(alpha=0.05)
        classifier2 = MLPClassifier(alpha=0.05)
    elif i+1 == 5:
        #5. AdaBoostClassifier:  with the default hyper-parameters. 
        #print("[3.3] Training classifier5: AdaBoostClassifier - with the default hyper-parameters")
        classifier1 = AdaBoostClassifier()
        classifier2 = AdaBoostClassifier()

    # Create sub-array by indexing from the best pp values from before, using k=5
    X_5_train = X_train[[sorted_min_5],:][0]
    y_5_train = y_train[[sorted_min_5]]
    X_5_1k = X_1k[[sorted_min_5],:][0]
    y_5_1k = y_1k[[sorted_min_5]]
    X_test = X_test[[sorted_min_5],:][0]
    y_test = y_test[[sorted_min_5]]

    # Fit the data to the 1K and 32K models, using only the k=5 best features
    classifier1.fit(X_5_1k, y_5_1k)
    classifier2.fit(X_5_train, y_5_train)
    y_pred1 = classifier1.predict(X_test)
    y_pred2 = classifier2.predict(X_test)
    conf_mat1 = confusion_matrix(y_test, y_pred1)
    conf_mat2 = confusion_matrix(y_test, y_pred2)
    acc1 = accuracy(conf_mat1)
    acc2 = accuracy(conf_mat2)
    out_str += str(acc1) + ', ' + str(acc2) + '\n'
    
    # Comments (lines 8 to 10 of a1_3.3.csv), answer the following questions:
    #   (a)  What features,  if any,  are chosen at both the low and high(er) amounts of input data?  Also provide a possible explanation as to why this might be.
    #   (b)  Are p-values generally higher or lower given more or less data?  Why or why not?
    #   (c)  Name the top 5 features chosen for the 32K training case. Hypothesize as to why those particular features might differentiate the classes.
    comment_8 = "Some features that were common to both the 32K sample data and a select 1K example used (both within the first 20 features) were the following: receptiviti_health_oriented; liwc_certain; liwc_shehe. These are not terribly surprising in differentiating poitical leaning. Health care is a controversial topic whose beliefs vary widely across the spectrum. This is also true of gender (she/he) which is a complex movement more supported by left-leaning political views and more black-and-white as we move further right. Finally certainty (especially in this Fake News era) is evidently a big issue as of late - creating divide against whatever media each political sub-group deems to be the truth."
    comment_9 = "Comparing p-values across the 1K and 32K sample sizes showed that the 32K samples had much lower p-values (ranging from too small for python to represent (earning a 0.0) to a maximum value of 0.36. Meanwhile the 1K sample p-values ranged from e-193 to 0.6. This can be explained by more data being available to train the model and allow each feature to make more distinguishing classifications. This is generally the case."
    comment_10 = "172-receptiviti_health_oriented; 68-liwc_focuspast; 67-liwc_focusfuture; 121-receptiviti_anxious; 122-receptiviti_artistic. From the names we can infer what these features are about. As mentioned above health is a dividing issue on the political spectrum. Anxiety is also a very common attribute to the right (especially Alt) and could easily be seen as a strong differentiator. Artistic could refer to the mroe liberal attitude of the left. Finally the last two features that are in the top-5 for differentiation among the 32K sample size are past and future tense. This may indicate a forward vs past-thinking attitude about some of the political groups."
    
    out_str += comment_8 + '\n' + comment_9 + '\n' + comment_10
    
    output_filename = 'a1_3.3.csv'
    f = open(output_filename, "w") 
    f.write(out_str)
    f.close()
    
    #f = open(output_filename, "r")
    #print(f.read())
    #f.close()
 
def class34( filename, i ):
    ''' This function performs experiment 3.4
     
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    # Load data from the input file (output from Task 2)      
    data = np.load(filename)
    feats_array = data.f.arr_0
    features = feats_array[:,:173]
    labels = feats_array[:,173]
    acc_array = np.zeros((5, 5))
    
    # Split the data into 5 groups of 8Kx173 training data and 8Kx1  each (pseudo-recursively use the train_test_split function to randomly partition it)
    #X_remainder, X_group1, y_remainder, y_group1 = train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=42)
    #X_remainder, X_group2, y_remainder, y_group2 = train_test_split(X_remainder, y_remainder, train_size=0.75, test_size=0.25, random_state=42)
    #X_remainder, X_group3, y_remainder, y_group3 = train_test_split(X_remainder, y_remainder, train_size=2/3, test_size=1/3, random_state=42)
    #X_group5, X_group4, y_group5, y_group4 = train_test_split(X_remainder, y_remainder, train_size=0.5, test_size=0.5, random_state=42)
    
    
    # Run 5-fold KFold on each classifier
    kf = KFold(n_splits=5, shuffle=True)
    output_str = ""
    for c in range(5):
        if c+1 == 1:
            #1. SVC:  support vector machine with a linear kernel.
            #print("[3.4] Training classifier1: SVC - support vector machine with a linear kernel")
            classifier = SVC(kernel='linear', max_iter=max_iter) 
        elif c+1 == 2:
            #2. SVC:  support vector machine with a radial basis function (γ= 2) kernel.
            #print("[3.4] Training classifier2: SVC - support vector machine with a radial basis function (γ= 2) kernel")
            classifier = SVC(kernel='rbf', gamma=2, max_iter=max_iter) 
        elif c+1 == 3:
            #3. RandomForestClassifier:  with a maximum depth of 5, and 10 estimators.
            #print("[3.4] Training classifier3: RandomForestClassifier - with a maximum depth of 5, and 10 estimators")
            classifier = RandomForestClassifier(max_depth=5, n_estimators=10)
        elif c+1 == 4:
            #4. MLPClassifier:  A feed-forward neural network, with α = 0.05.
            #print("[3.4] Training classifier4: MLPClassifier - A feed-forward neural network, with α = 0.05")
            classifier = MLPClassifier(alpha=0.05)
        elif c+1 == 5:
            #5. AdaBoostClassifier:  with the default hyper-parameters. 
            #print("[3.4] Training classifier5: AdaBoostClassifier - with the default hyper-parameters")
            classifier = AdaBoostClassifier()
    
        fold_count = 0
        for train_index, test_index in kf.split(features, labels):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            
            # Train the model on the 5 folds, for each classifier
            # One line per fold
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            conf_mat = confusion_matrix(y_test, y_pred)
            acc_array[fold_count,c] = accuracy(conf_mat)
            fold_count += 1
        
        fold_count += 1
    for j in range(5):
        output_str += ",".join(str(entry) for entry in acc_array[j]) + '\n'
    
    indices = list(range(5))
    indices.remove(i-1)
    for j in indices:
        S = stats.ttest_rel(acc_array[:,i-1], acc_array[:,j])
        output_str += str(S[1]) + ', '
    output_str = output_str.rstrip(', ') + '\n'
    
    # Comment on any significance you observe, or any lack thereof, and hypothesize as to why, in one to three sentences
    comments = "From the p-values obtained we see that there is a tiny relative p-value between the best classifier (RandomForestClassifier) and the others (orderr of e-5 to e-7). The exception being MPC for which there is a larger p-value (order of 0.01). This implies that we can't reject the similar average null hypothesis and therefore they would both yield similar results - this reinforces results from part 3.1 where they were both about equally good but with RandomForestClassifier getting the edge."
    output_str += comments
    
    output_filename = 'a1_3.4.csv'
    f = open(output_filename, "w") 
    f.write(output_str)
    f.close()
    
    #f = open(output_filename, "r")
    #print(f.read())
    #f.close()
    
     
if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()
    
    filename = args.input
    experiments = [1,2,3,4]
    if 1 in experiments:
        print("Performing experiment 3.1:")
        (X_train, X_test, y_train, y_test, iBest) = class31(filename)
        
    if 2 in experiments:
        print("Performing experiment 3.2:")
        (X_1k, y_1k) = class32(X_train, X_test, y_train, y_test, iBest)
        
    if 3 in experiments:
        print("Performing experiment 3.3:")
        class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
        
    if 4 in experiments:
        print("Performing experiment 3.4:")
        class34( filename, iBest)


