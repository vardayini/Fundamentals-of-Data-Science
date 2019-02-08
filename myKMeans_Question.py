
"""
K-means clustering (Question)
Name: Vardayini Sharma
"""
import pandas as pd
import numpy as np
import random as rd
import sys

def myKMeans(dataset, k, seed):
    """Returns the centroids of k clusters for the input dataset.
    
    Parameters
    ----------
    dataset: a pandas DataFrame
    k: number of clusters
    seed: a random state to make the randomness deterministic
    
    Examples
    ----------
    myKMeans(df, 5, 123)
    myKMeans(df)
    
    Notes
    ----------
    The centroids are returned as a new pandas DataFrame with k rows.
    
    """
    #global numData
    #Get numerical features
    numData = getNumFeatures(dataset)
    #initial random centroids
    initCentroids = getInitialCentroids(numData, k, seed)
    #labels specifying the cluster membership
    labels = getLabels(numData, initCentroids)
    #adding labels to the dataframe
    numData['labels'] =  labels
    #Computing new centroids
    centroids = computeCentroids(numData, labels)
    #drop labels to add cluster membership of next iteration
    numData.drop(['labels'],axis=1, inplace=True)
    #assign current centroids as oldCentroids and computed as newCentroids
    #global oldCentroid
    oldCentroid = initCentroids.copy()
    #global newCentroid
    newCentroid = centroids.copy()
    #global n
    #initializing numIterations as n=0
    n=0
    
    while True:
        #if the function returns False, It will break out of the while loop
        if not stopClustering(oldCentroid, newCentroid, n, 100, 1e-10):
            break
        #labels specifying the cluster membership
        labels = getLabels(numData, oldCentroid)
        #adding labels to the dataframe
        numData['labels'] =  labels
        #assign current centroids as oldCentroids and computed as newCentroids
        oldCentroid = newCentroid.copy()
        #computing newCentroids
        newCentroid = computeCentroids(numData, labels)
        #drop labels to add cluster membership of next iteration
        numData.drop(['labels'],axis=1, inplace=True)
        #incrementing value of numIterations
        n = n+1
        
    return newCentroid
    
    
    


def getNumFeatures(dataset):
    """Returns a dataset with only numerical columns in the original dataset"""
    return dataset.select_dtypes(include=['float','int', 'long', 'complex'])

def getInitialCentroids(dataset, k, seed):
    """Returns k randomly selected initial centroids of the dataset"""
    #seed for deterministic results
    rd.seed(seed)  
    
    #list of column names
    cols = list(dataset.columns)   
    
    #empty array to store centroid values
    centroids = np.ndarray(shape=(k, len(cols)), dtype=float)  
    
    #centroid for every cluster
    for i in range(k):            
        #Loop for every feature   
        for j in range(len(cols)):
            #picking a random index between the range of index
            col_index = rd.randint(dataset.index.min(), (dataset.index.max()-1)) 
            #appending it to the centroids array
            centroids[i,j] = dataset.loc[col_index][cols[j]]
    
    return pd.DataFrame(centroids)

def getLabels(dataset, centroids):
    """Assigns labels (i.e. 0 to k-1) to individual instances in the dataset.
    Each instance is assigned to its nearest centroid.
    """ 
    centroids = np.array(centroids)
    #to get the value of k
    k = centroids.shape[0]
    # length of the dataset columnwise
    #len_df = len(dataset.columns)
    #Maximum possible number to be used in finding minimum
    dataset['distance'] = np.sys.float_info.max
    dataset['label'] = 0
    df1 =  dataset.copy()
    #to calculate to which cluster the point is closest to
    for i in range(k):
        #Finding temporary distance to compare
        df1['temp_dist'] = ((dataset.drop(['label','distance'],axis=1) - centroids[i,:])**2).sum(axis=1)** 0.5
        #assigning that cluster membership if distance is minimum
        dataset['label']  = np.where((dataset['distance'] > df1['temp_dist']), i, dataset['label'])
        #assigning distance if this distance is minimum
        dataset['distance']  = np.where((dataset['distance'] > df1['temp_dist']), df1['temp_dist'], dataset['distance'])
    
    #saving the labels in an array
    labels = np.array(dataset['label'])
    #dropping the columns that were created to assign labels
    dataset.drop(['label','distance'], axis=1, inplace=True)
    return labels


def computeCentroids(dataset, labels):
    """Returns the centroids of individual groups, defined by labels, in the dataset"""
    
    #did the computed the distance under getLabels
    #computing mean for the centroid
    df3=dataset.groupby(['labels']).mean().reset_index()
    
    #return np.array(df3.drop('labels',axis=1))
    return df3.drop('labels',axis=1,errors={'ignore'})

def stopClustering(oldCentroids, newCentroids, numIterations, maxNumIterations=100, tol=1e-4):
    """Returns a boolean value determining whether the k-means clustering converged.
    Two stopping criteria: 
    (1) The distance between the old and new centroids is within tolerance OR
    (2) The maximum number of iterations is reached 
    """

    #converting to numpy arrays for calculations
    old = np.array(oldCentroids)
    new = np.array(newCentroids)
    #to check if number of iterations exceed the maximum iterations allowed
    if (numIterations > maxNumIterations):
        return False
    #to check if distance is less than tol
    elif ((((old - new)**2).sum()**0.5) < tol):
        return False
    else:
        return True
    
        