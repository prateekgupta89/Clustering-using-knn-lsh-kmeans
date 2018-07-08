import turicreate as tc                                        # machine learning library 
import matplotlib.pyplot as plt                                # plotting
import numpy as np                                             # dense matrices
from scipy.sparse import csr_matrix                            # sparse matrices
from sklearn.preprocessing import normalize                    # normalizing vectors
from sklearn.metrics import pairwise_distances                 # pairwise distances
import sys      
import os
import collections

def load_sparse_csr(filename):
    '''
    Function to get TF-IDF vectors
    '''

    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix( (data, indices, indptr), shape)

def get_initial_centroids(data, k, seed=None):
    '''
    Randomly choose k data points as initial centroids
    '''
    if seed is not None:
        np.random.seed(seed)
    # number of data points
    n = data.shape[0]
        
    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)
    
    centroids = data[rand_indices,:].toarray()
    
    return centroids

def assign_clusters(data, centroids):
    '''
    Function to assign clusters to data points
    '''
    
    # Compute distances between each data point and the set of centroids:
    distances_from_centroids = pairwise_distances(data, centroids, metric='euclidean')
    
    # Compute cluster assignments for each data point:
    cluster_assignment = np.argmin(distances_from_centroids, axis=1) 
    
    return cluster_assignment

def revise_centroids(data, k, cluster_assignment):
    '''
    Function to reassign cluster centroids
    '''

    new_centroids = []

    for i in xrange(k):
        # Select all data points that belong to cluster i.
        member_data_points = data[cluster_assignment==i]
        # Compute the mean of the data points.
        centroid = member_data_points.mean(axis=0)
        
        # Convert numpy.matrix type to numpy.ndarray type
        centroid = centroid.A1
        new_centroids.append(centroid)
    
    new_centroids = np.array(new_centroids)
    
    return new_centroids

def compute_heterogeneity(data, k, centroids, cluster_assignment):
    '''
    Function to compute heterogenity of clusters
    '''    

    heterogeneity = 0.0
    for i in xrange(k):
        
        # Select all data points that belong to cluster i.
        member_data_points = data[cluster_assignment==i, :]
        
        # check if i-th cluster is non-empty
        if member_data_points.shape[0] > 0:
            # Compute distances from centroid to data points
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)
        
    return heterogeneity

def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    '''
    This function runs k-means on given data and initial set of centroids.
    maxiter: maximum number of iterations to run.
    record_heterogeneity: (optional) a list, to store the history of heterogeneity as function 
                          of iterations. If None, do not store the history.
    verbose: if True, print how many data points changed their cluster labels in each iteration
    '''
    
    centroids = initial_centroids[:]
    prev_cluster_assignment = None
    
    for itr in xrange(maxiter):        
        if verbose:
            print itr
        
        # Make cluster assignments using nearest centroids
        cluster_assignment = assign_clusters(data, centroids)
            
        # Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
        centroids = revise_centroids(data, k, cluster_assignment)
            
        # Check for convergence: if none of the assignments changed, stop
        if prev_cluster_assignment is not None and \
          (prev_cluster_assignment==cluster_assignment).all():
            break
        
        # Print number of new assignments 
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment!=cluster_assignment)
            if verbose:
                print '    {0:5d} elements changed their cluster assignment.'.format(num_changed)   
        
        # Record heterogeneity convergence metric
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)
        
        prev_cluster_assignment = cluster_assignment[:]
        
    return centroids, cluster_assignment

def plot_heterogeneity(heterogeneity, k):
    '''
    Function to plot heterogenity
    '''

    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.show()

def smart_initialize(data, k, seed=None):
    '''
    Function to se k-means++ to initialize a good set of centroids
    '''
    if seed is not None:
        np.random.seed(seed)
    centroids = np.zeros((k, data.shape[1]))
    
    # Randomly choose the first centroid.
    # Since we have no prior knowledge, choose uniformly at random
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx,:].toarray()
    # Compute distances from the first centroid chosen to all the other data points
    squared_distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()**2
    
    for i in xrange(1, k):
        # Choose the next centroid randomly, so that the probability for each data point to be chosen
        # is directly proportional to its squared distance from the nearest centroid.
        # A new centroid should be as far as from ohter centroids as possible.
        idx = np.random.choice(data.shape[0], 1, p=squared_distances/sum(squared_distances))
        centroids[i] = data[idx,:].toarray()
        # Compute distances from the centroids to all data points
        squared_distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean')**2,axis=1)
    
    return centroids

def kmeans_multiple_runs(data, k, maxiter, num_runs, seed_list=None, verbose=False):
    '''
    Function to run k-means multiple times and store the best result
    '''

    heterogeneity = {}
    
    min_heterogeneity_achieved = float('inf')
    best_seed = None
    final_centroids = None
    final_cluster_assignment = None
    
    for i in xrange(num_runs):
        
        # Use UTC time if no seeds are provided 
        if seed_list is not None: 
            seed = seed_list[i]
            np.random.seed(seed)
        else: 
            seed = int(time.time())
            np.random.seed(seed)
        
        # Use k-means++ initialization
        initial_centroids = smart_initialize(data, k, seed) 
        
        # Run k-means
        centroids, cluster_assignment = kmeans(data, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
        
        # To save time, compute heterogeneity only once in the end
        heterogeneity[seed] = compute_heterogeneity(data, k, centroids, cluster_assignment)
        
        if verbose:
            print 'seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed])
            sys.stdout.flush()
        
        # if current measurement of heterogeneity is lower than previously seen,
        # update the minimum record of heterogeneity.
        if heterogeneity[seed] < min_heterogeneity_achieved:
            min_heterogeneity_achieved = heterogeneity[seed]
            best_seed = seed
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment
    
    # Return the centroids and cluster assignments that minimize heterogeneity.
    return final_centroids, final_cluster_assignment

if __name__ == '__main__':

    # Load the dataset
    wiki = tc.SFrame('./data/people_wiki.gl/')
    # Load TF-IDF vectors
    tf_idf = load_sparse_csr('./data/people_wiki_tf_idf.npz')
    # Load mapping from index to word
    map_index_to_word = tc.SFrame('./data/people_wiki_map_index_to_word.gl/')

    # Normalize TF-IDF vectors
    tf_idf = normalize(tf_idf)

    k = 3
    heterogeneity = []
    initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                       record_heterogeneity=heterogeneity, verbose=True)
    plot_heterogeneity(heterogeneity, k)

    cluster_counts = np.bincount(cluster_assignment)
    temp = np.nonzero(cluster_counts)[0]
    print 'The cluster assignment is --- '
    print zip(temp, cluster_counts[temp])
  
    print 'Experimenting with random initialization'
    k = 10
    heterogeneity = {}
    import time
    start = time.time()
    for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
        initial_centroids = get_initial_centroids(tf_idf, k, seed)
        centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
        # To save time, compute heterogeneity only once in the end
        heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
        print '--------------'
        print 'seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed])
        cluster_counts = np.bincount(cluster_assignment)
        temp = np.nonzero(cluster_counts)[0]
        print 'The cluster assignment is '
        print zip(temp, cluster_counts[temp])
        sys.stdout.flush()
    end = time.time()
    print 'Time taken = %f' % end-start

    print 'Experimenting with k-means++ initialization'
    k = 10
    heterogeneity_smart = {}
    start = time.time()
    for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
        initial_centroids = smart_initialize(tf_idf, k, seed)
        centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=None, verbose=False)
        # Compute heterogeneity only once in the end
        heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
        print 'seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity_smart[seed])
        sys.stdout.flush()
    end = time.time()
    print 'Time taken = %f' %end-start

    plt.figure(figsize=(8,5))
    plt.boxplot([heterogeneity.values(), heterogeneity_smart.values()], vert=False)
    plt.yticks([1, 2], ['k-means', 'k-means++'])
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.show()
