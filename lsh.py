import numpy as np                                             # dense matrices
import turicreate as tc                                        # machine learning library
from scipy.sparse import csr_matrix                            # sparse matrices
from sklearn.metrics.pairwise import pairwise_distances        # pairwise distances
from copy import copy                                          # deep copies
import matplotlib.pyplot as plt                                # plotting
from itertools import combinations
import time

def norm(x):
    '''
    Compute norm of a sparse vector
    '''
    
    sum_sq = x.dot(x.T)
    norm = np.sqrt(sum_sq)
    return norm

def load_sparse_csr(filename):
    '''
    Function to extract TF-IDF vectors for each document
    '''
    
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)

def generate_random_vectors(num_vector, dim):
    '''
    Function to generate random vectors of dim dimesnion
    '''

    return np.random.randn(dim, num_vector)

def train_lsh(data, num_vector=16, seed=None):
    '''
    Function to compute bin indices for documents
    '''

    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)
  
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
  
    table = {}
    
    # Partition data points into bins
    bin_index_bits = (data.dot(random_vectors) >= 0)

    # Encode bin index bits into integers
    bin_indices = bin_index_bits.dot(powers_of_two)

    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # If no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = []
        # Fetch the list of document ids associated with the bin and add the document id to the end.
        table[bin_index].append(data_index)

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}
    
    return model

def cosine_distance(x, y):
    '''
    Function to compute cosine distance
    '''

    xy = x.dot(y.T)
    dist = xy/(norm(x)*norm(y))
    return 1-dist[0,0]

def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    '''
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.
    '''
    
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
    
    # Allow the user to provide an initial set of candidates.
    candidate_set = copy(initial_candidates)
    
    for different_bits in combinations(range(num_vector), search_radius):       
        # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = not alternate_bits[i] 
        
        # Convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)
        
        # Fetch the list of documents belonging to the bin indexed by the new bit vector.
        # Then add those documents to candidate_set
        # Make sure that the bin exists in the table!
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])        
    
    return candidate_set

def query(vec, model, k, max_search_radius):
  
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    num_vector = random_vectors.shape[1]
    
    
    # Compute bin index for the query vector, in bit representation.
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()
    
    # Search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in xrange(max_search_radius+1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)
    
    # Sort candidates by their true distances from the query
    nearest_neighbors = tc.SFrame({'id':candidate_set})
    candidates = data[np.array(list(candidate_set)),:]
    nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten()
    
    return nearest_neighbors.topk('distance', k, reverse=True), len(candidate_set)

if __name__ == '__main__':

    # Load the dataset
    wiki = tc.SFrame('./data/people_wiki.gl/')
    # add row number, starting at 0
    wiki = wiki.add_row_number()

    # Load TF-IDF vectors
    corpus = load_sparse_csr('./data/people_wiki_tf_idf.npz')

    num_documents = corpus.shape[0]
    vocab_size = corpus.shape[1]
    print 'Number of documents = %d' % num_documents
    print 'Vocabulary size = %d' % vocab_size

    print wiki[wiki['name'] == 'Barack Obama']
    print wiki[wiki['name'] == 'Joe Biden']

    model = train_lsh(corpus, num_vector=16, seed=143)

    print model['table'][model['bin_indices'][35817]]

    # Finding other documents in the same bin as Obama's
    doc_ids = list(model['table'][model['bin_indices'][35817]])
    doc_ids.remove(35817) # display documents other than Obama

    # filter by id column
    docs = wiki.filter_by(values=doc_ids, column_name='id')
    print docs

    # TF-IDF vectors for Obama and Biden
    obama_tf_idf = corpus[35817,:]
    biden_tf_idf = corpus[24478,:]

    # Computing cosine distance for Biden which is two bits off from Obama
    # and documents in the same bin as Obama
    print 'Cosine distance from Barack Obama'
    print 'Barack Obama - {0:24s}: {1:f}'.format('Joe Biden',
                                             cosine_distance(obama_tf_idf, biden_tf_idf))
    for doc_id in doc_ids:
        doc_tf_idf = corpus[doc_id,:]
        print 'Barack Obama - {0:24s}: {1:f}'.format(wiki[doc_id]['name'],
                                             cosine_distance(obama_tf_idf, doc_tf_idf))

    # Fetch the 10 nearest documents to Obama
    result, num_candidates_considered = query(corpus[35817,:], model, k=10, max_search_radius=3)
    print "Top 10 nearest neighbor to Obama's article"
    print result.join(wiki[['id', 'name']], on='id').sort('distance')

    num_candidates_history = []
    query_time_history = []
    max_distance_from_query_history = []
    min_distance_from_query_history = []
    average_distance_from_query_history = []

    for max_search_radius in xrange(17):
        start=time.time()
        # Perform LSH query using Barack Obama, with max_search_radius
        result, num_candidates = query(corpus[35817,:], model, k=10,
                                   max_search_radius=max_search_radius)
        end=time.time()
        query_time = end-start  # Measure time
        
        print '-------'
        print 'Radius:', max_search_radius
        # Display 10 nearest neighbors, along with document ID and name
        print result.join(wiki[['id', 'name']], on='id').sort('distance')

        # Collect statistics on 10 nearest neighbors
        average_distance_from_query = result['distance'][1:].mean()
        max_distance_from_query = result['distance'][1:].max()
        min_distance_from_query = result['distance'][1:].min()
    
        num_candidates_history.append(num_candidates)
        query_time_history.append(query_time)
        average_distance_from_query_history.append(average_distance_from_query)
        max_distance_from_query_history.append(max_distance_from_query)
        min_distance_from_query_history.append(min_distance_from_query)
