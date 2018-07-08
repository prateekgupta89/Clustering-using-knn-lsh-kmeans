import turicreate as tc                          # to load the data
import matplotlib.pyplot as plt                  # plotting
import numpy as np                               # dense matrices
from scipy.sparse import csr_matrix              # sparse matrices
import sklearn
from sklearn.neighbors import NearestNeighbors

def load_sparse_csr(filename):
    '''
    Function to load word count vectors
    '''

    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)

def unpack_dict(matrix, map_index_to_word):
    '''
    Function to represent word_count vectors in a dictionary form
    '''    
    table = list(map_index_to_word.sort('index')['category'])
    
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    
    num_doc = matrix.shape[0]

    return [{k:v for k,v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i+1]] ],
                                 data[indptr[i]:indptr[i+1]].tolist())} \
               for i in xrange(num_doc) ]

def top_words(name):
    '''
    Get a table of the most frequent words in the given person's wikipedia page.
    '''

    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False)

def top_words_tf_idf(name):
    '''
    Get a table of the most frequent words in the given person's wikipedia page
    based on tf-idf
    '''
    
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)

def has_top_words(word_count_vector, common_words):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())
    # return True if common_words is a subset of unique_words
    # return False otherwise
    if common_words.issubset(unique_words):
        return True
    else:
        return False

if __name__ == '__main__':

    #Load dataset
    wiki = tc.SFrame('./data/people_wiki.gl/')
    wiki = wiki.add_row_number()

    # Get word count vectors for the documents
    word_count = load_sparse_csr('./data/people_wiki_word_count.npz')

    # Word-to-index mapping
    map_index_to_word = tc.SFrame('./data/people_wiki_map_index_to_word.gl/')

    # Find nearest neighbors using word count vectors
    model = NearestNeighbors(metric='euclidean', algorithm='brute')
    model.fit(word_count)

    # Obtain the row number for Barack Obama's artice
    print wiki[wiki['name'] == 'Barack Obama']
    
    # Run the k-nearest neighbor algorithm with Obama'a article
    distances, indices = model.kneighbors(word_count[35817], n_neighbors=10)

    # Display the indices and distances of the 10 nearest neighbors
    neighbors = tc.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
    print wiki.join(neighbors, on='id').sort('distance')[['id','name','distance']]

    # convert word count vectors to a dictionary
    wiki['word_count'] = unpack_dict(word_count, map_index_to_word)

    # Get the top most frequent words in a given person's wikipedia page
    obama_words = top_words('Barack Obama')
    print obama_words

    barrio_words = top_words('Francisco Barrio')
    print barrio_words

    combined_words = obama_words.join(barrio_words, on='word')

    combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})

    combined_words.sort('Obama', ascending=False)
    print combined_words

    common_words = list(combined_words['word'])

    wiki['has_top_words'] = wiki['word_count'].apply(lambda x:has_top_words(x, set(common_words[0:5])))

    # Number of documents having the top 5 same words as Obama's article
    num_documents = sum(list(wiki['has_top_words']))
    print num_documents

    # Load TF-IDF vectors
    tf_idf = load_sparse_csr('./data/people_wiki_tf_idf.npz')

    # Store tf-idf vectors in dictionary form
    wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)
    
    # Finf nearest neighbor using TF-IDF vectors
    model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
    model_tf_idf.fit(tf_idf)

    # Run the k-nearest neighbor algorithm with Obama'a article
    distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)

    # Display the indices and distances of the 10 nearest neighbors
    neighbors = tc.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
    print wiki.join(neighbors, on='id').sort('distance')[['id', 'name', 'distance']]

    obama_tf_idf = top_words_tf_idf('Barack Obama')
    print obama_tf_idf

    schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
    print schiliro_tf_idf
    
    combined_words_tf_idf = obama_tf_idf.join(schiliro_tf_idf, on='word')
 
    combined_words_tf_idf = combined_words_tf_idf.rename({'weight':'Obama', 'weight.1':'Schiliro'})

    combined_words_tf_idf.sort('Obama', ascending=False)
    print combined_words_tf_idf

    common_words = list(combined_words_tf_idf['word'])

    wiki['has_top_words_tf_idf'] = wiki['word_count'].apply(lambda x:has_top_words(x, set(common_words[0:5])))
    
    # Number of documents having the top 5 same words as Obama's article using TF-IDF weights
    num_documents = sum(list(wiki['has_top_words_tf_idf']))
    print num_documents
   
    # Computing nearest neighbors using cosine similarity
    model2_tf_idf = NearestNeighbors(algorithm='brute', metric='cosine')
    model2_tf_idf.fit(tf_idf)
    distances, indices = model2_tf_idf.kneighbors(tf_idf[35817], n_neighbors=100)
    neighbors = tc.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
    nearest_neighbors_cosine = wiki.join(neighbors, on='id')[['id', 'name', 'distance']].sort('distance')
    print nearest_neighbors_cosine
       
