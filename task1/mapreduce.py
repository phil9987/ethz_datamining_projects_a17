import numpy as np
from timeit import default_timer as timer
import sys
from math import *

num_shingles = 8193
p2 = 32416189777  # Prime for hash functions
p = 257114623
permutations = None  # Cache permutations
num_permutations = 1000
num_buckets = 10000
num_rows_per_band = 20

# Fix seed for random number generator because we need to ensure that the same hash functions
# are used for all documents (columns) - Note: it does not suffice to make the relevant
# variables global because runner.py runs several instances of this script in different processes
random_seed = 37 
random_seed2 = 15485927


def mapper(key, value):
    # t0 = timer()
    # key: None
    # value: one line of input file, representing a column of the input matrix
    global num_permutations
    
    # Parsing
    col_string = value.split(" ")
    key_string = col_string.pop(0)
    col = list(map(int, col_string))  # Column of input matrix (= one document)
    key = int(key_string.split("_")[1])
    
    # Compute signature of given shingle column
    # t1 = timer()
    # signature = compute_signature_fast(col, num_permutations)
    signature = compute_signature(col, num_permutations)
    # t2 = timer()

    buckets = compute_buckets(signature)
    
    for bucket in list(buckets):
        col.append(key)     # append key to column
        yield bucket, col
    # t3 = timer()
    # print("pre:{} sig:{} post:{}".format(t1-t0,t2-t1,t3-t2))


def reducer(key, values):
    # key: key from mapper used to aggregate, the bucket number
    # values: list of all value for that key, the documents with this bucket number
    if len(values) == 1:  # no duplicates
        return
    sorted_docs = sorted(values, key=lambda x: x[-1])  # sort values by key (last element of list)
    # print "Converting %i documents" % len(sorted_docs)
    sorted_docs = [(doc[-1], set(doc[:-1])) for doc in sorted_docs]  # unpack into key value tuples and convert to sets
    doc1_old = -1
    doc2_old = -1
    for i, doc1 in enumerate(sorted_docs):
        for doc2 in sorted_docs[i + 1:len(sorted_docs)]:
            if doc1[0] == doc1_old and doc2[0] == doc2_old:
                continue
            doc1_old = doc1[0]
            doc2_old = doc2[0]
            if jaccard_similarity(doc1[1], doc2[1]) < 0.85:
                # print "Comparing %i and %i: mismatch" % (doc1[0], doc2[0])
                continue
            # print "Comparing %i and %i: match" % (doc1[0], doc2[0])
            #
            # print(doc1, doc2)
            yield doc1[0], doc2[0]

def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(x, y))
    union_cardinality = len(set.union(x, y))
    return intersection_cardinality / float(union_cardinality)


def compute_signature(col, num_permutations=1024):
    # t0 = timer()
    global num_shingles
    global permutations

    # Compute permutations if not done so yet (reuse for all documents)
    if permutations is None:
        np.random.seed(random_seed)
        permutations = np.empty((num_shingles, num_permutations))
        for i in range(num_permutations):
            permutations[:,i] = np.random.permutation(num_shingles)

    # Minhashing
    # signature: One column of the signature matrix, len(signature) == num_permutations
    signature = np.ones(num_permutations) * 65536
    # t1 = timer()
    for r in col:
        signature = np.minimum(signature, permutations[r,:])
    # t2 = timer()

    # print("pre:{} min:{}".format(t1 - t0, t2 - t1))
    return signature

# computes the signature matrix without explicitly creating the permutations, instead it uses hash functions. See exercise slides page 29 for reference
def compute_signature_fast(col, num_permutations=1024):
    np.random.seed(random_seed)

    a_s = np.random.random_integers(1, num_shingles+1, num_permutations) # draw a for hash functions of form (a*s + b mod p) mod num_shingles
    b_s = np.random.random_integers(0, num_shingles+1, num_permutations) # draw b for hash functions

    signature = np.ones(num_permutations) * 65536
    for r in col:   # col is in sparse representation, so every entry is a '1'
        hashes = np.mod(np.mod(np.add(np.multiply(a_s, r), b_s), p), num_shingles)  
        signature = np.minimum(signature, hashes)      # set signature to element-wise-minimum of signature and the newly calculated hashes

    return signature


def compute_buckets(signature_column):

    global num_buckets
    global num_rows_per_band
    global random_seed2
    global p

    np.random.seed(random_seed2)

    num_hashes = len(signature_column)
    num_bands = num_hashes / num_rows_per_band
    np.random.seed(random_seed2)
    a_s = np.random.random_integers(1, num_buckets+1, num_hashes) # draw a for hash functions of form (a*s + b mod p) mod num_buckets
    b_s = np.random.random_integers(0, num_buckets+1, num_hashes) # draw b for hash functions

    hashes = np.mod(np.mod(np.add(np.multiply(a_s, signature_column), b_s), p), num_buckets)
    #print("calculated hashes: {}".format(hashes))

    for i in range(0, num_bands):
        start_idx = i*num_rows_per_band
        end_idx = start_idx + num_rows_per_band
        yield (sum(hashes[start_idx:end_idx]) % num_buckets) + num_buckets*(i)