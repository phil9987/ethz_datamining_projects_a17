import glob
import os

import sys

import time


def yield_pattern(path):
    """Yield lines from each file in specified folder"""
    for i in glob.iglob(path):
        if os.path.isfile(i):
            with open(i, "r") as fin:
                for line in fin:
                    yield line


def parse():
    for value in yield_pattern("data/handout_shingles.txt"):
        col_string = value.split(" ")
        key_string = col_string.pop(0)
        col = list(map(int, col_string))  # Column of input matrix (= one document)
        key = int(key_string.split("_")[1])
        col.append(key)
        yield col


def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


sorted_docs = sorted(parse(), key=lambda x: x[-1])  # sort values by key (last element of list)
doc1_old = -1
doc2_old = -1
start = time.time()
for i, doc1 in enumerate(sorted_docs):
    for doc2 in sorted_docs[i + 1:len(sorted_docs)]:
        if doc1[-1] == doc1_old and doc2[-1] == doc2_old:
            continue
        doc1_old = doc1[-1]
        doc2_old = doc2[-1]
        jaccard_similarity(doc1[:-1], doc2[:-1])
        # sys.stdout.write("\rComparing %i and %i" % (doc1[-1], doc2[-1]))
print("Completed in time {}".format(time.time()-start))