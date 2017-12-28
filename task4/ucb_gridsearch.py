from __future__ import division
import numpy as np
import numpy.linalg as lin
#from time import perf_counter as timer
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

logger.info("alpha: {} r0: {} r1: {}".format(alpha, r0, r1))
articles = None
M = None
Minv = None
b = None
lookup = None
x = None
z = None
matches = 0
clicks = 0
rec_time = 0
upd_time = 0
dirty = None
w = None

start_time = 0
should_stop = False
stop_after = 1600
#stop_after = 10

def set_articles(articles_):
    # This is called once at the beginning - Can use it to do all our initialization
    global articles, M, b, lookup, dirty, w, Minv, start_time, should_stop

    # Dictionary containing 80 articles in [0,1]^6
    articles = articles_
    n_articles = len(articles)
    dim = 6

    M = np.empty((n_articles, dim, dim))
    Minv = np.empty((n_articles, dim, dim))
    M[:] = np.eye(dim)
    b = np.zeros((n_articles, dim))
    dirty = [True]*n_articles
    w = np.empty((n_articles, dim))

    lookup = dict(zip(articles.keys(), range(n_articles)))
    start_time = time.time()
    print('start_time', start_time)
    should_stop = False


def update(reward):
    global M, b, x, z, matches, clicks, upd_time, dirty, Minv, should_stop, r0, r1
    #start = timer()

    if reward >= 0 and not should_stop:
        matches += 1
        if reward == 1:
            reward = r1
            '''clicks += 1
            if clicks % 10 == 0:
                print("CTR: {:.2%} clicks: {} matches: {}".format(clicks / matches, clicks, matches))
                #print("total times:", rec_time, upd_time)
            '''
        else:
            reward = r0
        idx = lookup[x]
        M[idx] = M[idx] + np.outer(z, z)
        Minv[idx] = lin.inv(M[idx])
        # print(np.multiply(reward,z), reward, z, b[idx])
        b[idx] = b[idx] + np.multiply(reward, z)
        dirty[idx] = True
    #upd_time += timer() - start


def recommend(time_, user_features, choices):
    #start = timer()
    global articles, M, b, lookup, alpha, x, z, time, rec_time, dirty
    global w, Minv, should_stop, start_time, stop_after
    z = user_features
    #time = time_

    if not should_stop and time.time() - start_time > stop_after:
        should_stop = True

    ucb = np.zeros(len(choices))

    for i, c in enumerate(choices):
        idx = lookup[c]
        if not should_stop and dirty[idx]:
            # print(M[idx])
            # Minv = lin.inv(M[idx])
            #w[idx] = lin.solve(M[idx], b[idx])
            w[idx] = np.dot(Minv[idx], b[idx])
            # w[idx] = np.matmul(lin.inv(M[idx]), b[idx])
            # print(w[idx], z, alpha, M[idx])
            dirty[idx] = False

        ucb[i] = np.dot(w[idx], z) + alpha * np.sqrt(np.dot(z, np.matmul(Minv[idx], z)))
        # ucb[i] = np.dot(w, z) + alpha * np.sqrt(np.dot(z, np.dot(Minv, z)))
        # ucb[i] = np.dot(w[idx], z) + alpha * np.sqrt(np.dot(z, np.matmul(lin.inv(M[idx]), z)))
        # print(ucb[i])

    # print(ucb)
    max_idx = np.argmax(ucb)
    x = choices[max_idx]
    # print("recommend took {}".format(timer() - start))
    #rec_time += timer() - start
    return x
