from __future__ import division
import numpy as np
import numpy.linalg as lin
import time

alpha = 3.8
r1 = 18.5
r0 = -1.03
articles = None
M = None
Minv = None
b = None
lookup = None
x = None
z = None
dirty = None
w = None

start_time = 0
should_stop = False
stop_after = 1600

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
    should_stop = False


def update(reward):
    global M, b, x, z, dirty, Minv, should_stop, r1, r0

    if reward >= 0 and not should_stop:
        if reward == 1:
            reward *= r1
        else:
            reward = r0
        idx = lookup[x]
        M[idx] = M[idx] + np.outer(z, z)
        Minv[idx] = lin.inv(M[idx])
        b[idx] = b[idx] + np.multiply(reward, z)
        dirty[idx] = True


def recommend(time_, user_features, choices):
    global articles, M, b, lookup, alpha, x, z, time, dirty
    global w, Minv, should_stop, start_time, stop_after
    z = user_features

    if not should_stop and time.time() - start_time > stop_after:
        should_stop = True

    ucb = np.zeros(len(choices))

    for i, c in enumerate(choices):
        idx = lookup[c]
        if not should_stop and dirty[idx]:
            w[idx] = np.matmul(Minv[idx], b[idx])
            dirty[idx] = False
        ucb[i] = np.dot(w[idx], z) + alpha * np.sqrt(np.dot(z, np.matmul(Minv[idx], z)))

    max_idx = np.argmax(ucb)
    x = choices[max_idx]
    return x