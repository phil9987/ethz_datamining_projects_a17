import numpy as np 
from scipy.spatial import distance
import math, time

convergence_value = 0.000001

def kplusplus_init(samples):

    start = time.time()

    num_samples = samples.shape[0]

    # Choose random first cluster center
    centroids = np.zeros((200, 250))
    idx  = np.random.randint(low=0, high=num_samples-1)
    centroids[0,:] = samples[idx, :]
    
    # Flag points that have been used
    used = np.array([False]*num_samples)
    used[idx] = True
    
    # Repeatedly choose each of the samples with probability proportional to the
    # distance to their closest cluster center as next cluster center
    for i in range(1, 200):
        #print(i)

        # Compute distances for each point to closest centroid
        sq_dsts = distance.cdist(samples[~used], centroids[:i], 'euclidean')
        dist_to_closest_centroid = np.amin(sq_dsts, axis=1) # Shape [num_samples]
        
        # Normalize
        p_e = np.exp(dist_to_closest_centroid)
        pr = p_e / np.sum(p_e)

        # Sample next centroid and remove this sample from choices
        next_idx = np.random.choice(len(dist_to_closest_centroid), p=pr)
        centroids[i, :] = samples[~used][next_idx, :]
        used[next_idx] = True

    #print('k++ init duration {}'.format(time.time()-start))

    return centroids

# returns the index of the closest centroid
def closest_center_idx(point, centroids):
    distances_point_to_centroids = distance.cdist(np.array(point, ndmin=2),centroids, 'euclidean')
    idx = np.argmin(distances_point_to_centroids)
    #if math.isnan(distances_point_to_centroids[0][idx]):
    #    print ('point', point, 'closest centroid', centroids[idx])
    return (idx, distances_point_to_centroids[0][idx])


def mapper(key, value):
    # key: None
    # value: one line of input file

    yield "centroids", value

def reducer(key, values):
    value = values

    #print('key',key)
    #print('value', np.shape(value))
    samples = value
    num_samples, num_features = np.shape(samples)

    centroids = kplusplus_init(samples)
    #print('centroids initialized', np.shape(centroids))


    converged = False
    last_tot_loss = -1
    start = time.time()
    while not converged:

        sum_variables = np.zeros_like(centroids)
        sum_counter = np.zeros(200)

        tot_loss = 0
        for sample in samples:
            # find closest center
            center_idx, sq_dist = closest_center_idx(sample, centroids)
            # we add the distance in the beginning of the loop s.t. we don't need to calculate it twice per iteration
            #print('sq_dist', sq_dist)
            tot_loss += sq_dist

            # add current datapoint to corresponding sum variable (position in array) -> center is gonna be relocated to mean of all assigned data points
            sum_variables[center_idx] += sample
            sum_counter[center_idx] += 1

        # Update centroids
        for i, s in enumerate(sum_counter):
            if s > 0:
                centroids[i,:] = sum_variables[i,:] / s
            else:
                # Do this s.t. inactive the centroids are recognizeable
                centroids[i,:] = np.zeros(num_features)

        tot_loss /= num_samples 
        difference = abs(last_tot_loss - tot_loss)
        #print('loss_difference: ', difference)
        if last_tot_loss is not -1 and difference < convergence_value: 
            converged = True
        else:
            last_tot_loss = tot_loss

    #print('k-means duration {}'.format(time.time()-start))
    yield centroids
