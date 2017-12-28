import numpy as np
#from sklearn.kernel_approximation import RBFSampler

# Constants
n_samples_orig = 2000
n_features_orig = 400

## Parameters
print_interval = 50 # Set to 0 to disable
n_iter = int(4.0 * n_samples_orig) # Only n_iter <= n_samples supported so far

# SGD
lr = 5.0
decay = 0.00000000001 # Decay for learning rate. Set to None to disable
regulariz_param = 0.0

# Adam
lr_adam = 0.08
beta1 = 0.9
beta2 = 0.999
eps = 0.000001

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    
    n_samples, n_features = X.shape
    n_features_target = 1000

    # For the ith datapoint xi (dim n_features) we constract a new feature map
    # of dim D=n_features_target as follows:
    # sqrt(2/D) * [cos(w0^T xi + b), cos(w1^T xi + b), ...., cos(wD^T xi + b)]
    #W = np.random.randn(n_features, n_features_target)
    '''
    W = np.random.randn(n_features, n_features_target)
    b = 2 * np.pi * np.random.rand(n_features_target) # Note that this gets broadcasted over the samples (not over features)
    X_res = np.sqrt(2.0 / n_features_target) * np.cos(np.matmul(X, W) + b)
    '''

    np.random.seed(42)
    idx = np.random.permutation(n_features)[:128]

    XX = np.empty((n_samples, len(idx)**2))
    for j in range(len(idx)):
        for k in range(len(idx)):
            XX[:, j*len(idx)+k] = np.multiply(X[:, j], X[:, k])

    return np.concatenate([X, XX, np.power(X, 2)], axis=1)



def mapper(key, value):
    # key: None
    # value: list of strings representing part of the training set: [n_samples, n_features]

    # Parse input
    inputs = [list(map(float, sample.split())) for sample in value]
    targets = np.empty(n_samples_orig)
    X = np.empty((n_samples_orig, n_features_orig))
    for i, sample in enumerate(inputs):
        targets[i] = sample[0]
        X[i, :] = sample[1:]

    X = transform(X)
    n_samples, n_features = X.shape

    # Initialize weights (TODO: ideas ?)
    w = np.random.randn(n_features)  # Standard normal distribution
    m = np.zeros(n_features) # adam
    v = np.zeros(n_features) # adam

    # Training
    epochs = (n_iter // n_samples) + 1 # Number of (not necessarily full) epochs
    remaining_iter = n_iter % n_samples
    tot_loss = 0
    avg_losses = []
    for e in range(epochs):
        if e == epochs - 1:
            perm = np.random.permutation(n_samples)[:remaining_iter]
        else:
            perm = np.random.permutation(n_samples)

        for i, (x, y) in enumerate(zip(X[perm, :], targets[perm])):
            global_step = e*perm.shape[0] + i
            loss, grad = hinge_loss(x, y, w, regulariz_param)
            tot_loss += loss
            #w = update_weights(w, grad, global_step)
            w, m, v = update_weights_adam(w, grad, m, v, global_step)

            if print_interval != 0 and global_step % print_interval == 0:
                avg_loss = tot_loss/print_interval
                avg_losses.append(avg_loss)
                print('[step {}] AVG {}'.format(global_step, avg_loss))
                tot_loss = 0
    
    yield 'weights', w


def hinge_loss(x, y, w, lam):
    wx = np.dot(w, x)
    loss = lam*np.dot(w, w) + max(0, 1 - y*wx)
    
    # TODO: Subgradient at 1?
    grad = 2*lam*w
    if y*wx < 1:
        grad -= y*x

    return loss, grad

def update_weights(w, grad, iterations):
    global lr, decay
    if decay is not None:
        lr *= 1.0/(1.0 + decay*iterations)
    return np.subtract(w, lr*grad)

def update_weights_adam(w, grad, m, v, iterations):
    # https://medium.com/@nishantnikhil/adam-optimizer-notes-ddac4fd7218
    m = beta1*m + (1 - beta1)*grad
    v = beta2*v + (1 - beta2)*np.power(grad, 2)
    m_hat = m/(1 - beta1**(iterations+1))
    v_hat = v/(1 - beta2**(iterations+1))
    return np.subtract(w, np.divide(lr_adam*m_hat, np.sqrt(v_hat) + eps)), m, v

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    yield np.mean(values, axis=0)
