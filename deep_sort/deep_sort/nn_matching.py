# vim: expandtab:ts=4:sw=4
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _cka_distance(a, b, scale=0.2, data_is_normalized=False):
    """Compute cka distance between `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a vector len(b) such that eleement (j)
        contains the cka distance between `a` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        
    #######################################################
    sigi = np.median(_pdist(a, b))
    sigv = sigi*np.array([0.3,0.4,0.5,0.6,0.7,0.8]).astype(np.float32)
    
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    
    # create a batch tensor with first dimension indicating the number of 
    # detections
    b = tf.reshape(b,(-1,1,tf.shape(a)[1]))
    c = tf.constant([1,tf.shape(a)[0].numpy(),1],tf.int32)
    b = tf.tile(b, c) # repeat each detection to match the number of instances in tracker
    
    #kernels###############################################
    i = 2
    scalar_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1, length_scale=sigv[i])
    scalar_kernely = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1, length_scale=sigv[i])
    
    a_k = tf.expand_dims(a,0)
    # create a broadcastable shape for creating gram matrices
    c = tf.constant([tf.shape(b)[0].numpy(),1,1], tf.int32)
    a_k = tf.tile(a_k, c)
    
    # apply kernels########################################
    k = scalar_kernel.matrix(a_k, b)
    l = scalar_kernely.matrix(a, a)
    l = tf.tile(tf.expand_dims(l,0), c)
    
    #centralizar###########################################
    # N = tf.shape(l)[1]
    # N2 = tf.cast(tf.shape(l)[1],dtype=tf.float32)
    
    #empirical centering matrix
    # h = tf.eye(N) - (1.0/N2)*tf.ones([N,1])*tf.ones([1,N])     
    # h = tf.tile(tf.expand_dims(h,0), c)
    
    kc = k#tf.matmul(h,tf.matmul(k,h))
    lc = l#tf.matmul(h,tf.matmul(l,h))
    
    # CKA based functional
    trkl = tf.linalg.trace(tf.matmul(kc,lc,transpose_b=True))
    trkl = trkl / tf.linalg.norm(trkl, keepdims=True)
    # trkk = tf.linalg.trace(tf.matmul(kc,kc))
    # trll = tf.linalg.trace(tf.matmul(lc,lc))
    
    # f = 1. -trkl/tf.sqrt(tf.math.multiply(trkk,trll))
    # print(1. -trkl)
    
    #####funcion de costo##################################
    return 1. -trkl# cka cost function

# Define custom distance
# centered kernel alignmet (CKA-based)
#http://www.jmlr.org/papers/volume13/cortes12a/cortes12a.pdf
#https://www.frontiersin.org/articles/10.3389/fnins.2017.00550/full
def _nn_cka_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cka).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    scale : scale parameter for the gaussian kernels

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cka distance to set of instances `x`.

    """
    distances = _cka_distance(x, y)
    # Return a function
    return distances


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean","cosine", or cka.
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        elif metric == "cka":
            self._metric = _nn_cka_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        '''Este código es muy inteligente, permite controlar la cantidad 
        de features que tiene un objeto que se está siguiendo (track). El 
        máximo por defecto es 100. Entonces, para el track i, se guardan las 
        características obtenidas en los 100 frames pasados. Si llega uno nuevo,
        el primer feature sale del qeue "self.samples[target]", y en su lugar,
        se guarda el último feature obtenido.'''
        
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
