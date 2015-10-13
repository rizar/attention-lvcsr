import numpy

COPY = 0
INSERTION = 1
DELETION = 2
SUBSTITUTION = 3

INFINITY = 10 ** 9

def _edit_distance_matrix(y, y_hat):
    """Returns the matrix of edit distances.

    Returns
    -------
    dist : numpy.ndarray
        dist[i, j] is the edit distance between the first
        i characters of y and the first j characters of y_hat.
    action : numpy.ndarray
        action[i, j] is the action applied to y_hat[j - 1]  in a chain of
        optimal actions transducing y_hat[:j] into y[:i].

    """
    dist = numpy.zeros((len(y) + 1, len(y_hat) + 1), dtype='int64')
    action = dist.copy()
    for i in xrange(len(y) + 1):
        dist[i][0] = i
    for j in xrange(len(y_hat) + 1):
        dist[0][j] = j

    for i in xrange(1, len(y) + 1):
        for j in xrange(1, len(y_hat) + 1):
            if y[i - 1] != y_hat[j - 1]:
                cost = 1
            else:
                cost = 0
            insertion_dist = dist[i - 1][j] + 1
            deletion_dist = dist[i][j - 1] + 1
            substitution_dist = dist[i - 1][j - 1] + 1 if cost else INFINITY
            copy_dist = dist[i - 1][j - 1] if not cost else INFINITY
            best = min(insertion_dist, deletion_dist,
                       substitution_dist, copy_dist)

            dist[i][j] = best
            if best == insertion_dist:
                action[i][j] = action[i - 1][j]
            if best == deletion_dist:
                action[i][j] = DELETION
            if best == substitution_dist:
                action[i][j] = SUBSTITUTION
            if best == copy_dist:
                action[i][j] = COPY

    return dist, action


def edit_distance(y, y_hat):
    """Edit distance between two sequences.

    Parameters
    ----------
    y : str
        The groundtruth.
    y_hat : str
        The recognition candidate.

    the minimum number of symbol edits (i.e. insertions,
    deletions or substitutions) required to change one
    word into the other.

    """
    return _edit_distance_matrix(y, y_hat)[-1][-1]


def wer(y, y_hat):
    return edit_distance(y, y_hat) / float(len(y))


def optimistic_error_matrix(y, y_hat, alphabet):
    """Optimistic error estimate.

    Parameters
    ----------
    y : str
        The groundtruth.
    y_hat : str
        The recognition candidate.
    alphabet : iterable

    Returns
    -------
    errors : numpy.ndarray
        A matrix of a shape (len(y_hat) + 1, num_characters).  Let best[j]
        be the least possible edit distance that you can get by optimally
        continuing y_hat[:j+1] (not adding any characters is also allowed).
        Let best[j, c] be the least you can get by continuing the
        concatenation of y_hat[j+1] and the character number c. errors[j,
        c] is defined as best[j, c] - best[j - 1], that indicates if
        continuing y_hat[:j+1] with the character number c decreases the
        optimistic estimate of your future error rate.

    """
    dist, _  = _edit_distance_matrix(y, y_hat)

    best = numpy.zeros((len(y_hat), len(alphabet)), dtype='int64')
    for j in range(len(y_hat)):
        for c in range(len(alphabet)):
            best[j, c] = dist[:, j].min() + 1

    for i in range(len(y)):
        for j in range(len(y_hat)):
            for c in range(len(alphabet)):
                cost = 1 - (y[i] == alphabet[c])
                best[j, c] = min(best[j, c], dist[i, j] + cost)

    return dist.min(axis=0)[:-1, None] - best
