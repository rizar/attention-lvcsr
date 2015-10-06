import numpy

def _edit_distance_matrix(y, y_hat):
    """Returns the matrix of edit distances.

    Returns
    -------
    dist : numpy.ndarray
        dist[i, j] is the edit distance between the first
        i characters of y and the first j characters of y_hat.

    """
    plen, tlen = len(y_hat), len(y)

    dist = [[0 for i in range(tlen+1)] for x in range(plen+1)]
    for i in xrange(plen+1):
        dist[i][0] = i
    for j in xrange(tlen+1):
        dist[0][j] = j

    for i in xrange(plen):
        for j in xrange(tlen):
            if y_hat[i] != y[j]:
                cost = 1
            else:
                cost = 0
            dist[i+1][j+1] = min(
                dist[i][j+1] + 1, #  deletion
                dist[i+1][j] + 1, #  insertion
                dist[i][j] + cost #  substitution
                )

    return numpy.array(dist).T


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
    dist = _edit_distance_matrix(y, y_hat)

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
