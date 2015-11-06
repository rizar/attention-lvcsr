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
    action : numpy.ndarray
        action[i, j] is the action applied to y_hat[j - 1]  in a chain of
        optimal actions transducing y_hat[:j] into y[:i].
        i characters of y and the first j characters of y_hat.

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
    return _edit_distance_matrix(y, y_hat)[0][-1, -1]


def wer(y, y_hat):
    return edit_distance(y, y_hat) / float(len(y))


def reward_matrix(y, y_hat, alphabet, eos_label):
    dist, _,  = _edit_distance_matrix(y, y_hat)
    y_alphabet_indices = [alphabet.index(c) for c in y]
    if y_alphabet_indices[-1] != eos_label:
        raise ValueError("Last character of the groundtruth must be EOS")

    # Optimistic edit distance for every y_hat prefix
    optim_dist = dist.min(axis=0)
    pess_reward = -optim_dist

    # Optimistic edit distance for every y_hat prefix plus a character
    optim_dist_char = numpy.tile(
        optim_dist[:, None], [1, len(alphabet)]) + 1
    pess_char_reward = numpy.tile(
        pess_reward[:, None], [1, len(alphabet)]) - 1
    for i in range(len(y)):
        for j in range(len(y_hat) + 1):
            c = y_alphabet_indices[i]
            cand_dist = dist[i, j]
            if cand_dist < optim_dist_char[j, c]:
                optim_dist_char[j, c] = cand_dist
                pess_char_reward[j, c] = -cand_dist
    for j in range(len(y_hat) + 1):
        # Here we rely on y[-1] being eos_label
        pess_char_reward[j, eos_label] = -dist[len(y) - 1, j]
    return pess_char_reward

def gain_matrix(y, y_hat, alphabet, given_reward_matrix=None):
    y_hat_indices = [alphabet.index(c) for c in y_hat]
    reward = (given_reward_matrix.copy() if given_reward_matrix is not None
              else reward_matrix(y, y_hat, alphabet))
    reward[1:] -= reward[:-1][numpy.arange(len(y_hat)), y_hat_indices][:, None]
    return reward
