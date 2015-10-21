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


def pessimistic_accumulated_reward(y, y_hat, alphabet):
    dist, _,  = _edit_distance_matrix(y, y_hat)

    # Optimistic edit distance for every y_hat prefix
    optim_dist = dist.min(axis=0)
    pess_acc_reward = dist.argmin(axis=0) - optim_dist

    # Optimistic edit distance for every y_hat prefix plus a character
    optim_dist_char = numpy.tile(
        optim_dist[:, None], [1, len(alphabet)]) + 1
    pess_acc_char_reward = numpy.tile(
        pess_acc_reward[:, None], [1, len(alphabet)]) - 1
    for i in range(len(y)):
        for j in range(len(y_hat)):
            for c in range(len(alphabet)):
                # We consider appending a character c to y_hat
                # after the character y_hat[j] and aligning to y[i].
                # This means the first j characters of y_hat must
                # produce the first i character of y.
                cand_dist = dist[i, j] + (0 if alphabet[c] == y[i] else 1)
                if cand_dist < optim_dist_char[j, c]:
                    optim_dist_char[j, c] = cand_dist
                    pess_acc_char_reward[j, c] = i + 1 - cand_dist
    # Note, that each character j to the minimal i
    # out of the best ones. That makes the reward estimate pessimistic.
    return pess_acc_char_reward

def per_character_reward(y, y_hat, alphabet):
    y_hat_indices = [alphabet.find(c) for c in y_hat]
    reward = pessimistic_accumulated_reward(y, y_hat, alphabet)
    reward[1:] -= reward[:-1][numpy.arange(len(y_hat)), y_hat_indices][:, None]
    return reward


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
    optimistic_action : numpy.ndarray

    """
    dist, action  = _edit_distance_matrix(y, y_hat)

    best = numpy.zeros((len(y_hat), len(alphabet)), dtype='int64')
    optimistic_action = best.copy()
    for j in range(len(y_hat)):
        for c in range(len(alphabet)):
            best[j, c] = dist[:, j].min() + 1
            optimistic_action[j, c] = DELETION

    for i in range(len(y)):
        for j in range(len(y_hat)):
            for c in range(len(alphabet)):
                cost = 1 - (y[i] == alphabet[c])
                candidate_dist = dist[i, j] + cost
                if candidate_dist < best[j, c]:
                    best[j, c] = candidate_dist
                    optimistic_action[j, c] = COPY if cost else SUBSTITUTION

    return dist.min(axis=0)[:-1, None] - best, optimistic_action
