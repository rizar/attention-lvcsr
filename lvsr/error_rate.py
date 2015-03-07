def levenshtein(y, y_hat):
    """levenshtein distance between two sequences.

    the minimum number of symbol edits (i.e. insertions,
    deletions or substitutions) required tochange one
    word into the other.

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

    return dist[-1][-1]

def wer(y, y_hat):
    return levenshtein(y, y_hat) / float(len(y))
