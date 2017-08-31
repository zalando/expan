from heapq import heapify, heappush, heappop

def categorical_binning(items, n_bins):
    """ Performs greedy (non-optimal) binning
    :param items: list of items to bin according to their frequencies
    :return: list of bins [(total_weight, [items])]
    """

    # count items
    weights = {}
    for item in items:
        if not item in weights:
            weights[item] = 1
        else:
            weights[item] += 1

    # we need items sorted in decreasing order
    pairs = sorted([(weight, [item]) for (item, weight) in weights.items()], reverse=True)

    # take min(n_bins, len(pairs)) haviest pairs as initial bins
    bins = [(weight, labels) for (_, (weight, labels)) in zip(range(n_bins), pairs)]

    # too little data, just return what we have so far
    if len(pairs) <= n_bins:
        return bins

    heapify(bins)
    
    # go through pairs, from heaviest to lightest
    for (pair_weight, pair_labels) in pairs[n_bins:]:
        # take the lightest bin
        bin_weight, bin_labels = heappop(bins)
        # add the heaviest item to it
        heappush(bins, (bin_weight + pair_weight, bin_labels + pair_labels))

    return bins
