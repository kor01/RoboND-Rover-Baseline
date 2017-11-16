import numpy as np


def vote_on_mesh(
    values, bins, min_val=None, max_val=None):
  min_val = min_val or values.min()
  max_val = max_val or values.max()

  interval = (max_val - min_val) / bins
  mesh = np.linspace(min_val, max_val, bins + 1)
  predicates = np.abs(mesh[:, None] - values) < interval
  predicates = predicates.astype('int32')
  num_votes = predicates.sum(axis=-1)
  return predicates, num_votes


def cluster_on_mesh(values, interval=None, bins=None):
  if len(values) == 0:
    return []

  if interval is not None:
    bins = int((values.max() - values.min()) / interval)

  predicates, num_votes = vote_on_mesh(values, bins)
  clusters = []
  current_cluster = []
  for i in range(bins):
    ids = np.where(predicates[i])[0]
    if len(ids) == 0:
      if len(current_cluster) != 0:
        clusters.append(list(set(current_cluster)))
        current_cluster = []
    else:
      current_cluster.extend(ids)

  if len(current_cluster) > 0:
    clusters.append(list(current_cluster))
  return clusters
