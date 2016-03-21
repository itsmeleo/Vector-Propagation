
def single_cell_propagation(params):
    C, idx3d, nb_max, dist_max, to_check_self, R, pos, posVF, successor, predecessor = params
    # idx3d = LT.kdtrees[t]
    dists, closest_cells = idx3d.query(posVF[C], nb_max)
    closest_cells = np.array(to_check_self)[list(closest_cells)]
    max_value = np.max(np.where(dists<dist_max))
    cells_to_keep = closest_cells[:max_value]
    # med = median_average_bw(cells_to_keep, R, pos)
    # print type (cells_to_keep)
    subset_dist = [np.mean([pos[cii] for cii in predecessor[ci]], axis=0) - pos[ci] for ci in cells_to_keep if not ci in R]
    if subset_dist != []:
        med_distance = spatial.distance.squareform(spatial.distance.pdist(subset_dist))
        med = subset_dist[np.argmin(np.sum(med_distance, axis=0))]
    else:
        med = [0, 0, 0]
    return C, med
