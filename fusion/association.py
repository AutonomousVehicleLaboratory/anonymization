import numpy as np

def compute_distance_matrix(trackers, 
                            observations, 
                            method = "euclidean", 
                            covariance = None,
                            trackers_class = None,
                            observations_class = None,
                            between_class_panelty = None,
                            angle_correction = False,
                            angle_dimension = None):
    """ Associate trackers and observations based on give method.
    Args:
        trackers: n by d array, n trackers, with d attributes.
        observations: m by d array, m observations, with d attributes.
        method: str giving the method for calculating distance
                1. euclidean distance
                2. mahanobis distance
                3. 2D IoU
                4. 3D IoU
        covariance: n by d by d array, n trackers, each d by d covariance matrix. used in mahanobis distance.
    """
    if len(trackers) == 0 or len(observations) == 0:
        return None
    
    if method == "euclidean":
        distance_matrix = np.linalg.norm(trackers.reshape(trackers.shape[0], 1, -1) - \
                                         observations.reshape(1, observations.shape[0],-1), axis=2)
    elif method == "mahalanobis":
        if angle_correction and angle_dimension is not None:
            offset_list = []
            for tracker in trackers:
                corrected_offset_list = []
                for observation in observations:
                    new_observation = np.array(observation).reshape(-1)
                    angle = new_observation[angle_dimension]

                    # within range
                    if angle >= np.pi: angle -= np.pi * 2
                    elif angle < -np.pi: angle += np.pi * 2

                    if angle >= tracker[angle_dimension] + np.pi:
                        angle -= np.pi * 2
                    elif angle <= tracker[angle_dimension] - np.pi:
                        angle += np.pi * 2

                    # acute angle
                    if angle >= tracker[angle_dimension] + np.pi/2:
                        angle -= np.pi
                    if angle < tracker[angle_dimension] - np.pi/2:
                        angle += np.pi
                    
                    # if angle != new_observation[angle_dimension]:
                    #     print(angle, new_observation[angle_dimension])
                    assert(-np.pi/2 <= angle - tracker[angle_dimension] < np.pi/2)
                    new_observation[angle_dimension] = angle

                    corrected_offset_list.append(tracker.reshape(-1) - new_observation)
                offset_list.append(corrected_offset_list)
            offset = np.array(offset_list).reshape(trackers.shape[0], observations.shape[0], 1, -1)
            temp = offset @ np.linalg.inv(covariance).reshape(trackers.shape[0], 1, covariance.shape[1], covariance.shape[2])
            distance_matrix = np.sqrt(temp @ np.transpose(offset, axes=[0, 1, 3, 2])).reshape(trackers.shape[0],-1)
        else:
            distance = trackers.reshape(trackers.shape[0], 1, -1) - observations.reshape(1, observations.shape[0],-1)
            distance = distance.reshape(trackers.shape[0], observations.shape[0], 1, -1)
            temp = distance @ np.linalg.inv(covariance).reshape(trackers.shape[0], 1, covariance.shape[1], covariance.shape[2])
            distance_matrix = np.sqrt(temp @ np.transpose(distance, axes=[0, 1, 3, 2])).reshape(trackers.shape[0],-1)
    elif method == "2diouxyxy":
        trackers = trackers.reshape(-1,4)
        observations = observations.reshape(-1,4)
        S_trackers = (trackers[:,2:3] - trackers[:,0:1]) * \
                     (trackers[:,3:4] - trackers[:,1:2])
        S_observations = (observations[:,2:3] - observations[:,0:1]) * \
                         (observations[:,3:4] - observations[:,1:2])
        MinXMax = np.maximum(trackers[:,0:1], observations[:,0:1].T)
        MinYMax = np.maximum(trackers[:,1:2], observations[:,1:2].T)
        MaxXMin = np.minimum(trackers[:,2:3], observations[:,2:3].T)
        MaxYMin = np.minimum(trackers[:,3:4], observations[:,3:4].T)
        x_diff = (MaxXMin - MinXMax)
        y_diff = (MaxYMin - MinYMax)
        x_diff[x_diff < 0] = 0
        y_diff[y_diff < 0] = 0
        Intersection = x_diff * y_diff
        Union = (S_trackers + S_observations.T) - Intersection
        IoU = Intersection / Union

        distance_matrix = 1 - IoU
    else:
        raise NotImplementedError

    if method == "euclidean" or method == "mahalanobis":
        if trackers_class is not None and observations_class is not None:
            same_class_table = np.char.equal(np.array(trackers_class).reshape(-1,1),
                                                np.array(observations_class).reshape(1,-1))
            distance_panelty = np.ones_like(same_class_table, dtype=np.float) * between_class_panelty
            distance_panelty[same_class_table] = 0
            distance_matrix += distance_panelty

    return distance_matrix

def greedy_match_by_distance_matrix(distance_matrix, threshold = None):
    """ Match paris using greedy algorithm based on the distance matrix.
    Args:
        distance_matrix: n by m matrix.
        threshold: max distance for matching.
    """
    associations = []
    row_inds = np.arange(distance_matrix.shape[0])
    col_inds = np.arange(distance_matrix.shape[1])
    ids = np.vstack([np.repeat(row_inds, distance_matrix.shape[1]), 
                     np.tile(col_inds, distance_matrix.shape[0])])

    distance_matrix = distance_matrix.reshape(-1)
    idx_order = np.argsort(distance_matrix)

    row_inds = row_inds.tolist()
    col_inds = col_inds.tolist()

    for idx in idx_order:
        row_id, col_id = ids[:,idx]
        if threshold is not None and distance_matrix[idx] > threshold:
            break
        elif row_id in row_inds and col_id in col_inds:
            row_inds.remove(row_id)
            col_inds.remove(col_id)
            associations.append([row_id, col_id])
        elif len(row_inds) == 0 or len(col_inds) == 0:
            break

    return associations, row_inds, col_inds

def match_by_distance_matrix(distance_matrix, method = "greedy", threshold = None):
    """ Match pairs based on the distance matrix.
    Args:
        distance_matrix: n by m matrix
        matching_method: str giving the methold for matching by distance matrix
                1. greedy
                2. hungarian
        threshold: max distance for matching.
    """
    if method == "greedy":
        associations = greedy_match_by_distance_matrix(distance_matrix, threshold)
    elif method == "hungarian":
        raise NotImplementedError
    else:
        raise NotImplementedError
    return associations


def associate(trackers,
              observations,
              distance_method = "euclidean",
              matching_method = "greedy",
              covariance = None,
              threshold = None,
              trackers_class = None,
              observations_class = None,
              between_class_panelty = None,
              angle_correction = False,
              angle_dimension = None):
    """ Associate trackers and observations based on give method.
    Args:
        trackers: n by d array, n trackers, with d attributes.
        observations: m by d array, m observations, with d attributes.
        distance_method: str given the method for calculating distance
                1. euclidean distance
                2. mahanobis distance
                3. 2D IoU
                4. 3D IoU
        matching_method: str giving the methold for matching by distance matrix
                1. greedy
                2. hungarian
        covariance: n by d by d array, n trackers, each d by d covariance matrix.
                    Used in mahanobis distance.
        threshold: max distance for matching.
    """

    distance_matrix = compute_distance_matrix(trackers, 
                                              observations,
                                              distance_method,
                                              covariance,
                                              trackers_class,
                                              observations_class,
                                              between_class_panelty,
                                              angle_correction,
                                              angle_dimension)

    # print('distance_matrix', distance_matrix)
    # print('trackers', len(trackers))
    # print('observations', len(observations))

    if distance_matrix is None:
        associations = [], [i for i in range(len(trackers))], [j for j in range(len(observations))]
    else:
        associations = match_by_distance_matrix(distance_matrix, matching_method, threshold)

    # print('associations:', associations)
    return associations