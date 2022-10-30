import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy import spatial

def is_float(value):
    """ Checks if a value can be cast as float and returns a boolean

    Args:
        value (any type):   a numeric or non-numeric value

    Returns:
        True (boolean):     if value can be converted to float
        False (boolean):    if value cannot be converted to float
    """
    try:
        float(value)
    except ValueError:
        return False
    else:
        return True

class feat_calc:
    def __init__(self, xy_array):
        # Confirm input is of type np.array([])
        if not isinstance(xy_array, np.ndarray):
            raise TypeError("Input must be of type numpy.ndarray")
        # Confirm input array contains numeric values
        is_numeric = lambda x: list(map(is_float, x))
        numeric_check = is_numeric(xy_array.flatten())
        if ~np.all(numeric_check):
            raise TypeError("Input must be numeric data of type numpy.ndarray")
        # Confirm input array contains more than one point
        if len(xy_array) < 2:
            raise ValueError("Input numpy.ndarray must contain more than one point")
        # Confirm input array is a 2D array
        _, dim = xy_array.shape
        if dim != 2:
            raise ValueError("Input numpy.ndarray must be a 2D numpy.ndarray of datapoints")
        self.xy_array = xy_array

    # Net Displacement & Effiency Features
    def NetDisplacementEfficiency(self):
        points_array = self.xy_array
        net_displacement_value = np.linalg.norm(points_array[0]-points_array[-1])
        netDispSquared = pow(net_displacement_value, 2)
        points_a = points_array[1:, :]
        points_b = points_array[:-1, :]
        dist_ab_SumSquared = sum(pow(np.linalg.norm(points_a-points_b, axis=1), 2))
        efficiency_value = netDispSquared / ((len(points_array)-1) * dist_ab_SumSquared)
        return net_displacement_value, efficiency_value

    # Straightness & Bending Features
    def SummedSinesCosines(self):
        points_array = self.xy_array
        # Look for repeated positions in consecutive frames
        compare_against = points_array[:-1]
        # make a truth table identifying duplicates
        duplicates_table = points_array[1:] == compare_against
        # Sum the truth table across the rows, True = 1, False = 0
        duplicates_table = duplicates_table.sum(axis=1)
        # If both x and y are duplicates, value will be 2 (True + True == 2)
        duplicate_indices = np.where(duplicates_table == 2)
        # Remove the consecutive duplicates before sin, cos calc
        points_array = np.delete(points_array, duplicate_indices, axis=0)
        # Generate three sets of points
        points_set_a = points_array[:-2]
        points_set_b = points_array[1:-1]
        points_set_c = points_array[2:]
        # Generate two sets of vectors
        ab = points_set_b - points_set_a
        bc = points_set_c - points_set_b
        # Evaluate sin and cos values
        cross_products = np.cross(ab, bc)
        dot_products = np.einsum('ij,ij->i', ab, bc)
        product_magnitudes_ab_bc = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1)
        cos_vals = dot_products / product_magnitudes_ab_bc
        cos_mean_val = np.mean(cos_vals)
        sin_vals = cross_products / product_magnitudes_ab_bc
        sin_mean_val = np.mean(sin_vals)
        return sin_mean_val, sin_vals, cos_mean_val, cos_vals

    # Radius of Gyration and Asymmetry
    def RadiusGyrationAsymmetrySkewnessKurtosis(self):
        points_array = self.xy_array
        center = points_array.mean(0)
        normed_points = points_array - center[None, :]
        radiusGyration_tensor = np.einsum('im,in->mn', normed_points, normed_points)/len(points_array)
        eig_values, eig_vectors = np.linalg.eig(radiusGyration_tensor)
        radius_gyration_value = np.sqrt(np.sum(eig_values))
        asymmetry_numerator = pow((eig_values[0] - eig_values[1]), 2)
        asymmetry_denominator = 2 * (pow((eig_values[0] + eig_values[1]), 2))
        asymmetry_value = - math.log(1 - (asymmetry_numerator / asymmetry_denominator))
        maxcol = list(eig_values).index(max(eig_values))
        dominant_eig_vect = eig_vectors[:, maxcol]
        points_a = points_array[:-1]
        points_b = points_array[1:]
        ba = points_b - points_a
        proj_ba_dom_eig_vect = np.dot(ba, dominant_eig_vect) / np.power(np.linalg.norm(dominant_eig_vect), 2)
        skewness_value = stats.skew(proj_ba_dom_eig_vect)
        kurtosis_value = stats.kurtosis(proj_ba_dom_eig_vect)
        return radius_gyration_value, asymmetry_value, skewness_value, kurtosis_value

    # Fractal Dimension
    def FractalDimension(self):
        points_array = self.xy_array
        # Check if points are on the same line:
        x0, y0 = points_array[0]
        points = [ (x, y) for x, y in points_array if ( (x != x0) or (y != y0) ) ]
        slopes = [ ((y - y0) / (x - x0)) if (x != x0) else None for x, y in points ]
        if all( s == slopes[0] for s in slopes):
            raise ValueError("Fractal Dimension cannot be calculated for points that are collinear")
        total_path_length = np.sum(pow(np.sum(pow(points_array[1:, :] - points_array[:-1, :], 2), axis=1), 0.5))
        stepCount = len(points_array)
        candidates = points_array[spatial.ConvexHull(points_array).vertices]
        dist_mat = spatial.distance_matrix(candidates, candidates)
        maxIndex = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        largestDistance = dist_mat[maxIndex]
        fractal_dimension_value = math.log(stepCount) / math.log(stepCount * largestDistance * math.pow(total_path_length, -1))
        return fractal_dimension_value

    # Gaussianity & Alpha(tau) at a specific lag time
    def GaussianityAlphatau(self, tau=5, alpha='upper'):
        # The default of tau=5 was determined from the linear region of alpha(tau)
        # Where alpha(tau) = d log MSD / d log tau
        points_array = self.xy_array
        if tau == 0:
            return 0, 0, 0, 0
        if alpha == 'upper':
            lag_range = [tau, tau + 1]
            output_index = 0
        elif alpha == 'lower':
            lag_range = [tau - 1, tau]
            output_index = 1
        gauss_lags = []
        gauss_vals = []
        quartic_vals = []
        tamsd_vals = []
        for lag in lag_range:
            points_a = points_array[:-lag]
            points_b = points_array[lag:]
            dist_ab = np.linalg.norm(points_b - points_a, axis=1)
            tamsd = (1 / (len(points_array) - lag)) * np.sum(np.power(dist_ab, 2))
            quartic_moment = (1 / (len(points_array) - lag)) * np.sum(np.power(dist_ab, 4))
            gaussianity = ((1 * quartic_moment) / (2 * np.power(tamsd, 2))) - 1
            gauss_lags.append(lag)
            gauss_vals.append(gaussianity)
            quartic_vals.append(quartic_moment)
            tamsd_vals.append(tamsd)
        Gaussianity = gauss_vals[output_index]
        tamsd = tamsd_vals[output_index]
        quartic_moment = quartic_vals[output_index]
        alphatau_val = float(np.diff(np.log(tamsd_vals)) / np.diff(np.log(gauss_lags)))
        return Gaussianity, tamsd, quartic_moment, alphatau_val

    # Track Length
    def TrackLength(self):
        points_array = self.xy_array
        return len(points_array)


def calculate_all_features(tracks_df):
    """Takes in a dataframe of tracks with sorted IDs

    Args:
        tracks_df ([pandas dataframe]): tracks dataframe of the form ['Frame', 'X', 'Y', 'ID']

    Returns:
        [list]: one for each feature
    """
    if not isinstance(tracks_df, pd.DataFrame):
        raise TypeError("Input for calculate_all_features must be of type pandas dataframe")
    # TODO add more exceptions for dataframe without proper column names and shape
    # Dataframe should have unique IDs for every track, no duplicates, even across classifications
    trackID_list = sorted(list(tracks_df.ID.unique()))
    netDisps, sumCosines, sumSines, efficiency, asymmetry, radGyrs, skewness, kurtosis, fracDims, gaussianity, alphatau, trackLens = ([] for i in range(12))
    print('')
    for index, trackID in enumerate(trackID_list):
        # Output which track is being processed as a progress bar
        print(f'\r                                                             \r', end='')
        print(f'On Track {index+1} out of {len(trackID_list)}', end='')
        mask = (tracks_df['ID'] == trackID)
        # Drop any skipped frames
        points_array = np.array(tracks_df.loc[mask][['X', 'Y']].dropna())
        # Load a single trajectory
        single_trajectory = feat_calc(points_array)
        # Calculate the features
        tempNetDisp, tempEffic = single_trajectory.NetDisplacementEfficiency()
        tempSumSin, _, tempSumCos, _  = single_trajectory.SummedSinesCosines()
        tempRadGyr, tempAsymm, tempSkew, tempKurt  = single_trajectory.RadiusGyrationAsymmetrySkewnessKurtosis()
        tempFracDim = single_trajectory.FractalDimension()
        tempGauss, _, _, tempAlphatau = single_trajectory.GaussianityAlphatau(tau=5)
        tempTrkLen  = single_trajectory.TrackLength()
        # Consolidate values to list
        netDisps.append(tempNetDisp)
        efficiency.append(tempEffic)
        sumCosines.append(tempSumCos)
        sumSines.append(tempSumSin)
        radGyrs.append(tempRadGyr)
        asymmetry.append(tempAsymm)
        skewness.append(tempSkew)
        kurtosis.append(tempKurt)
        fracDims.append(tempFracDim)
        gaussianity.append(tempGauss)
        alphatau.append(tempAlphatau)
        trackLens.append(tempTrkLen)
    print('') # Blank line to force new line character and carriage return
    return netDisps, sumCosines, sumSines, efficiency, asymmetry, radGyrs, skewness, kurtosis, fracDims, gaussianity, alphatau, trackLens


if __name__ == '__main__':
    pass