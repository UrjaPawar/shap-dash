import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
import random



class Neighborhood:
    def __init__(self, obj):
        self.feats = obj.features
        self.continuous = obj.continuous
        self.categorical = self.feats.copy()
        for feat in self.continuous:
            self.categorical.remove(feat)
        self.feature_range = obj.feature_range
        self.maxi = {feat: max(obj.df[feat]) for feat in self.continuous}
        self.mini = {feat: min(obj.df[feat]) for feat in self.continuous}
        self.precisions = obj.dec_precisions
        self.changes_count_dict = obj.changes_count_dict

    def get_probs(self, val, feat):
        combs = [(val, b) for b in self.feature_range[feat]]
        costs = {}
        den = len(self.feature_range[feat])
        for comb in combs:
            if comb in self.changes_count_dict[feat]:
                cost = 1 - (self.changes_count_dict[feat][comb]) / den
            elif comb[::-1] in self.changes_count_dict[feat]:
                cost = 1 - (self.changes_count_dict[feat][comb[::-1]]) / den
            costs[comb[1]] = cost
        res = [(costs[val] / sum(costs.values())) for val in self.feature_range[feat]]
        return res

    def calculateMahalanobis(self, y=None, data=None, cov=None):
        x = cdist(y, data, 'mahalanobis', VI=cov)
        return x

    def calculatel2(self, original_sample, new_sample):
        return cdist(original_sample, new_sample, 'euclidean')

    def generate_probabilities(self, n):
        center = (n - 1) / 2
        probabilities = np.zeros(n)
        for i in range(n):
            probabilities[i] = 1 - abs((i - center) / center)
        probabilities /= probabilities.sum()
        return probabilities[1:-1]

    def generate_range(self, start, end, precision):
        numbers = []
        step = round(10 ** (-precision),precision)
        while round(start, precision) < end:
            numbers.append(round(start, precision))
            start += step
        return numbers
    
    def get_continuous_samples(self, low, high, precision, size=1000, seed=None, probability=False):
        if seed is not None:
            np.random.seed(seed)
        high = round(high + 10 ** (-precision), precision)
        range_ = self.generate_range(low, high, precision)
        if probability:
            result = np.random.choice(range_, size, p=self.generate_probabilities(len(range_)+2))
        else:
            result = np.random.choice(range_, size)
        return result

    def get_neighbourhood_samples(self, fixed_features_values, original_sample, sampling_size, use_range=True, feature_range=None,
                                  bound_continuous=True, probability=False, sampling_random_seed=23):
        if sampling_random_seed is not None:
            random.seed(sampling_random_seed)
        samples = []
        for feature in self.feats:
            if len(fixed_features_values)!=0 and feature in fixed_features_values:
                sample = [original_sample[feature]] * sampling_size
            elif feature in self.continuous:
                if use_range:
                    half_range = round(0.5 * (self.changes_count_dict[feature]), self.precisions[feature])
                    if (half_range*2)>self.changes_count_dict[feature]:
                        half_range-=1
                    low = original_sample[feature] - half_range
                    high = original_sample[feature] + half_range
                    if bound_continuous:
                        low = low if low > self.mini[feature] else self.mini[feature]
                        high = high if high < self.maxi[feature] else self.maxi[feature]
                    sample = self.get_continuous_samples(
                        low, high, self.precisions[feature], size=sampling_size,
                        seed=sampling_random_seed, probability=probability)
                else:
                    sample = self.get_continuous_samples(
                        self.mini[feature], self.maxi[feature], self.precisions[feature], size=sampling_size,
                        seed=sampling_random_seed, probability=probability)
            else:
                if probability:
                    # print(feature,original_sample[feature])
                    sample = np.random.choice(self.feature_range[feature], size=sampling_size,
                                            p=self.get_probs(original_sample[feature], feature))
                else:
                    sample = random.choices(self.feature_range[feature], k=sampling_size)
            samples.append(sample)
        samples = pd.DataFrame(dict(zip(self.feats, samples)))
        return samples

    def generate_neighbourhood(self, feature_to_freeze, sample, columns, no_of_neighbours=200, probability=False,
                               bound=False, use_range=False, truly_random=False):
        features_to_change = self.feats.copy()
        if len(feature_to_freeze)!=0:
            for feat in feature_to_freeze:
                features_to_change.remove(feat)
        random_instances = self.get_neighbourhood_samples(
            feature_to_freeze, sample, no_of_neighbours, use_range,
            self.feature_range, bound, probability=probability, sampling_random_seed=23)
    
        if not truly_random:
            neighbors = pd.DataFrame(
                np.repeat([sample.values], no_of_neighbours, axis=0), columns=columns)
            # Loop to change one feature at a time, then two features, and so on.
            for num_features_to_vary in range(1, len(features_to_change) + 1):
                selected_features = np.random.choice(features_to_change, (no_of_neighbours, 1), replace=True)
                for k in range(no_of_neighbours):
                    neighbors.at[k, selected_features[k][0]] = random_instances.at[k, selected_features[k][0]]
            return neighbors
        else:
            return random_instances

    def get_importance(self, clf, original_sample, neighbors, weighted = False):
        original_output = clf.predict([original_sample])[0]
        outputs = clf.predict(neighbors)
        major_ = stats.mode(outputs)
        percent = len(outputs[outputs == original_output]) / len(neighbors)
        fi_arr = []
        if weighted:
            for feat in list(neighbors.columns):
                if feat in self.continuous:
                    percents = []
                    high = round(self.maxi[feat] + 10 ** (-self.precisions[feat]), self.precisions[feat])
                    range_all = np.array(self.generate_range(self.mini[feat], high, self.precisions[feat]))
                    half_range = round(0.5 * (self.changes_count_dict[feat]), self.precisions[feat])
                    low = original_sample[feat] - half_range
                    high_ = original_sample[feat] + half_range
                    low = low if low > self.mini[feat] else self.mini[feat]
                    high_ = high_ if high_ < self.maxi[feat] else self.maxi[feat]
                    high_ = round(high_ + 10 ** (-self.precisions[feat]), self.precisions[feat])
                    range_sample = self.generate_range(low, high_, self.precisions[feat])
                    lower_range = range_all[range_all < range_sample[0]]
                    upper_range = range_all[range_all > range_sample[-1]]
                    new_neighbors = neighbors.copy()
                    if len(lower_range)!=0:
                        lower_result = np.random.choice(lower_range, len(neighbors))
                        new_neighbors[feat] = lower_result
                        new_outputs = clf.predict(new_neighbors)
                        percents.append(len(outputs[new_outputs == original_output]) / len(neighbors))
                    if len(upper_range)!=0:
                        upper_result = np.random.choice(upper_range, len(neighbors))
                        new_neighbors[feat] = upper_result
                        new_outputs = clf.predict(new_neighbors)
                        percents.append(len(outputs[new_outputs == original_output]) / len(neighbors))
                    fi_arr.append([round((percent_-percent)/percent,2) for percent_ in percents])
                else:
                    if len(self.feature_range[feat])==2:
                        new_neighbors = neighbors.copy()
                        new_neighbors[feat] = self.feature_range[feat][0] * len(neighbors)
                        outputs_0 = clf.predict(new_neighbors)
                        new_neighbors[feat] = self.feature_range[feat][1] * len(neighbors)
                        outputs_1 = clf.predict(new_neighbors)
                        percent_0 = len(outputs[outputs_0 == original_output]) / len(neighbors)
                        percent_1 = len(outputs[outputs_1 == original_output]) / len(neighbors)
                        fi_arr.append([round((percent_0-percent)/percent,2), round((percent_1-percent)/percent,2)])
                    else:
                        percents=[]
                        for val in self.feature_range[feat]:
                            if val != original_sample[feat]:
                                new_neighbors = neighbors.copy()
                                new_neighbors[feat] = val * len(neighbors)
                                new_outputs = clf.predict(new_neighbors)
                                percents.append(len(outputs[new_outputs == original_output]) / len(neighbors))
                            else:percents.append(-1)
                        fi_arr.append([round((percent_-percent)/percent,2) for percent_ in percents])

            return fi_arr, percent, major_[0][0]

    # def get_importance_gen(self, clf, original_sample, neighbors, generic_neighbors):
    #     original_output = clf.predict([original_sample])[0]
    #     outputs = clf.predict(neighbors)
    #     major_ = stats.mode(outputs)
    #     percent = len(outputs[outputs == original_output]) / len(neighbors)
    #     fi_arr = []
    #     for feat in list(neighbors.columns):
    #         neighbors[feat] =
    #         generic_neighbors[]
    #
    #         return fi_arr, percent, major_[0][0]

