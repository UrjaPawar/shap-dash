import numpy as np
from tqdm import tqdm
import pandas as pd
from evaluation import Evaluation, measure_kendall_correlation
from shap_lime_cf import XAI
from neighborhood import Neighborhood
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


# balanced - 0 for skewed towards same, 1 for skewed towards other, and 2 for balanced and -1 for not counting
# restricted - -1 for false, 0 for inside, 1 for outside
minmax = MinMaxScaler()
repeat_configs = ["dist_random_opposite","dist_random_same","dist_random_balanced","dist_same","dist_balanced",
                  "dist_opposite","generic_random_opposite","generic_random_same","generic_random_balanced",
                  "generic_opposite","generic_same","generic_balanced"]
neighborhood_configs = {
    "dist": {"distance": True, "training": False, "custom":False, "balanced": -1, "restricted": -1, "random": False},
    "dist_random_opposite": {"distance": True, "training": False, "custom": False, "balanced": 1, "restricted": -1, "random": True},
    "dist_random_same": {"distance": True, "training": False, "custom": False, "balanced": 0, "restricted": -1, "random": True},
    "dist_random_balanced": {"distance": True, "training": False, "custom": False, "balanced": 2, "restricted": -1, "random": True},
    "dist_opposite": {"distance": True, "training": False, "custom": False, "balanced": 1, "restricted": -1, "random": False},
    "dist_same": {"distance": True, "training": False, "custom": False, "balanced": 0, "restricted": -1, "random": False},
    "dist_balanced": {"distance": True, "training": False, "custom": False, "balanced": 2, "restricted": -1, "random": False},
    "dist_outside": {"distance": True, "training": False, "custom": False, "balanced": -1, "restricted": 1, "random": False},
    "dist_inside": {"distance": True, "training": False, "custom": False, "balanced": -1, "restricted": 0, "random": False},
    "dist_random": {"distance": True, "training": False, "custom": False, "balanced": -1, "restricted": -1, "random": True},
    "dist_random_outside": {"distance": True, "training": False, "custom": False, "balanced": -1, "restricted": 1, "random": True},
    "dist_random_inside": {"distance": True, "training": False, "custom": False, "balanced": -1, "restricted": 0, "random": True},

    "generic": {"distance": False, "training": False, "custom": False, "balanced": -1, "restricted": -1, "random": False},
    "generic_random_opposite": {"distance": False, "training": False, "custom": False, "balanced": 1, "restricted": -1, "random": True},
    "generic_random_same": {"distance": False, "training": False, "custom": False, "balanced": 0, "restricted": -1, "random": True},
    "generic_random_balanced": {"distance": False, "training": False, "custom": False, "balanced": 2, "restricted": -1, "random": True},
    "generic_opposite": {"distance": False, "training": False, "custom": False, "balanced": 1, "restricted": -1, "random": False},
    "generic_same": {"distance": False, "training": False, "custom": False, "balanced": 0, "restricted": -1, "random": False},
    "generic_balanced": {"distance": False, "training": False, "custom": False, "balanced": 2, "restricted": -1, "random": False},
    "generic_outside": {"distance": False, "training": False, "custom": False, "balanced": -1, "restricted": 1, "random": False},
    "generic_inside": {"distance": False, "training": False, "custom": False, "balanced": -1, "restricted": 0, "random": False},
    "generic_random": {"distance": False, "training": False, "custom": False, "balanced": -1, "restricted": -1, "random": True},
    "generic_random_outside": {"distance": False, "training": False, "custom": False, "balanced": -1, "restricted": 1, "random": True},
    "generic_random_inside": {"distance": False, "training": False, "custom": False, "balanced": -1, "restricted": 0, "random": True},

    "standard": {"distance": False, "training": True, "custom": False, "balanced": -1, "restricted": -1, "random": False},
    "standard_same": {"distance": False, "training": True, "custom": False, "balanced": 0, "restricted": -1, "random": False},
    "standard_opposite": {"distance": False, "training": True, "custom": False, "balanced": 1, "restricted": -1, "random": False},
    "standard_balanced": {"distance": False, "training": True, "custom": False, "balanced": 2, "restricted": -1, "random": False},
    "standard_outside": {"distance": False, "training": True, "custom": False, "balanced": -1, "restricted": 1, "random": False},
    "standard_inside": {"distance": False, "training": True, "custom": False, "balanced": -1, "restricted": 0, "random": False},
    "cluster": {"distance": False, "training": False, "custom": False, "balanced": -1, "restricted": -1, "random": False}
}

custom_configs = {
    "custom": {"distance": False, "training": False, "custom": True, "balanced": -1, "restricted": -1, "random": False},
    "custom_random_opposite": {"distance": False, "training": False, "custom": True, "balanced": 1, "restricted": -1,
                               "random": True},
    "custom_random_same": {"distance": False, "training": False, "custom": True, "balanced": 0, "restricted": -1,
                           "random": True},
    "custom_random_balanced": {"distance": False, "training": False, "custom": True, "balanced": 2, "restricted": -1,
                               "random": True},
    "custom_opposite": {"distance": False, "training": False, "custom": True, "balanced": 1, "restricted": -1,
                        "random": False},
    "custom_same": {"distance": False, "training": False, "custom": True, "balanced": 0, "restricted": -1,
                    "random": False},
    "custom_balanced": {"distance": False, "training": False, "custom": True, "balanced": 2, "restricted": -1,
                        "random": False},
    "custom_outside": {"distance": False, "training": False, "custom": True, "balanced": -1, "restricted": 1,
                       "random": False},
    "custom_inside": {"distance": False, "training": False, "custom": True, "balanced": -1, "restricted": 0,
                      "random": False},
    "custom_random": {"distance": False, "training": False, "custom": True, "balanced": -1, "restricted": -1,
                      "random": True},
    "custom_random_outside": {"distance": False, "training": False, "custom": True, "balanced": -1, "restricted": 1,
                              "random": True},
    "custom_random_inside": {"distance": False, "training": False, "custom": True, "balanced": -1, "restricted": 0,
                             "random": True},

}

class Framework:
    def __init__(self, eval_obj):
        self.eval_obj: Evaluation = eval_obj

    def get_nbr(self, sample, json, size, distance="Euc",feature_froze = []):
        e = self.eval_obj
        pred = e.clf.predict([sample[e.features]])[0]
        if json["training"]:
            neighbors = e.traindf[e.features]
        else:
            neighbors = e.context.generate_neighbourhood(feature_froze, sample, self.eval_obj.features,
                                                         1000,
                                                         False, True, False, True)
        filtered_nbrs = neighbors
        filtered_nbrs = filtered_nbrs.drop_duplicates()

        same_class_samples = []
        different_class_samples = []
        if json["restricted"] != -1 or json["balanced"] != -1:
            preds = np.array(self.eval_obj.clf.predict(filtered_nbrs))
            if json["restricted"] != -1:
                if json["restricted"] == 1:
                    filtered_nbrs = filtered_nbrs[preds != pred]
                else:
                    filtered_nbrs = filtered_nbrs[preds == pred]
            elif json["balanced"] != -1:
                same_class_samples = filtered_nbrs[preds == pred]
                different_class_samples = filtered_nbrs[preds != pred]
                total_samples = size
                if json["distance"] == False:
                    if json["balanced"] == 0:
                        num_same = int(total_samples * 0.75)
                        num_diff = total_samples - num_same
                    elif json["balanced"] == 1:
                        num_same = int(total_samples * 0.25)
                        num_diff = total_samples - num_same
                    else:
                        num_same = int(total_samples * 0.5)
                        num_diff = total_samples - num_same
                    num_same = min(num_same, len(same_class_samples))
                    num_diff = min(num_diff, len(different_class_samples))

                    same_class_subset = same_class_samples.sample(n=num_same)
                    different_class_subset = different_class_samples.sample(n=num_diff)

                    filtered_nbrs = pd.concat([same_class_subset, different_class_subset])
                    filtered_nbrs = shuffle(filtered_nbrs)

        # if json["custom"]:
        #     optimized_C = optimize_C(e.traindf[e.features], self.eval_obj.clf.predict(e.traindf[e.features]))
        #     # optimized_C= optimize_C_cvxpy(e.traindf[e.features], self.eval_obj.clf.predict(e.traindf[e.features]))
        #     dists = e.context.calculateMahalanobis(filtered_nbrs[e.context.feats],
        #                                            np.array(sample[e.features]).reshape(1, -1),
        #                                            optimized_C)[:, 0]
        #     filtered_nbrs = filtered_nbrs[dists > 0]
        #     dists = dists[dists > 0]
        #     inds = np.argsort(dists)
        #     # inds = np.argsort(dists[:, 0])
        #     dists = dists[inds]
        #     filtered_nbrs = filtered_nbrs.iloc[inds]
        if json["distance"]:
            if distance == "MB":
                dists = e.context.calculateMahalanobis(filtered_nbrs[e.context.feats],
                                                       np.array(sample[e.features]).reshape(1, -1),
                                                       np.cov(e.traindf[e.features].values))[:, 0]
            else:
                dists = e.context.calculatel2(filtered_nbrs[e.features],
                                              np.array(sample).reshape(1, -1))[:, 0]
            if json["balanced"]!=-1:
                dists_same = e.context.calculateMahalanobis(same_class_samples[e.context.feats],
                                                       np.array(sample[e.features]).reshape(1, -1),
                                                       np.cov(e.traindf[e.features].values))[:, 0]
                dists_opp = e.context.calculateMahalanobis(different_class_samples[e.context.feats],
                                                            np.array(sample[e.features]).reshape(1, -1),
                                                            np.cov(e.traindf[e.features].values))[:, 0]
                filtered_same_nbrs = same_class_samples[dists_same > 0]
                filtered_diff_nbrs = different_class_samples[dists_opp > 0]

                dists_same = dists_same[dists_same > 0]
                dists_opp = dists_opp[dists_opp > 0]
                inds_same = np.argsort(dists_same)
                inds_opp = np.argsort(dists_opp)
                dists_same = dists_same[inds_same]
                dists_opp = dists_opp[inds_opp]
                filtered_same_nbrs = filtered_same_nbrs.iloc[inds_same]
                filtered_diff_nbrs = filtered_diff_nbrs.iloc[inds_opp]
                total_samples = size
                if json["balanced"] == 0:
                    num_same = int(total_samples * 0.75)
                    num_diff = total_samples - num_same
                elif json["balanced"] == 1:
                    num_same = int(total_samples * 0.25)
                    num_diff = total_samples - num_same
                else:
                    num_same = int(total_samples * 0.5)
                    num_diff = total_samples - num_same
                num_same = min(num_same, len(filtered_same_nbrs))
                num_diff = min(num_diff, len(filtered_diff_nbrs))

                same_class_subset = filtered_same_nbrs.iloc[:num_same]
                different_class_subset = filtered_diff_nbrs.iloc[:num_diff]

                filtered_nbrs = pd.concat([same_class_subset, different_class_subset])
                filtered_nbrs = shuffle(filtered_nbrs)
                dists = e.context.calculateMahalanobis(filtered_nbrs,
                                                            np.array(sample[e.features]).reshape(1, -1),
                                                            np.cov(e.traindf[e.features].values))[:, 0]

            else:
                filtered_nbrs = filtered_nbrs[dists > 0]
                dists = dists[dists > 0]
                inds = np.argsort(dists)
                # inds = np.argsort(dists[:, 0])
                dists = dists[inds]
                filtered_nbrs = filtered_nbrs.iloc[inds]
        else:
            dists = e.context.calculatel2(filtered_nbrs[e.features],
                                          np.array(sample).reshape(1, -1))[:, 0]




        filtered_nbrs = filtered_nbrs.iloc[:size]

        weights = 1 / dists[:size]
        weights /= sum(weights)
        return filtered_nbrs, weights


    # def typicality(self,dist):
    #     results = []
    #     densities = []
    #     clusters = []
    #     self.eval_obj.testdf[e.features].to_csv(path+"/data.csv")
    #     for i, instance in self.eval_obj.testdf[e.features].iterrows():
    #         pred = self.eval_obj.clf.predict([instance])[0]
    #         densities.append({"index":i,
    #                           "density":self.eval_obj.density.get_density_score([instance])[0]})
    #         clusters.append({"index":i,
    #                           "cluster":self.eval_obj.density.get_cluster([instance])[0]})
    #         for neighborhood_config in neighborhood_configs:
    #             neighborhood, weights = self.get_nbr(instance, neighborhood_config,distance=dist)
    #             if len(neighborhood) == 0:
    #                 results.append({'sample_index': i, 'neighborhood_config': neighborhood_config,
    #                                 'same_class_proportion': 0})
    #             else:
    #                 neighborhood_preds = self.eval_obj.clf.predict(neighborhood)
    #                 same_class_proportion = round(
    #                     len(neighborhood_preds[neighborhood_preds == pred]) / len(neighborhood), 2)
    #
    #                 results.append({'sample_index': i, 'neighborhood_config': neighborhood_config,
    #                                 'same_class_proportion': same_class_proportion})
    #     clusters = pd.DataFrame(clusters)
    #     clusters.to_csv(path+"/clusters.csv",index=False)
    #     densities = pd.DataFrame(densities)
    #     densities.to_csv(path+"/densities.csv",index=False)
    #     results = pd.DataFrame(results)
    #     results.to_csv(path + '/same_class_proportion_results.csv', index=False)

    # def get_fi_cluster(self, sample, method):
    #     nbr_json = {"no_of_neighbours": 1000, "probability": False, "bound": True,
    #                 "use_range": False, "truly_random": True}
    #     clusters = list(self.eval_obj.density.get_cluster(self.eval_obj.traindf[self.eval_obj.features]))
    #     cluster = self.eval_obj.density.get_cluster([sample])[0]
    #     nbrhood = self.eval_obj.traindf[clusters == cluster]
    #     nbrhood = nbrhood[self.eval_obj.features]
    #     if method == "SHAP":
    #         xai_obj = XAI(self.eval_obj.clf, self.eval_obj.data, self.eval_obj.train_inds, "SVM", None, nbrhood)
    #         return xai_obj.get_shap_vals(sample)
    #
    #     elif method == "SUFF":
    #         sample = sample[e.features]
    #         try:
    #             return self.eval_obj.nece_suff_obj.sufficiency(sample, self.eval_obj.clf.predict([sample]),
    #                                                            self.eval_obj.clf,
    #                                                            self.eval_obj.traindf[self.eval_obj.features], nbr_json
    #                                                            , use_metric="cluster", density=self.eval_obj.density)
    #         except Exception as er:
    #             print(er)
    #             return []
    #
    #     elif method == "LIME_weights":
    #         sample = sample[e.features]
    #         distances = np.linalg.norm(nbrhood.values - sample.values, axis=1)
    #         distances[distances == 0] = 0.01
    #
    #         weights = 1 / distances
    #         weights /= sum(weights)
    #         try:
    #             return self.eval_obj.xai.get_lime(sample, original=True, nbrhood_=None, weights=weights)
    #         except Exception as er:
    #             print(er)
    #             return []
    #     elif method == "CF":
    #         score = []
    #         for f in e.features:
    #             score.append(len(nbrhood[nbrhood[f] != sample[f]]))
    #         scores = [a / sum(score) for a in score]
    #         return scores

    def get_fi(self, sample, nbr_config, method):
        if nbr_config == "cluster":
            return self.get_fi_cluster(sample, method)
        if nbr_config != "Normal":
            json = neighborhood_configs[nbr_config]
            nbr_json = {"no_of_neighbours": 1000, "probability": False, "bound": True,
                        "use_range": False, "truly_random": json["random"]}

        if method == "SHAP":
            if nbr_config == "Normal":
                return self.eval_obj.xai.get_shap_vals(sample)
            else:
                if json["distance"] == False:
                    dist = "Euc"
                else: dist="MB"
                nbrhood, weights = self.get_nbr(sample, nbr_config,distance=dist)
            if len(nbrhood) == 0:
                return []
            xai_obj = XAI(self.eval_obj.clf, self.eval_obj.data, self.eval_obj.train_inds, "SVM", None, nbrhood)
            return xai_obj.get_shap_vals(sample)
        elif method == "SUFF" and nbr_config!="Normal":

            sample = sample[e.features]
            try:
                return self.eval_obj.nece_suff_obj.sufficiency(sample, self.eval_obj.clf.predict([sample]),
                                                               self.eval_obj.clf,
                                                               self.eval_obj.traindf[self.eval_obj.features],
                                                               nbr_json, use_metric="MB")
            except Exception as er:
                print(er)
                return []
        elif method == "NECE" and nbr_config != "Normal":
            sample = sample[e.features]
            try:
                return self.eval_obj.nece_suff_obj.necessity(sample, self.eval_obj.clf.predict([sample]),
                                                             self.eval_obj.clf,
                                                             self.eval_obj.traindf[self.eval_obj.features],
                                                             nbr_json, use_metric="MB")
            except Exception as er:
                print(er)
                return []
        elif method == "LIME_weights":
            if nbr_config == "Normal":
                distances = np.linalg.norm(e.traindf[e.features].values - sample.values, axis=1)
                distances[distances==0] = 0.01

                weights = 1 / distances
                weights /= sum(weights)
                try:
                 return self.eval_obj.xai.get_lime(sample, original=True, nbrhood_=None, weights=weights)
                except:
                    return []
            else:
                if json["distance"] == False:
                    dist = "Euc"
                else:
                    dist = "MB"
                nbrhood, weights = self.get_nbr(sample, nbr_config, distance=dist)
            if len(nbrhood) == 0:
                return []
            xai_obj = XAI(self.eval_obj.clf, self.eval_obj.data, self.eval_obj.train_inds, "SVM", None, nbrhood)
            try:
                return xai_obj.get_lime(sample, False, nbrhood, weights)
            except Exception as er:
                print(er)
                return []
        elif method == "CF":
            if nbr_config == "Normal":
                nbrhood, weights = self.get_nbr(sample, "dist_outside", distance="Euc")
            else:
                if json["distance"] == False:
                    dist = "Euc"
                else:
                    dist = "MB"
                nbrhood, weights = self.get_nbr(sample, nbr_config,distance=dist)
                # nbrhood = nbrhood.iloc[:10]
            if len(nbrhood) == 0:
                return []
            score = []
            for f in e.features:
                score.append(len(nbrhood[nbrhood[f] != sample[f]]))
            scores = [a / sum(score) for a in score]
            return scores
        else: return []





def reorder_features(fi_scores, feature_list,subset_length):
    sorted_indices = sorted(range(len(fi_scores)), key=lambda k: fi_scores[k], reverse=True)
    reordered_feature_list = [feature_list[i] for i in sorted_indices]
    grouped_features = [reordered_feature_list[i:i+subset_length] for i in range(0, len(reordered_feature_list) - subset_length + 1)]
    return grouped_features

configs_dict = {
    "SHAP":['dist', 'dist_random_opposite', 'dist_random_same', 'dist_random_balanced', 'dist_opposite', 'dist_same',
 'dist_balanced', 'dist_outside', 'dist_inside', 'dist_random', 'dist_random_outside', 'dist_random_inside','standard'],
 "LIME_weights":['dist', 'dist_random_opposite', 'dist_random_same', 'dist_random_balanced', 'dist_opposite', 'dist_same',
 'dist_balanced', 'dist_random', 'standard'],
  "CF": ['dist_outside', 'dist_random_outside', 'standard_outside']
}
# configs_dict = {
#     "SHAP":['dist_outside'],
#  "LIME_weights":['dist_balanced', 'dist_random', 'standard'],
#   "CF": ['dist_outside',"standard_outside"]
# }

 #,
# datasets = [ "cerv","diab_without_insulin","heart_db"]
# fi_scores_original = []
# folders = {"diab_without_insulin":"diab",
#            "heart_db":"heart",
#            "cerv":"cerv"}
# a = pd.DataFrame()
# for dataname in datasets:
#     input_data = {"data": dataname, "classifier": "SVM", "fold": "fold1"}
#     path = "res/" + folders[dataname]
#     e = Evaluation(input_data,"svm")
#
#     feat_inds = list(range(len(e.features)))
#     train_data = e.traindf
#
#     samples = e.testdf[e.features].iloc[:90]
#     f = Framework(e)
#
#     score_collection=[]
#     additional_data=[]
#     set_len = 4
#     for config in tqdm(neighborhood_configs.keys()):
#         for method in tqdm(["SHAP","LIME","CF"]):
#             if config not in configs_dict[method]: continue
#             for idx, sample in samples.iterrows():
#                 fi_scores = f.get_fi(sample, config, method)
#                 suff_scores = f.get_fi(sample, "dist_random" , "SUFF")
#                 nece_scores = f.get_fi(sample, "dist_random", "NECE")
#                 score_collection.append({
#                     "data":folders[dataname],
#                     "method":method,
#                     "scores":fi_scores,
#                     "index":idx,
#                     "config":config
#                 })
#                 score_collection.append({
#                     "data": folders[dataname],
#                     "method": "SUFF",
#                     "scores": suff_scores,
#                     "index": idx,
#                     "config": config
#                 })
#                 score_collection.append({
#                     "data": folders[dataname],
#                     "method": "NECE",
#                     "scores": nece_scores,
#                     "index": idx,
#                     "config": config
#                 })
#                 if len(fi_scores)!=0 and len(suff_scores)!=0:
#                     fi_scores = np.array([abs(score) for score in fi_scores])
#                     additional_data.append({
#                         "index":idx,
#                         "config":config,
#                         "methods":method+"-"+"SUFF",
#                         "corr":measure_kendall_correlation(fi_scores,suff_scores)
#                     })
#                 if len(fi_scores)!=0 and len(nece_scores)!=0:
#                     fi_scores = np.array([abs(score) for score in fi_scores])
#                     additional_data.append({
#                         "index":idx,
#                         "config":config,
#                         "methods":method+"-"+"NECE",
#                         "corr":measure_kendall_correlation(fi_scores,nece_scores)
#                     })
#
#     score_collection = pd.DataFrame(score_collection)
#     additional_data = pd.DataFrame(additional_data)
#     score_collection.to_csv(path+"/fi_scores_cf.csv")
#     additional_data.to_csv(path + "/corrs_.csv")
#     #             fi_scores = np.array([abs(score) for score in fi_scores])
#     #             feature_sets = reorder_features(fi_scores,e.features,set_len)
#     #             nbr_json = {"no_of_neighbours": 1000, "probability": False, "bound": True,
#     #                         "use_range": False, "truly_random": True}
#     #             decreasing_values = list(range(len(feature_sets), 0, -1))
#     #             # suff_scores = e.nece_suff_obj.sufficiency_sets(sample, e.clf.predict([sample]),e.clf,
#     #             #                                                e.traindf[e.features], nbr_json,feature_sets)
#     #             nece_scores = e.nece_suff_obj.necessity_sets(sample, e.clf.predict([sample]), e.clf,
#     #                                                            e.traindf[e.features], nbr_json, feature_sets)
#     #             additional_data.append({
#     #                                     "index":idx,
#     #                                     "config":config,
#     #                                     "methods":method+"-"+"NECE",
#     #                                     "corr":measure_kendall_correlation(decreasing_values,nece_scores)
#     #                                 })
#     # additional_data = pd.DataFrame(additional_data)
#     # additional_data.to_csv(path + f"/nece_corrs_seq_set_shap{set_len}.csv")
#
#     cors_df = additional_data
#     cors_df["data"] = folders[dataname]
#     a = a._append(cors_df,ignore_index=True)
# a.to_csv("folder/corrs.csv")
