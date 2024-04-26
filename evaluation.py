import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from med_dataset import Data
from neighborhood import Neighborhood
from suff_nece import Nece_Suff
from shap_lime_cf import XAI
from density_cluster import Density
from sklearn.metrics import recall_score
from tqdm import tqdm
from plotting import plot_histograms
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import statistics
import warnings

warnings.filterwarnings("ignore")


def measure_kendall_correlation(ranking1, ranking2):
    kendal = stats.kendalltau(ranking1, ranking2)
    return kendal.correlation


class Evaluation:
    def __init__(self, input_data):
        self.input_data = input_data
        self.data = self.get_data(False)
        self.inds_path = "analysis_outputs/" + input_data["data"] + "/" + input_data["fold"]
        self.test_inds = joblib.load(self.inds_path + "/test")
        self.train_inds = joblib.load(self.inds_path + "/train")
        if input_data["data"] == "cerv":
            self.train_inds = self.train_inds[self.train_inds <= 1252]
            self.test_inds = self.test_inds[self.test_inds <= 1252]

        self.clf = self.get_clf()
        print("recall: ", self.get_recall())
        self.context = Neighborhood(self.data)
        self.traindf = self.data.df.iloc[self.train_inds]
        self.testdf = self.data.df.iloc[self.test_inds]
        self.density = Density(self.data, self.train_inds)
        self.nece_suff_obj = Nece_Suff(self.context)
        self.features = np.array(self.data.features)
        self.densities = self.density.get_density_score(self.testdf[self.features])
        self.top_10_high_density = np.argsort(self.densities)[-10:]
        self.top_10_low_density = np.argsort(self.densities)[:10]

        self.xai = XAI(self.clf, self.data, self.train_inds, input_data["classifier"], self.density)
        self.means = {}
        for feat in self.features:
            if feat in self.data.continuous:
                self.means[feat] = round(np.mean(self.traindf[feat]), self.data.dec_precisions[feat])
            else:
                self.means[feat] = statistics.mode(self.traindf[feat])
        # self.expln_use_range = input_data["explanandum_context"] == "medical"
        # neighborhood_json_none = {"no_of_neighbours": 500, "probability": False, "bound": True, "use_range": False,
        #                           "truly_random": True}
        # neighborhood_json_prob = {"no_of_neighbours": 500, "probability": True, "bound": True, "use_range": False,
        #                           "truly_random": True}
        #
        # self.nbr_dict = {
        #     "none": neighborhood_json_none,
        #     "prob": neighborhood_json_prob,
        # }
        # self.nbr_json = self.nbr_dict[input_data["nbr_json"]]
        # self.nbr_json = self.input_data["nbr_json"]

    def get_data(self, hot_encode):
        the_data = None
        if self.input_data["data"] == "heart_db":
            the_data = Data("Heart DB", hot_encode)
        elif self.input_data["data"] == "cerv":
            the_data = Data("Cervical DB", hot_encode)
        elif self.input_data["data"] == "diab_insulin" or self.input_data["data"] == "diab_without_insulin":
            the_data = Data("Diabetes DB", hot_encode, pick_without_insulin=True)
        return the_data

    def get_clf(self):
        if self.input_data["classifier"] == "MLP":
            clf = MLPClassifier(random_state=1, max_iter=300)
            clf.fit(self.data.df[self.data.features].iloc[self.train_inds],
                    self.data.df[self.data.target].iloc[self.train_inds])
        elif self.input_data["classifier"] == "SVM":
            clf = svm.SVC(kernel='linear', probability=True)
            clf.fit(self.data.df[self.data.features].iloc[self.train_inds],
                    self.data.df[self.data.target].iloc[self.train_inds])
        else:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(self.data.df[self.data.features].iloc[self.train_inds],
                    self.data.df[self.data.target].iloc[self.train_inds])
        return clf

    def get_recall(self):
        return recall_score(self.clf.predict(self.data.df[self.data.features].iloc[self.test_inds][self.data.features]),
                            self.data.df.iloc[self.test_inds][self.data.target])

    def ex3(self, sample_for_ex3, n, k, scores, output, clf):
        total = 0
        points = 0
        for i in range(200):
            features_to_change_from = np.random.choice(len(self.features), n, replace=False)
            temp = sample_for_ex3.copy()
            for feat in self.features[features_to_change_from]:
                temp[feat] = self.means[feat]
            if clf.predict([temp])[0] != output:
                scores = np.array(scores)
                inds_sorted = np.argsort(scores[features_to_change_from])
                features_revert = self.features[features_to_change_from[inds_sorted[-k:]]]
                for feature_revert in features_revert:
                    temp[feature_revert] = sample_for_ex3[feature_revert]
                if clf.predict([temp])[0] == output:
                    points += 1
                total += 1
        if total != 0:
            return points / total
        else:
            return 0

    def ex2(self, sample_for_ex2, k, scores, output, clf):
        inds_sorted = np.argsort(scores)
        nbr_ex1 = self.context.generate_neighbourhood(self.features[inds_sorted[-k:]], sample_for_ex2,
                                                      self.data.features, 200, False, True, False, True)
        outputs = clf.predict(nbr_ex1)
        return len(outputs[outputs == output]) / len(outputs)

    def ex1(self, sample_for_ex1, k, scores, output, clf):
        inds_sorted = np.argsort(scores)
        inds_to_fix = inds_sorted[:-k]
        nbr_ex1 = self.context.generate_neighbourhood(self.features[inds_to_fix], sample_for_ex1, self.data.features,
                                                      200, False, True, self.expln_use_range, True)
        outputs = clf.predict(nbr_ex1)
        return len(outputs[outputs != output]) / len(outputs)



# for diabetes we have only used shorten version of dataset, but for 4,5 we  used full
# random 4th diab is fully run
datasets = ["diab_without_insulin","cerv","heart_db"]
ticks = ["SHAP", "K-LIME", "DICE-CF", "SUFF", "NECE"]
clfs = [ "MLP", "Log-Reg","SVM"]
# expl_contexts = ["medical", "random"]
# nbr_jsons = ["none", "prob"]
expl_context = "medical"
nbr_json = "none"

# for data_name in datasets:
#     for model_name in clfs:
#         input_data = {"data": data_name, "classifier": model_name, "fold": "fold1", "explanandum_context": expl_context,
#                       "nbr_json": nbr_json}
#         print(input_data)
#         eval = Evaluation(input_data)
#         print(eval.get_recall())
#
# for data_name in datasets:
#     for model_name in clfs:
#         input_data = {"data": data_name, "classifier": model_name, "fold": "fold1", "explanandum_context": expl_context,
#                       "nbr_json": nbr_json}
#         print(input_data)
#         eval = Evaluation(input_data)
#         if data_name == "diab_without_insulin":
#             v = [4,5]
#         else:
#             v = [1,2,3,4,5]
#         for top_k in tqdm(v):
#             ex_score_list = []
#             suff_nece_corr = []
#             dump_path = expl_context + "_eval/" + nbr_json + "/" +\
#                         input_data['data'] + "/" + input_data["classifier"] + "/"
#             for ind in tqdm(range(len(eval.testdf.iloc[:100]))):
#                 original_sample = eval.testdf[eval.features].iloc[ind].copy()
#                 output = eval.clf.predict([original_sample])[0]
#                 lime_old = eval.xai.get_lime(original_sample, True, None)
#                 shap_old = eval.xai.get_shap_vals(original_sample)
#                 dice_fi = eval.xai.get_cf_fi(eval.testdf[eval.features].iloc[ind:ind+1], eval.xai.dice_explainer)
#                 suff_mb_false = eval.nece_suff_obj.sufficiency(original_sample, eval.clf.predict([original_sample]),
#                                                                eval.clf, eval.traindf[eval.features],
#                                                                eval.nbr_dict[nbr_json], use_metric="MB")
#                 nece_mb_false = eval.nece_suff_obj.necessity(original_sample, eval.clf.predict([original_sample]),
#                                                                eval.clf, eval.traindf[eval.features],
#                                                                eval.nbr_dict[nbr_json], use_metric="MB")
#                 suff_nece_corr.append(measure_kendall_correlation(suff_mb_false, nece_mb_false))
#                 ex1_scores = []
#                 ex2_scores = []
#                 ex3_scores = []
#
#                 for score in [shap_old, lime_old, dice_fi, suff_mb_false, nece_mb_false]:
#                     ex1_scores.append(eval.ex1(original_sample, top_k, score, output, eval.clf))
#                     ex2_scores.append(eval.ex2(original_sample, top_k, score, output, eval.clf))
#                     ex3 = []
#                     for no_of_features in [1,2,3]:
#                         ex3.append(eval.ex3(original_sample, top_k, no_of_features, score, output, eval.clf))
#                     ex3_scores.append(ex3)
#                 ex_score_list.append([ex1_scores, ex2_scores, ex3_scores])
#             joblib.dump(suff_nece_corr, dump_path + "suff_nece_corr_" + str(top_k))
#             joblib.dump(ex_score_list, dump_path + "ex_score_list_" + str(top_k))
#

'''
Code written above was used in main explananda evaluation

'''
# TODO check with use range =True explanandums
# TODO Finalise neighborhood for each thing - nece suff neigbhborhood none with mb,
# TODO density wise analysis of correlation and explananda


# less_correlated = [44, 6, 32, 35]
# high_correlated = [4, 13, 56, 40]
#
# print("oj")
# ex1s_none = []
# ex2s_none = []
# ex3s_none = []
#     for top_k in tqdm([1, 2, 3, 4, 5, 6]):
#         ex1 = []
#         ex2 = []
#         ex3 = []
#         for clf_name in ["MLP", "SVM", "log_clf"]:
#             dump_path = input_data['data'] + "/" + nbr_json + "/" + clf_name + "/"
#             suff_nece_corrs = joblib.load(dump_path + "suff_nece_corr_" + str(top_k))
#             ex_scores = joblib.load(dump_path + "ex_score_list_" + str(top_k))
#             exs = np.array(ex_scores)
#             ex1.append(np.mean(exs[:, 0], axis=0))
#             ex2.append(np.mean(exs[:, 1], axis=0))
#             # ex3.append(np.mean(exs[:, 2], axis=0))
#         if nbr_json == "none":
#             ex1s_none.append(ex1)
#             ex2s_none.append(ex2)
#             ex3s_none.append(ex3)
#         else:
#             ex1s_prob.append(ex1)
#             ex2s_prob.append(ex2)
#             ex3s_prob.append(ex3)

# for top_k in [1, 2, 3, 4, 5, 6]:
#     dump_path = input_data['data'] + "/" + "prob" + "/"
#     df_1 = pd.DataFrame(data=ex2s_prob[top_k - 1], columns=ticks)
#     df_1.index = clfs
#     ax = df_1.plot.bar()
#     ax.set_title("Explanandum 2, with top " + str(top_k) + " features")
#     plt.savefig(dump_path + "ex2_top_" + str(top_k) + ".png")
#     plt.clf()
# # df_2 = pd.DataFrame(data=ex1s_none[1], columns=ticks)
# # df_3 = pd.DataFrame(data=ex1s_none[2], columns=ticks)
# # df_4 = pd.DataFrame(data=ex1s_none[3], columns=ticks)
# print("ok")
