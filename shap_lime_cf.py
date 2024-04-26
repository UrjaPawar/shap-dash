from med_dataset import Data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import lime
import lime.lime_tabular
import dice_ml
from density_cluster import Density


class XAI:
    def __init__(self, clf, data: Data, train_inds, model_name, density, custom_nbrhood = []):
        self.model = clf
        self.data = data
        self.train_df = data.df[data.features+[data.target]].iloc[train_inds]
        if len(custom_nbrhood) > 0:
            self.custom_neighborhood = custom_nbrhood
        else:
            self.custom_neighborhood = self.train_df
        self.feats = data.features
        self.model_name = model_name
        self.density = density
        self.shap_explainer = self.get_shap_explainer()
        # if self.data.target in custom_nbrhood:
        #     self.dice_explainer = self.get_dice(custom_nbrhood[self.data.target])
        # else:



    def get_shap_explainer(self):
        if self.model_name == "Log-Reg":
            # return shap.Explainer(self.model, self.train_df[self.feats].iloc[15:25])
            return shap.Explainer(self.model, self.custom_neighborhood[self.data.features])
        elif self.model_name == "rf_clf":
            return shap.TreeExplainer(self.model)
        if self.model_name == "MLP" or self.model_name == "SVM":
            # masker = shap.maskers.Independent(data = self.train_df[self.feats])
            if len(self.custom_neighborhood) > 100:
                masker = shap.maskers.Independent(data=self.custom_neighborhood[self.data.features].iloc[:100])
            else:
                masker = shap.maskers.Independent(data=self.custom_neighborhood[self.data.features])
            # gradient and deep explainers might be requiring Image type inputs
            # return shap.KernelExplainer(self.model.predict, data=self.train_df[self.feats], masker=masker)
            if len(self.custom_neighborhood) > 100:
                return shap.KernelExplainer(self.model.predict, data=self.custom_neighborhood[self.data.features].iloc[:100],
                                        masker=masker)
            else:
                return shap.KernelExplainer(self.model.predict, data=self.custom_neighborhood[self.data.features],
                                            masker=masker)

    def get_shap_vals(self, sample):
        if self.model_name == "Log-Reg":
            shap_vals = self.shap_explainer(np.array(sample).reshape(1,-1))[0]
            return shap_vals.values
        elif self.model_name == "rf_clf":
            shap_vals = self.shap_explainer(np.array(sample).reshape(1, -1))
            return shap_vals.values[:,:,1] # for class 1
        elif self.model_name == "MLP" or self.model_name == "SVM":
            shap_vals = self.shap_explainer.shap_values(np.array(sample[self.data.features]).reshape(1, -1))
            # return shap_vals.values[:, :][0] # for MLP - 2D output as
            return shap_vals[0]
            # both classes show different shap values, we need both

    def get_lime(self, sample_for_lime, original, nbrhood_=None, weights=[]):
        if original:
            clusters = list(self.density.get_cluster(self.train_df[self.data.features]))
            cluster = self.density.get_cluster([sample_for_lime])[0]
            nbrhood = self.train_df[clusters == cluster]
            nbrhood = nbrhood[self.data.features]
            if len(weights) != 0:
                weights = weights[clusters == cluster]

        else:
            nbrhood = nbrhood_
        lime_log = LogisticRegression()
        if len(weights)==0:
            weights = None
        lime_log.fit(nbrhood[self.data.features], self.model.predict(nbrhood[self.data.features]),sample_weight=weights)
        return lime_log.coef_[0]

    def get_dice(self, targets):
        df = self.custom_neighborhood[self.data.features]
        df[self.data.target] = targets
        d_ = dice_ml.Data(dataframe=df, continuous_features=self.data.continuous,
                          outcome_name=self.data.target)
        m_ = dice_ml.Model(model=self.model, backend="sklearn")
        return dice_ml.Dice(d_, m_, method="random")

    def get_cf_fi(self, sample, dice_obj, no_of_cf = 100):
            try:
                log_fi = dice_obj.local_feature_importance(sample, total_CFs=no_of_cf)
                fi = log_fi.local_importance[0]
                res = []
                for feat in self.data.features:
                    res.append(fi[feat])
                return res
            except ValueError:
                return None
