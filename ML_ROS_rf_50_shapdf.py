import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

loaded_model = joblib.load('run14_rf_50_r2/run_output_models.pkl')

og_names = []
with open('run14_rf_50_r2/run_output_imp', 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            og_name = parts[0]
            og_names.append(og_name)

df = pd.read_csv('df_1_2mMextremes_20percent_noAOX.tsv', sep='\t')
df = df.set_index('Species')

og_names_reordered = [col for col in df.columns if col in og_names]

X = df[og_names_reordered + [col for col in df.columns if col not in og_names_reordered]]
X = X.drop("Class", axis=1)

explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer.shap_values(X)

shap_values_for_class = shap_values[1]

shap_values_for_class_50_feat = shap_values_for_class[:,:50]
shap_df = pd.DataFrame(shap_values_for_class_50_feat, columns= X.columns[:50], index = X.index)
shap_df = shap_df[og_names]
shap_df.to_csv('TESTING_RERUN_shap_vals_rf50_r2.tsv', sep="\t")
