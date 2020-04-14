# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from farm.experiment import run_experiment, load_experiments
import random 
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from collections import defaultdict

random.seed(a=None, version=2)
language = "eng" # eng iben hin
option = "" # "_sep_emoji" or ""
submission = "_dev" # make predictions for "_dev" or "_test" dataset
def main():
    config_files = [
	Path("experiments/text_classification/trac2020_eng_config.json")
    ]

    train = True
    make_submission = False
    max_ensemble_size = 3
    all_seeds = range(1, max_ensemble_size+1)
    runs = 2

    if train:
        for conf_file in config_files:
            experiments = load_experiments(conf_file)
            for experiment in experiments:
                for seed in all_seeds:
                    experiment.general.seed = seed
                    run_experiment(experiment)

    if not make_submission:
        df_dev = pd.read_csv("data/trac2020/trac2_"+language+"_dev"+option+".csv")
        label_to_index = {'NAG':'0', 'CAG':'1', 'OAG':'2'}
        target_names = ['NAG', 'CAG', 'OAG']
        y_true = df_dev.filter(['Sub-task A']).applymap(lambda x: label_to_index[x]).values

    averaged_f1_scores = [0 for i in range(1,len(all_seeds)+2)]
    all_f1_scores_weighted = defaultdict(int)
    all_f1_scores_macro = defaultdict(int)
    for run in range(1,runs+1):
        for number_of_samples in range(1,len(all_seeds)+1):
            sampled_seeds = random.sample(all_seeds, number_of_samples)
            dfs = []
            for seed in sampled_seeds:
                 #df_probabilities = pd.read_csv("/home/jurisc/FARM-Feb-2020/FARM/data/trac2020/trac2_"+language+submission+"_prediction-bert-base-multilingual-uncased-multilingual-trac2020-taskA-"+str(seed)+".csv")
                 df_probabilities = pd.read_csv("data/trac2020/trac2_"+language+submission+"_prediction-bert-base-uncased-english-trac2020-taskA-"+str(seed)+".csv")
                 df_probabilities['group'] = np.arange(len(df_probabilities))
                 dfs = dfs + [df_probabilities]

            final = pd.concat(dfs, ignore_index=True)
            final = final.groupby('group').mean()
            final = final.reset_index(drop=True)

            if not make_submission:
                y_pred = final.idxmax(axis = 1, skipna = True)
                averaged_f1_scores[number_of_samples] = averaged_f1_scores[number_of_samples] + f1_score(y_true, y_pred.values, average='weighted')
                all_f1_scores_weighted[(number_of_samples,run)] = f1_score(y_true, y_pred.values, average='weighted')

            if make_submission:
                final = final.idxmax(axis = 1, skipna = True).to_frame()
                label_to_index = {'0':'NAG', '1':'CAG', '2':'OAG'}
                #label_to_index = {'0':'NGEN', '1':'GEN'}
                final = final.applymap(lambda x: label_to_index[x])
                # merge with ids
                df_test = pd.read_csv("data/trac2020/trac2_"+language+"_test"+option+".csv")
                df_test = df_test.filter(['ID'])
                final = pd.concat([df_test, final], axis=1)
                final.columns = ['ID','Label']
                final.to_csv("data/trac2020/submission_"+language+"_"+str(number_of_samples)+".csv", index= False)

    for number_of_samples in range(1,len(all_seeds)+1):
        print("Ensemble size: "+str(number_of_samples), end = " ")
        for run in range(1,runs+1):
            print("Run: "+str(run), end = " ")
            print("Weighted macro-average F1-score: ", end = " ")
            print(all_f1_scores_weighted[(number_of_samples,run)], end = ',')
        print("")
if __name__ == "__main__":
    main()
