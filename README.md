[![License](https://img.shields.io/github/license/deepset-ai/farm)](https://github.com/deepset-ai/FARM/blob/master/LICENSE)

## Bagging BERT Models for Robust Aggression Identification

This repository contains code that reproduces our submission to the [Shared Tasks on Aggression Identification](https://sites.google.com/view/trac2/shared-task) in context of the TRAC workshop at LREC 2020. Please find more details in our paper [**Bagging BERT Models for Robust Aggression Identification**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2020bagging.pdf). A video that focuses on the ensembling technique is available on [YouTube](https://www.youtube.com/watch?v=W3056xbZmus).

The paper won five out of six subtasks. The following table lists weighted macro-averaged F1-scores for our submission and the best other submission for the three languages English, Hindi, and Bangla and the two tasks A (Aggression Identification), and task B (Misogynistic Aggression Identification).

|                       |Task A English|Task B English|Task A Hindi|Task B Hindi|Task A Bangla|Task B Bangla|
| ---                   | ---: | ---: | ---: | ---: | ---: | ---: |
| Our Submission        |**80.29** |85.14 |**81.28** |**87.81** |**82.19** |**93.85** |
| Best Other Submission |75.92 |**87.16** |79.44 |86.89 |80.83 |92.97 |

## Installation

Recommended:

    git clone https://github.com/julian-risch/KONVENS2019_and_LREC2020/FARM.git
    git checkout trac2020
    cd FARM
    pip install -r requirements.txt
    pip install --editable .

If you download the task dataset to ```data/trac2020/trac2_eng_train.csv``` and ```data/trac2020/trac2_eng_dev.csv``` and run the command ```python run_all_experiments.py``` (5 to 10min runtime + time for downloading the pre-trained BERT base model if not in cache) the output should look similar to the following:

    Ensemble size: 1 Run: 1 Weighted macro-average F1-score:  0.7222017651852523,Run: 2 Weighted macro-average F1-score:  0.7084656293593892,
    Ensemble size: 2 Run: 1 Weighted macro-average F1-score:  0.7371756051970508,Run: 2 Weighted macro-average F1-score:  0.7371756051970508,
    Ensemble size: 3 Run: 1 Weighted macro-average F1-score:  0.7935370278126409,Run: 2 Weighted macro-average F1-score:  0.7935370278126409,

## Citation

If you use our work, please cite our paper [**Bagging BERT Models for Robust Aggression Identification**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2020bagging.pdf) as follows:

    @inproceedings{risch2020bagging,
    author = {Risch, Julian and Krestel, Ralf},
    booktitle = {Proceedings of the Workshop on Trolling, Aggression and Cyberbullying (TRAC@LREC)},
    title = {Bagging BERT Models for Robust Aggression Identification},
    year = 2020
    }



## Acknowledgements

Thanks to deepset.ai for providing the underlying framework FARM: (**F**ramework for **A**dapting **R**epresentation **M**odels)
See the [full documentation](https://farm.deepset.ai) for more details about FARM.
