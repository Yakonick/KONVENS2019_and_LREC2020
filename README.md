[![License](https://img.shields.io/github/license/deepset-ai/farm)](https://github.com/deepset-ai/FARM/blob/master/LICENSE)

## Offensive Language Identification using a German BERT model

This repository contains code that reproduces our (Julian Risch, Anke Stoll, Marc Ziegele, Ralf Krestel) submission to the [Shared Task on the Identification of Offensive Language](https://projects.fzai.h-da.de/iggsa/) in context of the GermEval workshop at the Conference on Natural Language Processing (KONVENS) 2019. Please find more details in our paper [**Offensive Language Identification using a German BERT model**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2019offensive.pdf). 

The paper won the Shared Task on the Identification of Implicit and Explicit Offensive Language at GermEval2019. The full results of the task can be found [here](https://projects.fzai.h-da.de/iggsa/wp-content/uploads/2019/10/Auswertung_Abgaben_GermEval_2019_Subtask_3.xlsx). The training and test data can be found [here](https://projects.fzai.h-da.de/iggsa/data-2019/).

| Group ID | F1-Score |
| --- | ---: |
| hpiDEDIS (this paper) | 73.1 |
| upb | 70.8 |
| FoSIL | 69.6 |
| inriaFBK | 69.5 |
| HAU | 69.3 |
| rgcl | 68.9 |
| fkie | 58.4 |

## Installation

Recommended:

    git clone https://github.com/julian-risch/KONVENS2019_and_LREC2020/FARM.git
    git checkout germeval2019
    cd FARM
    pip install -r requirements.txt
    pip install --editable .


## Citation

If you use our work, please cite our paper [**Offensive Language Identification using a German BERT model**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2019offensive.pdf) as follows:

    @inproceedings{risch2019hpidedis,
    author = "Risch, Julian and Stoll, Anke and Ziegele, Marc and Krestel, Ralf",
    title = "hpiDEDIS at GermEval 2019: Offensive Language Identification using a German BERT model",
    booktitle = "Proceedings of the 15th Conference on Natural Language Processing (KONVENS)",
    pages = "403--408",
    address = "Erlangen, Germany",
    publisher = "German Society for Computational Linguistics \& Language Technology",
    year = "2019"
    }


## Acknowledgements

Thanks to deepset.ai for providing the underlying framework FARM: (**F**ramework for **A**dapting **R**epresentation **M**odels)
See the [full documentation](https://farm.deepset.ai) for more details about FARM.
