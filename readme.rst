(**F**\ ramework for **A**\ dapting **R**\ epresentation **M**\ odels)

.. image:: https://img.shields.io/github/license/deepset-ai/farm
:target: https://github.com/deepset-ai/FARM/blob/master/LICENSE
:alt: License

Offensive Language Identification using a German BERT model
#############

This repository contains code that reproduces our submission to the Shared Task on the Identification of Offensive Language in context of the GermEval workshop at the Conference on Natural Language Processing (KONVENS) 2019.
<https://projects.fzai.h-da.de/iggsa/>

Installation
#############

Recommended (because of active development)::

    git clone https://github.com/cgsee1/FARM.git
    cd FARM
    pip install -r requirements.txt
    pip install --editable .


Citation
#############
If you use our work, please cite our `paper <https://github.com/cgsee1/FARM/edit/germeval2019/risch2019hpidedis.pdf>`_
**Offensive Language Identification using a German BERT model** as follows:

```
@inproceedings{risch2019hpidedis,
abstract = "Pre-training language representations on large text corpora, for example, with BERT, has recently shown to achieve impressive performance at a variety of downstream NLP tasks. So far, applying BERT to offensive language identification for German- language texts failed due to the lack of pre-trained, German-language models. In this paper, we fine-tune a BERT model that was pre-trained on 12 GB of German texts to the task of offensive language identification. This model significantly outperforms our baselines and achieves a macro F1 score of 76\% on coarse-grained, 51\% on fine-grained, and 73\% on implicit/explicit classification. We analyze the strengths and weaknesses of the model and derive promising directions for future work.",
author = "Risch, Julian and Stoll, Anke and Ziegele, Marc and Krestel, Ralf",
booktitle = "Proceedings of GermEval (co-located with KONVENS)",
title = "Offensive Language Identification using a German BERT model",
year = 2019
}
```

Acknowledgements
############
Thanks to deepset.ai for providing the underlying framework FARM.
See the `full documentation <https://farm.deepset.ai>`_ for more details about FARM
