
.. image:: https://img.shields.io/github/license/deepset-ai/farm
:target: https://github.com/deepset-ai/FARM/blob/master/LICENSE
:alt: License

Offensive Language Identification using a German BERT model
#############

This repository contains code that reproduces our submission to the Shared Task on the Identification of Offensive Language in context of the GermEval workshop at the Conference on Natural Language Processing (KONVENS) 2019.
<https://projects.fzai.h-da.de/iggsa/>

Installation
#############

Recommended::

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
author = "Risch, Julian and Stoll, Anke and Ziegele, Marc and Krestel, Ralf",
booktitle = "Proceedings of GermEval (co-located with KONVENS)",
title = "Offensive Language Identification using a German BERT model",
year = "2019"
}
```


Acknowledgements
############
Thanks to deepset.ai for providing the underlying framework FARM: (**F**\ ramework for **A**\ dapting **R**\ epresentation **M**\ odels)
See the `full documentation <https://farm.deepset.ai>`_ for more details about FARM
