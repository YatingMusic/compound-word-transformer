# Compound Word Transformer


Authors: [Wen-Yi Hsiao](), [Jen-Yu Liu](), [Yin-Cheng Yeh]() and [Yi-Hsuan Yang]()

[**Paper (arXiv)**]() | [**Audio demo (Google Drive)**]() |

Official PyTorch implementation of AAAI2021 paper "Compound Word Transformer: Learning to Compose Full-Song Musicover Dynamic Directed Hypergraphs".

We presented a new variant of the Transformer that can processes multiple consecutive tokens at once at a time step. The proposed method can greatly reduce the length of the resulting sequence and therefore enhance the training and inference efficiency. We employ it to learn to compose expressive Pop piano music of full-song length (involving up to 10K individual to23 kens per song). In this repository, we open source our **Ailabs.tw Pop17K** dataset, and the codes for unconditional generation.


## Dependencies

* python 3.6
* Required packages:
```bash
pip install miditoolkit
pip install torch==1.5.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user pytorch-fast-transformers
pip install chorder
```

``chorder`` is our in-house rule-based symbolic chord recognition algorithm, which is developed by our former intern - [joshuachang2311](https://github.com/joshuachang2311/chorder). He is also a jazz pianist :musical_keyboard:. 


## Model
In this work, we conduct two scenario of generation:
* unconditional generation
    * To see the experimental results and the discussion, pleasee refer to [here](https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/Experiments.md). 

* conditional generation, leadsheet to full midi (ls2midi)
    * [**Working in progress**] The codes associated with this part are planning to open source in the future
        * melody extracyion (skyline) 
        * objective metrics
        * model

## Dataset
To preparing your own training data, please refer to [documentaion]() for further understanding.  
The full workspace of our dataset **Ailabs.tw Pop17K** are available [here](https://drive.google.com/drive/folders/1DY54sxeCcQfVXdGXps5lHwtRe7D_kBRV?usp=sharing).


## Acknowledgement
- PyTorch codes for transformer-XL is modified from [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl).
- Thanks [Yu-Hua Chen](https://github.com/ss12f32v) and [Hsiao-Tzu Hung](https://github.com/annahung31) for helping organize the codes.

