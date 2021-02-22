# Compound Word Transformer


Authors: [Wen-Yi Hsiao](https://github.com/wayne391), [Jen-Yu Liu](https://github.com/ciaua), [Yin-Cheng Yeh](https://github.com/yyeh26) and [Yi-Hsuan Yang](http://mac.citi.sinica.edu.tw/~yang/)

[**Paper (arXiv)**](https://arxiv.org/abs/2101.02402) | [**Audio demo (Google Drive)**](https://drive.google.com/drive/folders/1G_tTpcAuVpYO-4IUGS8i8XdwoIsUix8o?usp=sharing) | [**Blog**](https://ailabs.tw/human-interaction/compound-word-transformer-generate-pop-piano-music-of-full-song-length/) | [**Colab notebook**](https://colab.research.google.com/drive/1AU8iMhy10WxHj7yt3j8S3FQvvKvgXrr0)

Official PyTorch implementation of AAAI2021 paper "Compound Word Transformer: Learning to Compose Full-Song Musicover Dynamic Directed Hypergraphs".

We presented a new variant of the Transformer that can processes multiple consecutive tokens at once at a time step. The proposed method can greatly reduce the length of the resulting sequence and therefore enhance the training and inference efficiency. We employ it to learn to compose expressive Pop piano music of full-song length (involving up to 10K individual to23 kens per song). In this repository, we open source our **Ailabs.tw Pop17K** dataset, and the codes for unconditional generation.


## Dependencies

* python 3.6
* Required packages:
    * madmom
    * miditoolkit
    * pytorch-fast-transformers
 

``chorder`` is our in-house rule-based symbolic chord recognition algorithm, which is developed by our former intern - [joshuachang2311](https://github.com/joshuachang2311/chorder). He is also a jazz pianist :musical_keyboard:. 


## Model
In this work, we conduct two scenario of generation:
* unconditional generation
    * To see the experimental results and discussion, please refer to [here](https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/Experiments.md). 

* conditional generation, leadsheet to full midi (ls2midi)
    * [**Work in progress**] We plan to open source the code associated with this part in the future. 
        * melody extracyion (skyline) 
        * objective metrics
        * model

## Dataset
To prepare your own training data, please refer to [documentaion](https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/Dataset.md) for further understanding.  
Or, you can start with our **AIlabs.tw Pop17K**, which is available [here](https://drive.google.com/file/d/1qw_tVUntblIg4lW16vbpjLXVndkVtgDe/view?usp=sharing).

## Demo: Colab Notebook

The colab notebook is now available [here](https://colab.research.google.com/drive/1AU8iMhy10WxHj7yt3j8S3FQvvKvgXrr0).  
Thanks our intern [AdarshKumar712](https://github.com/AdarshKumar712) for organizing the codes.


## Acknowledgement
- PyTorch codes for transformer-XL is modified from [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl).
- Thanks [Yu-Hua Chen](https://github.com/ss12f32v) and [Hsiao-Tzu Hung](https://github.com/annahung31) for helping organize the codes.

