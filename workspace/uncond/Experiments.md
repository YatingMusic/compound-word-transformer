# Experiments

Experimental results of **unconditional** generation. We also elaborate detailed settings which are omitted in the paper because of the space limitation.

## Run the Codes
**cp-linear**

Edit the configration part at the begining of the `main-cp.py` file first.

```bash 
python main-cp.py
```

**remi-xl**

Edit the `config.yml` file first.

```bash 
# train
python train.py

# inference
python inference.py
```


## Model Settings
Backbone model:
* linear transformer (Linear): ["Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"](https://arxiv.org/abs/2006.16236)
* transformer-XL (XL): ["Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"](https://arxiv.org/abs/1901.02860)


All transformers share the same settings:
* number of layers: 12 
* number of heads: 8
* model hidden size: 512
* feed-forward layer size: 2,048

| Settings                    |  REMI + XL             |    CP + Linear         |  
|:---------------------------:|:----------------------:|:----------------------:|
| learning rate               | 2e-4                   | 1e-4                   |
| songs for training          | 1,612                  | 1,675                  |
| sequence length             | 7,680 (512 x 15)       | 3,584                  |
| dicionary size              | 332                    | 342                    |
| parameter amount            | 41,291,084             | 39,016,630             |
| recepetive field (train)    | 512 + 512 for memory   | 3,584                  |

Because the differnece of nature between the two representations, it's hard to keep the training data for the 2 settings totally equal. We adjust the `number of songs` and the `sequence length` to achieve reasonable balance. 

Under the limitation of hardware budget (single 2080ti GPU), we report the comparison between the 2 settings:
* REMI + XL: remi representation, transformer-XL
* CP + Linear: compound word (CP) representation, linear transformer


## Evaluation
### Memory Usage
The hardware budget is a single GPU and we use **Nvidia GeForce RTX 2080 GPU**, which has 11 GB memory. 

| # |  Representation+model  |    batch size          |   Memory Usage (GB)    |  Evaluation |
|:-:|:----------------------:|:----------------------:|:----------------------:|:-----------:|
| 1 |  REMI + XL             |         4              |   4                    |             |
| 2 |  REMI + XL             |         10             |   10                   |      O      |
| 3 |  CP + Linear           |         4              |   10                   |      O      |

Fro fair comparison, we let every model setting have its maximum memory consumption.
Notice that 


### Training  Efficiency 
Relation between quality of generated samples and the loss could be different according to different settings. Here, we still measure the training efficiency based on cross-entropy loss. 

[WIP]


### Inference Efficiency and Performance
We let each model to generate 50 songs and record the consumption time.

Records (JSON file):
* [cp-linear](./cp-linear/runtime_stats.json)
* [remi-xl](./remi-xl/runtime_stats.json)

|  Representation+model  |    Ave. Song Time      |   EOS    |
|:----------------------:|:----------------------:|:--------:|
|  REMI + XL             |     195.25446          |    X     |
|  CP + Linear           |      20.91956          |    O     |

`EOS` indicates whether the models are able to stop generation automatically - generating EOS token. 
For the CP+Linear setting, it only takes less than half minutes to generate a song and it also reveals its potential in real-time applications.

## Results
### Generated MIDI Files
* [cp-linear](./cp-linear/gen_midis)
* [remi-xl](./remi-xl/gen_midis)


### Checkpoints
* [cp-linear](https://drive.google.com/drive/folders/114uore7LHjAsM4eKXG9TfVZL5S3YY7nZ?usp=sharing)
* [remi-xl](https://drive.google.com/drive/folders/1tCaWQisPp_bcXKH5J3Nxmv6kUzJXs6qw?usp=sharing)

## Discussion
### About the generated samples
We find that the generated pieces of REMI-XL tend to stick to some patterns and occasionally fall to loop them for a quite long time, or even the entire song. The quality within the loop is suprisingly organized (clearly arpeggio in left hand, melody line in right hand), but also a bit of tedious. The samples from CP-linear have a rather different texture, the "structure" diversity is richer but it's also more aggressive in selecting pitches.

### About EOS
It turns out that REMI-XL failed in generating EOS sequence, which implies the sequence length might exceed the "effective length" becuase of it's rnn nature. In fact, we also tried remi+linear and cp+linear, and both of them success in this criterion.
