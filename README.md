# PyTorch Implementation of PACT: Parameterized Clipping Activation for Quantized Neural Networks
[Paper : PACT](https://arxiv.org/abs/1805.06085) 
I have implemented to reproduce quantization paper PACT on CIFAR10 with ResNet20
I have kept all experiment with first layer and last layer in 8bitwidth
Please look at **Table 2 in the paper**
| Bitwidth      | Original Paper ERR | Reproduced result ERR|
|-----------|---------:|--------:|
| 5 bitwidth|    8.4%    | 8.4%   |
| 4 bitwidth|    9.2%    | 9.11%   | 
| 3 bitwidth|    9.4%    | 9.91%   | 
| 2 bitwidth|    11.1%    | 12.93%   | 

## Note
3 bitwidth and 2 bitwidth has slightly lower performance than actual paper.

## Details of Pipeline
I have adopted training pipeline and pre-trained model from Akamaster's repo(https://github.com/akamaster/pytorch_resnet_cifar10) 

