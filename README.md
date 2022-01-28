<h1 align="center"><b>Deep Narrow NTK</b></h1>
<h3 align="center"><b>NTK Analysis of Deep Narrow Neural Networks</b></h1>
<p align="center">
</p> 
 
--------------

<br>

This is an repository for [NTK Analysis of Deep Narrow Neural Networks]().

<p align="center">
<embed src="thumbnail.pdf" width="800px" height="2100px" />

## Requirements
All the necessary packages required to run the codes are reported in `requirements.txt`.
  
The main packages are:

```
python==3.8.12
matplotlib==3.5.0
numpy==1.21.2
pytorch==1.10.0
torchvision==0.11.1
```
  

## Usage
### NTK Convergence
  
```
python ntk_convergence.py [--init] [--depths] [--num_trials] [--iterations] [--dtype] 
                          [--scales] [--learning_rates] [--output_dir] [--dev_num] [--seed]

optional arguments:
  --init                How to Initialize the Model (custom being our initialization)
  --depths              List of Depths
  --num_trials          How many independent trials to run
  --iterations          Number of training iterations
  --dtype               Data Type of the tensor: float32 (single) or float64
                        (double)
  --scales              List of scaling factors C_L
  --learning_rates      List of learning rates
  --output_dir          Output directory
  --dev_num             List of Cuda Device Number
  --seed                Random seed
```
  
For example,
  
```
python ntk_convergence.py --init custom --depths 100 10000 --num_trials 10 --iterations 2000 --dtype float64 --scales 5 50000 --learning_rate 0.0001 1e-14 --output_dir ./result --dev_num 0 --seed 0
```

### Mnist Training
  
```
train_mnist.py [--type] [--binary] [--init] [--depth] [--scale] [--learning_rate] 
               [--epochs] [--batch_size] [--eval_every] [--output_dir] [--dev_num] 
               [--seed] [--dtype]

optional arguments:
  --type                Type of architecture (mlp or cnn)
  --binary              Whether it is a binary classification or not
  --init                How to Initialize the Model (custom being our initialization)
  --depth               Depth of the model
  --scale               Scaling factors C_L
  --learning_rate       Learning rate
  --epochs              Epochs
  --batch_size          Batch Size
  --eval_every          How often to evaluate the model with test set
  --output_dir          Output directory
  --dev_num             List of Cuda Device Number
  --seed                Random seed
  --dtype               Data Type of the tensor: float32 (single) or float64
                        (double)
```
 
 For example
 1) MLP 4000 Depth
 ```
 python train_mnist.py --type mlp --binary False --init custom --depth 4000 --scale 4 --learning_rate 1e-5 --epochs 500 --batch_size 512 --eval_every 50 --output_dir ./result --dev_num 0 1 2 3 --seed 0 --dtype float32
 ```
 
 2) CNN L = 4000, 10 classes
```
 python train_mnist.py --type cnn --binary False --init custom --depth 4000 --scale 4 --learning_rate 1e-5 --epochs 2000 --batch_size 512 --eval_every 50 --output_dir ./result --dev_num 0 --seed 0 --dtype float32
```
 
 3) CNN L = 12000, binary
 ```
 python train_mnist.py --type cnn --binary True --init custom --depth 12000 --scale 12 --learning_rate 1e-8 --epochs 1000 --batch_size 512 --eval_every 50 --output_dir ./result --dev_num 0 1 2 3 --seed 0 --dtype float64
```
