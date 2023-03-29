# ERM++: An Improved Baseline for Domain Generalization

Official PyTorch implementation of [ERM++: An Improved Baseline for Domain Generalizaton]

Piotr Teterwak, Kuniaki Saito, Theodoros Tsiligkaridis, Kate Saenko, Bryan A. Plummer




## Installation

### Dependencies

```sh
conda env create -f requirements.yml
```

### Dataset Download


```sh
python -m scripts.download --data_dir=/my/datasets/path

#For PACS
git clone https://github.com/MachineLearning2020/Homework3-PACS

```

### Model Download
 ```sh
 cd scripts
 bash download_models.sh

```

This downloads all models except for MEAL distilled models. To download those,
please see the MEAL github (repository)[https://github.com/szq0214/MEAL-V2].

### Data Path Specification

Modify the data paths in   ```data/dataset.py```, at the top of the file.

## Running ERM++

An example, which splits off 20% of the training data for validation.
```sh
python main_erm.py --save_name <SAVE_NAME> --dataset domainnet --training_data "clipart infograph real quickdraw sketch" --validation_data painting --sma --save_dir <SAVE_DIR> --steps  60000 --train-val-split 0.8 --lr 5e-5 --save-freq 1000 --linear-steps 500 --sma-start-iter 600 --arch resnet_timm_augmix
```

Then, find the number of steps corresponding to the highest (printed in the log) validation accuracy, and retrain on the full data:

```sh
python main_erm.py --save_name <SAVE_NAME> --dataset domainnet --training_data "clipart infograph real quickdraw sketch" --validation_data painting --sma --save_dir <SAVE_DIR> --steps  60000 --lr 5e-5 --save-freq 1000 --linear-steps 500 --sma-start-iter 600 --arch resnet_timm_augmix
```


To see domain names for different datasets, please see the ```data/dataset.py``` file and search for transform_dict variables for different data.



## License and Acknowledgements

This project is released under the MIT license, included [here](./LICENSE).

This project include some code from [facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed) (MIT license),[kakaobrain/miro](https://github.com/kakaobrain/miro) (MIT license), and [salesforce/ensemble-of-averages](https://github.com/salesforce/ensemble-of-averages).  The structure and some text of the  README is borrowed from the  MIRO repository.
