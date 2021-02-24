## AdaCrowd: Unlabeled Scene Adaptation for Crowd Counting

This repository contains the code for [AdaCrowd: Unlabeled Scene Adaptation for Crowd
Counting]() <br>
IEEE Transactions on Multimedia 2021<br>

### Setup

Create a python virtual environment using either [virtualenv](https://docs.python.org/3/tutorial/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Install all the dependencies using `setup.py`
```python
python setup.py install
```

### Datasets
The details related to all the crowd counting datasets can be found in the following links.
1. [WorldExpo'10](http://www.ee.cuhk.edu.hk/~xgwang/expo.html)
2. [Mall](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)
3. [PETS](http://cs.binghamton.edu/~mrldata/pets2009)
4. [FDST](https://github.com/sweetyy83/Lstn_fdst_dataset)
5. [CityUHK-X](http://visal.cs.cityu.edu.hk/downloads/#cityuhk-x)

Note: For processing datasets, please refer to [C-3-Framework](https://github.com/gjy3035/C-3-Framework). Also update `DATA_PATH` in the `setting.py` for a particular dataset before using it for either training/testing.

### Training

1. **Baseline Methods**
    
    Refer to `config/baselines.py` configuration file to train baseline methods using different backbone networks (e.g. VGG-16 / ResNet-101) and to make appropriate changes.

    ```python
    python train_baselines.py
    ``` 
2. **AdaCrowd Methods**
    Refer to `config/adacrowd.py` configuration file to train adacrowd methods for different backbone networks.
    ```python
    python train_adacrowd.py
    ```

### Testing

Update `MODEL_PATH` in either `config/adacrowd.py` or `config/baselines.py` to test the model.

Refer to the following command-line arguments to run `test_adacrowd.py`.
```python
optional arguments:
  -h, --help            show this help message and exit
  -mn MODEL_NAME, --model_name MODEL_NAME
                        Name of the model to be evaluated
  -mp MODEL_PATH, --model_path MODEL_PATH
                        Path of the pre-trained model
  -data {WE,UCSD,Mall,PETS,City}, --dataset {WE,UCSD,Mall,PETS,City}
                        Dataset to be evaluated
  -k K_SHOT, --k_shot K_SHOT
                        Number of K images used for computing the affine transformation parameters
  -norm NUM_NORM, --num_norm NUM_NORM
                        Number of normalization layers
  -gpu GPU_ID, --gpu_id GPU_ID
                        GPU ID
  -r RESULT_FOLDER, --result_folder RESULT_FOLDER
                        Path of the results folder
  -t TRAILS, --trails TRAILS
                        Number of random trails to calculate mean and std scores
```

Similarly, refer to the following command-line arguments to run `test_baselines.py`.
```python
optional arguments:
  -h, --help            show this help message and exit
  -mn MODEL_NAME, --model_name MODEL_NAME
                        Name of the model to be evaluated
  -mp MODEL_PATH, --model_path MODEL_PATH
                        Path of the pre-trained model
  -data {WE,UCSD,Mall,PETS,City}, --dataset {WE,UCSD,Mall,PETS,City}
                        Dataset to be evaluated
  -norm NUM_NORM, --num_norm NUM_NORM
                        Number of normalization layers
  -gpu GPU_ID, --gpu_id GPU_ID
                        GPU ID
  -r RESULT_FOLDER, --result_folder RESULT_FOLDER
                        Path of the results folder
```

### Citation
```
@article{reddy2020adacrowd,
  title     =   {AdaCrowd: Unlabeled Scene Adaptation for Crowd Counting},
  author    =   {Reddy, Mahesh Kumar Krishna and Rochan, Mrigank and Lu, Yiwei and Wang, Yang},
  journal   =   {arXiv preprint arXiv:2010.12141},
  year      =   {2020}
}
```

#### Acknowledgements
This project borrows code from [C-3-Framework](https://github.com/gjy3035/C-3-Framework).

### License

The project is licensed under MIT license (please refer to LICENSE.txt for more details).
