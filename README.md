
<!-- Setup -->

## Setup
 * Set up conda environment with python 3.9, ex: `conda create -n ca4mi python=3.9 anaconda`
 * `conda activate ca4mi`
 * `cd ca4mi/`
 * `pip install -r requirements.txt` 

## Datasets
 * Download the datasets and place them in the `~/Datasets/` directory, ex: `~/Datasets/BCICompetition-IV2a`, `~/Datasets/BCICompetition-IV2b`, `~/Datasets/openBMI/`
 * Datasets links:
 * **BCI Competition IV-2a & BCI Competition IV-2b**: (https://www.bbci.de/competition/iv/)
 * **OpenBMI**: retrieve from: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html#moabb.datasets.Lee2019_MI
 
 * ** Note: The datasets should be in the following format:
   ```sh
    ./Datasets/BCICompetition-IV2a/
    ├── A1.mat
    ├── A3.mat
    ├── A3.mat
    ├── A4.mat
    ├── ...
    ├── A8.mat
    └── A9.mat
    ```
    ```sh
    ./Datasets/BCICompetition-IV2b/
    ├── B1.mat
    ├── B2.mat
    ├── B3.mat
    ├── B4.mat
    ├── ...
    ├── B5.mat
    └── B6.mat
    ```
    ```sh  
    ./Datasets/openBMI/
    ├── s1.mat
    ├── s2.mat
    ├── s3.mat
    ├── s4.mat
    ├── ...
    ├── s53.mat
    └── S54.mat
    ```

## Running CA4MI
1. Clone this repository.
 
2. Set your data path in the configuration files.
    Set data path in ./configs/ca4mi.yml.
3. Run the following command to train the model:
   ```sh
   python main/main.py --config ./configs/ca4mi.yml
   ```
   The trained model will be saved in the `./checkpoints/ca4mi` directory.

## Running Baselines
1. Clone this repository.
 
2. Set your data path in the configuration files.
    Set data path in ./configs/ewc.yml.
3. Run the following command to train the model:
   ```sh
   python main/main.py --config ./configs/ewc.yml
   ```
   The trained model will be saved in the `./checkpoints/ewc` directory. Change the config file to run other baselines.

# Citation
If you find this code useful for your research, please cite our paper:
```sh
@article{li2025toward,
  title={Toward Memory-Efficient Continual Adaptation for MI-EEG Decoding in BCIs},
  author={Li, Dan and Shin, Hye-Bin and Lee, Seong-Whan},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  volume={56},
  number={1},
  pages={766--778},
  year={2025},
  publisher={IEEE}
}
 ```




