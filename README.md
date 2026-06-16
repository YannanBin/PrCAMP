## Create and activate environment

```bash
conda env create -f environment.yml
conda activate py38
```


## Training 

The model relies on 2D Structures. To generate them in advance:

```
python SMILESto2D.py
```

To train a model for prediction, you can run:

```
python train_stage_one.py
python train_stage_two.py
```
