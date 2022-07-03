# Chest X-Ray Pneumonia Detection

Data from [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Details for Contributors

### Order of execution

1) initialize_model.py
2) data_cleansing_and_sorting.py
3) train_model.py

### Environment Variables

`DATA_DIRECTORY`:   
- Path to the directory containing the data. Defaults to `./data`
- Assumes the following directory structure:

```
data/
└── raw/
    ├── chest-xray/
    │   ├── test/
    │   │   ├── NORMAL/
    │   │   └── PNEUMONIA/
    │   ├── train/
    │   │   ├── NORMAL/
    │   │   └── PNEUMONIA/
    │   └── val/
    │       ├── NORMAL/
    │       └── PNEUMONIA/
    ├── test/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── val/
        ├── NORMAL/
        └── PNEUMONIA/
```
