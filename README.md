# Chest X-Ray Pneumonia Detection

Data from [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Details for Contributors

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
