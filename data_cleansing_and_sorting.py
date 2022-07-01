import os
import re
from PIL import Image
from tqdm import tqdm

DATA_DIRECTORY = os.environ['DATA_DIRECTORY'] if 'DATA_DIRECTORY' in os.environ else './data'
RAW_DIRECTORIES = os.path.join(DATA_DIRECTORY, 'raw')
PROCESSED_DIRECTORIES = os.path.join(DATA_DIRECTORY, 'processed')

IMAGE_SIZE = (256, 256)

paths = [x for x in os.walk(RAW_DIRECTORIES) if len(x[-1]) > 0]

for path in paths:
    path_split = re.split('/|\\\\', path[0])
    split_category = path_split[3]
    classification = path_split[4]
    
    for i, name in tqdm(zip(range(len(path[-1])), path[-1]),
                        desc=f'{split_category} {classification}',
                        total=len(path[-1])):
        image = Image.open(os.path.join(path[0], name))
        resized = image.resize(IMAGE_SIZE)
        if classification == 'NORMAL':
            resized.save(os.path.join(PROCESSED_DIRECTORIES, split_category, 'normal', f'normal_{i}.jpeg'))
        elif classification == 'PNEUMONIA':
            if 'bacteria' in name:
                resized.save(os.path.join(PROCESSED_DIRECTORIES, split_category, 'bacterial', f'bacterial_{i}.jpeg'))
            elif 'virus' in name:
                resized.save(os.path.join(PROCESSED_DIRECTORIES, split_category, 'viral', f'viral_{i}.jpeg'))