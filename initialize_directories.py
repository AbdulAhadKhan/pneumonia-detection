import os
import shutil

DATA_DIRECTORY = os.environ['DATA_DIRECTORY'] if 'DATA_DIRECTORY' in os.environ else './data'
PROCESSED_PATH = os.path.join(DATA_DIRECTORY, 'processed')

if not os.path.exists(DATA_DIRECTORY):
    print('Could not find data directory. ' + 
          'Please set the environment variable DATA_DIRECTORY to the correct path.')
    exit(1)

if os.path.exists(PROCESSED_PATH):
    answer = input('Processed data directory already exists. ' +
                     'Do you want to delete it? (y/N) ')
    if answer == 'y':
        shutil.rmtree(PROCESSED_PATH)
    else:
        print('Exiting...')
        exit(0)

print('Creating processed directory...')
os.makedirs(PROCESSED_PATH)

print('Creating training, validation, and test directories with classified subdirectories...')
for directory in ['train', 'val', 'test']:
    for class_name in ['bacterial', 'viral', 'normal']:
        os.makedirs(os.path.join(PROCESSED_PATH, directory, class_name))

if os.path.exists(os.path.join(DATA_DIRECTORY, 'models')):
    answer = input('Models directory already exists. ' +
                    'Deleting it will delete all models in it. ' +
                    'Do you want to delete it? (y/N) ')
    if answer == 'y':
        shutil.rmtree(os.path.join(DATA_DIRECTORY, 'models'))
    else:
        print('Exiting...')
        exit(0)

print('Creating directory for trained models...')
os.makedirs(os.path.join(DATA_DIRECTORY, 'models'))