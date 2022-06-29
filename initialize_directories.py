import os

DATA_DIRECTORY = os.environ['DATA_DIRECTORY']
PROCESSED_PATH = os.path.join(DATA_DIRECTORY, 'processed')

if os.path.exists(PROCESSED_PATH):
   print('Processed directory already exists. Skip initialization? (y/n): ', end='')
   answer = input()
   if answer == 'y':
      print('Skipping initialization')
      exit()

print('Removing processed directory...')
os.rmdir(PROCESSED_PATH)

print('Creating processed directory...')
os.mkdir(PROCESSED_PATH)

print('Creating training, validation, and test directories with classified subdirectories...')
for directory in ['training', 'validation', 'test']:
    for class_name in ['bacterial', 'viral', 'normal']:
        os.makedirs(os.path.join(PROCESSED_PATH, directory, class_name))