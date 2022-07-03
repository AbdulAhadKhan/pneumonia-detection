import os
import gc
import yaml

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from models import alex_net

DATA_DIRECTORY = os.environ['DATA_DIRECTORY'] \
    if 'DATA_DIRECTORY' in os.environ else './data'

train_dir = train_dir = './data/processed/train'
validation_dir = './data/processed/val'

models = [
    {
        'name': 'AlexNet',
        'model': alex_net,
    },
]

train = ImageDataGenerator(rescale=1./255)
validate = ImageDataGenerator(rescale=1./255)
optimizer_list = [
    {
        'name': "Adam",
        'optimizer': optimizers.Adam
    },
    {
        'name': "Nadam",
        'optimizer': optimizers.Nadam
    },
    {
        'name': "SGD",
        'optimizer': optimizers.SGD
    }
]

epochs = 100
learning_rate_list = [1e-3, 3e-3, 1e-4, 3e-4]
batch_size_list = [16, 24, 32, 64]
val_batch_sizes_list = [int(x/4) for x in batch_size_list]
steps_per_epoch_list = [int(1500/batch_size)
                        for batch_size in batch_size_list]
validation_steps_list = [int(500/val_batch_size)
                         for val_batch_size in val_batch_sizes_list]
early_stopping = EarlyStopping(monitor='acc', patience=5, restore_best_weights=True, verbose=1)

for model in models:
    for optimizer in optimizer_list:
        for learning_rate in learning_rate_list:
            for batch_size, val_batch_size, steps_per_epoch, validation_steps in \
                    zip(batch_size_list, val_batch_sizes_list, steps_per_epoch_list, validation_steps_list):
                
                model_name = f'{model["name"]}_{optimizer["name"]}_{learning_rate}_{batch_size}'

                train_datagenerator = train.flow_from_directory(
                    train_dir, target_size=(256, 256), batch_size=batch_size,
                    class_mode='categorical', color_mode='grayscale'
                )

                validation_datagenerator = validate.flow_from_directory(
                    validation_dir, target_size=(256, 256), batch_size=val_batch_size,
                    class_mode='categorical', color_mode='grayscale'
                )

                initialized_model = model['model'](optimizer=optimizer['optimizer'](
                    learning_rate=learning_rate), input_shape=(256, 256, 1))

                final_model = f'{model_name}.h5'
                final_model = os.path.join(
                    DATA_DIRECTORY, 'models', final_model)
                print(model_name)

                history = initialized_model.fit(
                    train_datagenerator, epochs=epochs, verbose=1, validation_data=validation_datagenerator,
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps , callbacks=[early_stopping]
                )

                initialized_model.save(final_model)
                with open(os.path.join(DATA_DIRECTORY, 'models', f'{model_name}.yml'), 'w') as f:
                    yaml.dump(history.history, f)

                del final_model, initialized_model, history
                del train_datagenerator, validation_datagenerator

                gc.collect()
