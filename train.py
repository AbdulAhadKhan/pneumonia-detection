import os
import pickle

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from generator import Generator
from models import alex_net

train_dir = './data/processed/train'
validation_dir = './data/processed/val'

model_dir = './models'
structure_dir = './diagrams/structures'

epochs = 100
batch_size = 18
val_batch_size = 8
adam = optimizers.Adam(learning_rate=3e-4, decay=1e-6)
model_details = []

alex_net = alex_net(optimizer=adam)

train_datagen = ImageDataGenerator(rescale=1/.255)
validation_datagen = ImageDataGenerator(rescale=1/.255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(200, 200), batch_size=batch_size, 
    class_mode='categorical', color_mode='grayscale'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(200, 200), batch_size=val_batch_size, 
    class_mode='categorical', color_mode='grayscale'
)
train_datagen_custom = Generator(train_datagen, train_dir, batch_size=batch_size)
validation_datagen_custom = Generator(validation_datagen, validation_dir, batch_size=val_batch_size)

train_generator_custom = train_datagen_custom.get_image_batch()
validation_generator_custom = validation_datagen_custom.get_image_batch()

classes = train_generator.class_indices
classes = {index:names for names, index in classes.items()}

with open('classes.pickle', 'wb') as fp:
    pickle.dump(classes, fp)
    print('Classes dumped....')

alex_net_basic = (alex_net, 'Alex Net Basic', train_generator, validation_generator)
model_details.append(alex_net_basic)

alex_net_normal = (alex_net, 'Alex Net Normalised', train_generator_custom, validation_generator_custom)
model_details.append(alex_net_normal)

alex_net_rescaled = (alex_net, 'Alex Net Rescaled', train_generator, validation_generator)
model_details.append(alex_net_rescaled)

for model_detail in model_details:
    model, model_title, train, validate = model_detail

    model_name = model_title.lower().replace(' ', '_')
    structure_name = '{}.png'.format(model_name)
    final_model = '{}.h5'.format(model_name)

    structure_name = os.path.join(structure_dir, structure_name)
    final_model = os.path.join(model_dir, final_model)

    print(model_title)

    history_model = model.fit(
        train, epochs=epochs, verbose=1, validation_data=validate,
        steps_per_epoch=20, validation_steps=5
    )