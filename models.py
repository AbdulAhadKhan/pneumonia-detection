import tensorflow

layers = tensorflow.keras.layers
Model = tensorflow.keras.models.Model
regularizers = tensorflow.keras.regularizers

def alex_net(input_shape=(200, 200, 1), number_of_classes=3, optimizer='adam'):
    input_layer = layers.Input(input_shape)
    conv1 = layers.Conv2D(96, 11, strides=4, activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(input_layer)
    pool1 = layers.MaxPool2D(3, 2)(conv1)

    conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(pool1)
    pool2 = layers.MaxPool2D(3, 2)(conv2)

    conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(pool2)
    conv4 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(conv3)
    pool3 = layers.MaxPool2D(3, 2)(conv4)

    flattened = layers.Flatten()(pool3)
    dense1 = layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(flattened)
    dropout1 = layers.Dropout(0.5)(dense1)

    dense2 = layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(l=0.001))(dropout1)
    drop2 = layers.Dropout(0.5)(dense2)

    preds = layers.Dense(number_of_classes, activation='softmax')(drop2)

    loss = 'binary_crossentropy' if number_of_classes == 2 else 'categorical_crossentropy'

    model = Model(input_layer, preds)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    return model