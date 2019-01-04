# coding: utf-8

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Conv2D, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import time
import util.pokemon_data as pokemon_data
train_images = pokemon_data.read_data()
print('start')
batch_size = 128
epochs = 5000
step = 1000
input_dim = 128


# build model
def build_generator():
    model = Sequential([
        Dense(3 * 10 * 10, input_dim=input_dim, activation=LeakyReLU(0.2)),
        BatchNormalization(),
        Reshape((10, 10, 3)),
        UpSampling2D(),
        Conv2D(64, (5, 5), padding='same', activation=LeakyReLU(0.2)),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(3, (5, 5), padding='same', activation='sigmoid')
    ])
    return model


def build_discriminator():
    model = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), input_shape=(40, 40, 3), padding='same', activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

base_generator = build_generator()
base_discriminator = build_discriminator()

# before train
optimizer = Adam()

# discriminator
discriminator = Model(inputs=base_discriminator.inputs, outputs=base_discriminator.outputs)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
discriminator.summary()

# GAN
generator = Model(inputs=base_generator.inputs, outputs=base_generator.outputs)
frozon_D = Model(inputs=base_discriminator.inputs, outputs=base_discriminator.outputs)
frozon_D.trainable = False
z = Input(shape=(batch_size,))
img = generator(z)
validity = frozon_D(img)

combine = Model(z, validity)
combine.compile(loss='binary_crossentropy', optimizer=optimizer)
combine.summary()

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

# start train
for i in range(epochs):
    beg = time.time()
    for j in range(step):
        train_mask = np.random.choice(len(train_images), batch_size)
        batch_images = train_images[train_mask]
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(batch_images, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, input_dim))
        g_loss = combine.train_on_batch(noise, valid)
    print ("%d %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
        i,
        j,
        d_loss[0],
        100 * d_loss[1],
        g_loss
    ))

    r = 5
    c = 5
    samples = []
    noise = np.random.normal(0, 1, (r * c, input_dim))
    gen_imgs = generator.predict(noise)
    pokemon_data.sample_image(gen_imgs, r, c, "GAN_%d" % i)
    end = time.time()
    print('use time:', end - beg)
    if not os.path.exists('output/weights'):
        os.mkdir('output/weights')
    combine.save_weights('output/weights/GANweight_%d.h5'%i)
    generator.save_weights('output/weights/Gweight_%d.h5'%i)
    discriminator.save_weights('output/weights/Dweight_%d.h5'%i)
