# coding: utf-8
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Conv2D, Dropout, Lambda, add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import os
import time
from functools import partial
import util.pokemon_data as pokemon_data
import keras.backend as K
batch_size = 128
epochs = 10000
step = 100
input_dim = 64
clip_value = 0.01
img_shape = (40, 40, 3)
n_discr = 5


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def random_weighted_average(real_img, fake_img):
    alpha = K.random_uniform((batch_size, 1, 1, 1))
    r = lambda x: alpha * x
    f = lambda x: (1.0 - alpha) * x
    R = Lambda(r)(real_img)
    F = Lambda(f)(fake_img)
    return add([R, F])


# build model
def build_generator():
    model = Sequential([
        Dense(3 * 10 * 10, input_dim=input_dim, activation=LeakyReLU(0.2)),
        BatchNormalization(),
        Reshape((10, 10, 3)),
        UpSampling2D(),
        Conv2D(32, (4, 4), padding='same', activation=LeakyReLU(0.2)),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(64, (4, 4), padding='same', activation=LeakyReLU(0.2)),
        BatchNormalization(),
        Conv2D(3, (4, 4), padding='same', activation='tanh')
    ])
    return model


def build_discriminator():
    model = Sequential([
        Conv2D(32, (4, 4), strides=(2, 2), input_shape=img_shape, padding='same', activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Conv2D(128, (4, 4), strides=(2, 2), padding='same', activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Flatten(),
        # WGAN: do not sigmoid in last layer of discrimator
        Dense(1)
    ])
    return model


def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)

if __name__ == "__main__":

    train_images = pokemon_data.read_data()
    train_images = train_images * 2.0 - 1.0
    print('start')
    base_generator = build_generator()
    base_discriminator = build_discriminator()

    # before train
    # WGAN: use RMSprop
    optimizer = RMSprop()

    # build discriminator
    frozon_G = Model(inputs=base_generator.inputs, outputs=base_generator.outputs)
    frozon_G.trainable = False
    # Real Image input
    real_img = Input(shape=img_shape)
    # Noise input
    z_disc = Input(shape=(input_dim,))
    # Generate Image Input
    fake_img = frozon_G(z_disc)

    discriminator = Model(inputs=base_discriminator.inputs, outputs=base_discriminator.outputs)
    fake = discriminator(fake_img)
    valid = discriminator(real_img)

    interpolated_img = random_weighted_average(real_img, fake_img)
    validity_inter = discriminator(interpolated_img)

    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty'

    discriminator_model = Model(
        inputs=[real_img, z_disc],
        outputs=[valid, fake, validity_inter]
    )
    discriminator_model.compile(
        loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
        optimizer=optimizer, loss_weights=[1, 1, 10]
    )

    frozon_D = Model(inputs=base_discriminator.inputs, outputs=base_discriminator.outputs)
    frozon_D.trainable = False
    generator = Model(inputs=base_generator.inputs, outputs=base_generator.outputs)
    z_gen = Input(shape=(input_dim,))
    img = generator(z_gen)
    valid = frozon_D(img)
    generator_model = Model(z_gen, valid)
    generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)

    if os.path.exists('Dweight.h5'):
        print('preload weight')
        base_discriminator.load_weights('Dweight.h5')
        base_generator.load_weights('Gweight.h5')
        generator_model.load_weights('GANweight.h5')

    valid = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))
    dummy = np.zeros((batch_size, 1))

    r = 5
    c = 5
    noise_sample = np.random.normal(0, 1, (r * c, input_dim))

    # start train
    for i in range(epochs):
        beg = time.time()
        for j in range(step):
            for k in range(n_discr):
                train_mask = np.random.choice(len(train_images), batch_size)
                batch_images = train_images[train_mask]
                noise = np.random.normal(0, 1, (batch_size, input_dim))

                d_loss = discriminator_model.train_on_batch(
                    [batch_images, noise],
                    [valid, fake, dummy]
                )
            noise = np.random.normal(0, 1, (batch_size, input_dim))
            g_loss = generator_model.train_on_batch(noise, valid)

        print (i, "[D loss: ", d_loss, "] [G loss: ", g_loss, "]")

        r = 5
        c = 5
        samples = []
        gen_imgs = generator.predict(noise_sample)
        gen_imgs = (gen_imgs + 1.0) * 0.5
        pokemon_data.sample_image(gen_imgs, r, c, "GAN_%d" % i)
        end = time.time()
        print('use time:', end - beg)
        if not os.path.exists('output/weights'):
            os.mkdir('output/weights')
        generator_model.save_weights('output/weights/GANweight_%d.h5' % i)
        generator.save_weights('output/weights/Gweight_%d.h5' % i)
        discriminator.save_weights('output/weights/Dweight_%d.h5' % i)
    # 112000
