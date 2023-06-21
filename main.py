from input_data import gt_class, gt_class_avg_spectra, indian_pines_corrected

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from generator_discriminator_model import make_discriminator_model, make_generator_model
from cyclegan import CycleGAN
from loss_functions import generator_loss_fn, discriminator_loss_fn, cycle_loss_fn, identity_loss_fn



# Instantiate generator and discriminator models
generator_A = make_generator_model()
generator_B = make_generator_model()
discriminator_A = make_discriminator_model()
discriminator_B = make_discriminator_model()


cycle_gan_model = CycleGAN(generator_A, generator_B, discriminator_A, discriminator_B)
cycle_gan_model.compile(generator_loss_fn, discriminator_loss_fn, cycle_loss_fn, identity_loss_fn)

# Start training
cycle_gan_model.fit(train_dataset, epochs=num_epochs)



