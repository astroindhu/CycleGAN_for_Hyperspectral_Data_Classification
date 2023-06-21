import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CycleGAN(keras.Model):
    def __init__(self, generator_A, generator_B, discriminator_A, discriminator_B):
        super(CycleGAN, self).__init__()
        self.generator_A = generator_A
        self.generator_B = generator_B
        self.discriminator_A = discriminator_A
        self.discriminator_B = discriminator_B

    def compile(self, generator_loss_fn, discriminator_loss_fn, cycle_loss_fn, identity_loss_fn, lambda_cycle=10.0,
                lambda_identity=0.5, learning_rate=2e-4):
        super(CycleGAN, self).compile()
        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.learning_rate = learning_rate
        self.generator_A_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5)
        self.generator_B_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5)
        self.discriminator_A_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5)
        self.discriminator_B_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5)

    def train_step(self, batch_data):
        real_A, real_B = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images
            fake_B = self.generator_A(real_A, training=True)
            fake_A = self.generator_B(real_B, training=True)

            # Reconstruct original images
            reconstructed_A = self.generator_B(fake_B, training=True)
            reconstructed_B = self.generator_A(fake_A, training=True)

            # Identity mapping
            same_A = self.generator_B(real_A, training=True)
            same_B = self.generator_A(real_B, training=True)

            # Generate adversarial labels
            disc_real_A = self.discriminator_A(real_A, training=True)
            disc_real_B = self.discriminator_B(real_B, training=True)
            disc_fake_A = self.discriminator_A(fake_A, training=True)
            disc_fake_B = self.discriminator_B(fake_B, training=True)

            # Calculate losses
            generator_A_loss = self.generator_loss_fn(disc_fake_A)
            generator_B_loss = self.generator_loss_fn(disc_fake_B)

            discriminator_A_loss = self.discriminator_loss_fn(disc_real_A, disc_fake_A)
            discriminator_B_loss = self.discriminator_loss_fn(disc_real_B, disc_fake_B)

            cycle_A_loss = self.cycle_loss_fn(real_A, reconstructed_A)
            cycle_B_loss = self.cycle_loss_fn(real_B, reconstructed_B)

            identity_A_loss = self.identity_loss_fn(real_A, same_A)
            identity_B_loss = self.identity_loss_fn(real_B, same_B)

            total_generator_loss = generator_A_loss + generator_B_loss + self.lambda_cycle * (
                        cycle_A_loss + cycle_B_loss) + self.lambda_identity * (identity_A_loss + identity_B_loss)
            total_discriminator_loss = discriminator_A_loss + discriminator_B_loss

        # Calculate gradients and apply updates
        generator_A_gradients = tape.gradient(total_generator_loss, self.generator_A.trainable_variables)
        self.generator_A_optimizer.apply_gradients(zip(generator_A_gradients, self.generator_A.trainable_variables))

        generator_B_gradients = tape.gradient(total_generator_loss, self.generator_B.trainable_variables)
        self.generator_B_optimizer.apply_gradients(zip(generator_B_gradients, self.generator_B.trainable_variables))

        discriminator_A_gradients = tape.gradient(total_discriminator_loss, self.discriminator_A.trainable_variables)
        self.discriminator_A_optimizer.apply_gradients(
            zip(discriminator_A_gradients, self.discriminator_A.trainable_variables))

        discriminator_B_gradients = tape.gradient(total_discriminator_loss, self.discriminator_B.trainable_variables)
        self.discriminator_B_optimizer.apply_gradients(
            zip(discriminator_B_gradients, self.discriminator_B.trainable_variables))

        return {
            "total_generator_loss": total_generator_loss,
            "total_discriminator_loss": total_discriminator_loss
        }
