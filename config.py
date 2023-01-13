'''
Config file containing the hyperparameters
'''
DATASET = 11
LATENT_LOSS_WEIGHT = 1 # hyperparameter weight for the latent loss
PERCEPTUAL_LOSS_WEIGHT = 1 # hyperparameter weight for the perceptual loss

# weights for the mocoganhd discriminator
G_LOSS_2D_WEIGHT = 0.25
G_LOSS_3D_WEIGHT = 0.25

image_disc_weight = 0.5
video_disc_weight = 0.5

D_LOSS_WEIGHT = 0.1

SAMPLE_SIZE_FOR_VISUALIZATION = 8
DISC_LOSS_WEIGHT = 0.25 # TODO - modify