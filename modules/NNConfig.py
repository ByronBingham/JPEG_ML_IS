# NN config
INPUT_SHAPE = (None, None, 3)
NN_MODEL = 'eqlri'

# Train config
JPEG_QUALITY = 50
EPOCHS = 50
LEARNING_RATE = 0.0001
DATASET_PREFETCH = 100
BATCH_SIZE = 3
GRAD_NORM = 1.0  # max value for gradients. Clipping gradients to prevent NaN issues
