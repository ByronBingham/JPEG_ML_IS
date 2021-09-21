# NN config
INPUT_SHAPE = (None, None, 3)
NN_MODEL = 'strrn'
MPRRN_FILTERS_PER_LAYER = 128
MPRRN_FILTER_SHAPE = 3
MPRRN_RRU_PER_IRB = 3
MPRRN_IRBS = 3

# Train config
LOAD_WEIGHTS = False
JPEG_QUALITY = 50
EPOCHS = 50
LEARNING_RATE = 0.00001
DATASET_PREFETCH = 100
BATCH_SIZE = 2
GRAD_NORM = 1.0  # max value for gradients. Clipping gradients to prevent NaN issues
ADAM_EPSILON = 0.001
SAMPLE_IMAGES = ["./sampleCartoonImage", "./samplePhotoImage"]
CHECKPOINTS_PATH = "./checkpoints"
