# NN config
INPUT_SHAPE = (None, None, 1)
NN_MODEL = 'strrn'
MPRRN_FILTERS_PER_LAYER = 32
MPRRN_FILTER_SHAPE = 3
MPRRN_RRU_PER_IRB = 1
MPRRN_IRBS = 1
L0_GRADIENT_MIN_LAMDA = 0.02
L0_GRADIENT_MIN_BETA_MAX = 0.00001

# Train config
LOAD_WEIGHTS = False
SAVE_AND_CONTINUE = True
JPEG_QUALITY = 50
EPOCHS = 50
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_INTERVAL = 8
LEARNING_RATE_DECAY = 10
DATASET_PREFETCH = 3000
BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1  # testing uses full images which takes a lot more memory
GRAD_NORM = 1.0  # max value for gradients. Clipping gradients to prevent NaN issues
ADAM_EPSILON = 0.001
ACCURACY_PSNR_THRESHOLD = 26.95
SAMPLE_IMAGES = ["sampleCartoonImage", "samplePhotoImage",
                 "sampleUrban100"]
CHECKPOINTS_PATH = "./checkpoints/"
DATASETS_DIR = "e:/datasets"
TRAINING_DATASET = "e:/datasets/div2k_dataset/preprocessed/"
