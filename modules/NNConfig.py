#######################################
# NN config
#######################################
INPUT_SHAPE = (None, None, 1)
IMAGE_CHANNELS = 1
NN_MODEL = 'dc_hourglass_interconnect_top_half_4'
DUAL_CHANNEL_MODELS = 'strrn strrn_encodedecode strrn_no_irb_residual strrn_no_irb_residual_encodedecode \
                dualchannelinterconnect_4 dualchannelinterconnect_struct_encodedecode dc_hourglass_interconnect_11\
                 dc_hourglass_interconnect_top_half_4 dc_hourglass_interconnect_bottom_half_1'
JPEG_QUALITY = 10

# STRRN config
MPRRN_TRAINING = 'None'  # 'aggregator'
STRUCTURE_MODEL = 'mprrn_structure'
TEXTURE_MODEL = 'mprrn_texture'
PRETRAINED_MODEL_PATH = './pretrainedModels/'
PRETRAINED_MPRRN_PATH = './pretrainedModels/'
PRETRAINED_STRUCTURE_PATH = './pretrainedModels/structure_checkpoint/'
PRETRAINED_TEXTURE_PATH = './pretrainedModels/texture_checkpoint/'
MPRRN_FILTERS_PER_LAYER = 32
STRUCTURE_FILTERS_PER_LAYER = 64
TEXTURE_FILTERS_PER_LAYER = 64
MPRRN_FILTER_SHAPE = 3
MPRRN_RRU_PER_IRB = 1
MPRRN_IRBS = 1
if JPEG_QUALITY == 10:
    L0_GRADIENT_MIN_LAMDA = 0.04
elif JPEG_QUALITY == 20:
    L0_GRADIENT_MIN_LAMDA = 0.02
elif JPEG_QUALITY == 30:
    L0_GRADIENT_MIN_LAMDA = 0.01
elif JPEG_QUALITY == 40:
    L0_GRADIENT_MIN_LAMDA = 0.005
else:
    L0_GRADIENT_MIN_LAMDA = 0.005
L0_GRADIENT_MIN_BETA_MAX = 10000

# DC Hourglass config
TOP_HALF_MODEL = 'dc_hourglass_interconnect_top_half_4'

#######################################
# Train config
#######################################
LOAD_WEIGHTS = False
SAVE_AND_CONTINUE = True
SAVE_TEST_OUT = True

EPOCHS = 20
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_INTERVAL = 8
LEARNING_RATE_DECAY = 10
LOSS_FUNCTION = 'mge_mse'  # 'mse' 'mprrn_loss' 'mse'

GRAD_NORM = 1.0  # max value for gradients. Clipping gradients to prevent NaN issues
ADAM_EPSILON = 0.001
if JPEG_QUALITY == 10:
    ACCURACY_PSNR_THRESHOLD = 28.43
elif JPEG_QUALITY == 20:
    ACCURACY_PSNR_THRESHOLD = 30.87
elif JPEG_QUALITY == 30:
    ACCURACY_PSNR_THRESHOLD = 32.93
elif JPEG_QUALITY == 40:
    ACCURACY_PSNR_THRESHOLD = 33.87
else:
    ACCURACY_PSNR_THRESHOLD = 33.87

BATCH_SIZE = 32
DATASET_PREFETCH = BATCH_SIZE * 5
TEST_BATCH_SIZE = 1  # testing uses full images which takes a lot more memory
DROPOUT_RATE = 0.2

DATASET_EARLY_STOP = True
TRAIN_EARLY_STOP = 1  # 000  # number of batches
VALIDATION_EARLY_STOP = 1
TEST_EARLY_STOP = 1  # 00

EVEN_PAD_DATA = 8  # should be powers of 2

TRAIN_DIFF = False

SAMPLE_IMAGES = ["sampleCartoonImage",  # "samplePhotoImage",
                 "sampleUrban100"]
CHECKPOINTS_PATH = "./checkpoints/"
DATASETS_DIR = "e:/datasets"
TRAINING_DATASET = 'div2k_greyscale_ql10'  # 'div2k_greyscale_ql10'
RESULTS_DIR = "e:/savedResults"

#######################################
# Test config
#######################################
USE_CPU_FOR_HIGH_MEMORY = True
TEST_DATASET = "urban100_greyscale_ql10"  # "reds_lr_dataset" "urban100_dataset4x" "bsd_s500_test_dataset"
# "pixiv_test_dataset" "urban100_greyscale_ql10"
TEST_CHECKPOINT_DIR = PRETRAINED_MODEL_PATH + "aggregator_checkpoint/"  # "aggregator_checkpoint/"
# "dc_interconnect_checkpoint/"
