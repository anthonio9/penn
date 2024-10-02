MODULE = 'penn'

# Configuration name
CONFIG = 'polypennfcn-15ks-batch'

# gset only
DATASETS = ['gset']

EVALUATION_DATASETS = ['gset']

STEPS = 50000

LOG_INTERVAL = 500

CHECKPOINT_INTERVAL = 5000  # steps

# audio parameters
SAMPLE_RATE = 11025

# the original hopsize is 256 samples, this is 4 times less than that
HOPSIZE = 64 

WINDOW_SIZE = HOPSIZE

# use only the voiced frames
VOICED_ONLY = True

STRING_INDEX = None

# poly pitch net model
MODEL = 'polypennfcn2'

PITCH_CATS = 6

GSET_SPLIT_PLAYERS = True

NUM_TRAINING_FRAMES = 128 

BATCH_SIZE = 32 

NORMALIZATION = 'batch'

FCN = True

DECODER = 'argmax'

# Batch size to use for evaluation
EVALUATION_BATCH_SIZE = None

KERNEL_SIZE = 15 

PADDING_SIZE = 7

SILENCE_HEAD = True
