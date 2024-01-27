MODULE = 'penn'

# Configuration name
CONFIG = 'fcnf0++-gset-multi-hot'

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

# use only the voiced frames
VOICED_ONLY = True

LOSS = 'binary_cross_entropy'

LOSS_MULTI_HOT = True

GAUSSIAN_BLUR = False

STRING_INDEX = None

PITCH_CATS = 6

GSET_SPLIT_PLAYERS = True

EVALUATE = False

