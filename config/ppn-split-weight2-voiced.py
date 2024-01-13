MODULE = 'penn'

# Configuration name
CONFIG = 'ppn-split-weight2-voiced'

# gset only
DATASETS = ['gset']

EVALUATION_DATASETS = ['gset']

LOG_INTERVAL = 500

# audio parameters
SAMPLE_RATE = 11025

# the original hopsize is 256 samples, this is 4 times less than that
HOPSIZE = 64 

# use only the voiced frames
VOICED_ONLY = True

STRING_INDEX = None

# poly pitch net model
MODEL = 'ppn'

PITCH_CATS = 6

GSET_SPLIT_PLAYERS = True

WEIGHT_DECAY = 2e-4
