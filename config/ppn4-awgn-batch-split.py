MODULE = 'penn'

# Configuration name
CONFIG = 'ppn4-awgn-batch-split'

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

STRING_INDEX = None

# poly pitch net model
MODEL = 'ppn4'

PITCH_CATS = 6

NORMALIZATION = 'batch'

GSET_SPLIT_PLAYERS = True

GSET_AUGUMENT = [50, 30, 10, 5, 3]
