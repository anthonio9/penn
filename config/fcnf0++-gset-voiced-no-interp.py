MODULE = 'penn'

# Configuration name
CONFIG = 'fcnf0++-gset-voiced-no-interp'

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

INTERPOLATE_UNVOICED = False
