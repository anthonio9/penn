MODULE = 'penn'

# Configuration name
CONFIG = 'ppn-split-midi60'

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
MODEL = 'ppnmidi60'

PITCH_BINS = 60

PITCH_CATS = 6

GSET_SPLIT_PLAYERS = True

REMOVE_OVERHANGS = True

MIDI60 = True

MIDI_OFFSET_RAND = 36

GAUSSIAN_BLUR = False

INTERPOLATE_UNVOICED = False
