import shutil

import torchutil

import penn


###############################################################################
# Download datasets
###############################################################################


@torchutil.notify('download')
def datasets(datasets):
    """Download datasets"""
    if 'mdb' in datasets:
        mdb()

    if 'ptdb' in datasets:
        ptdb()

    if 'gset' in datasets:
        gset()


###############################################################################
# Individual datasets
###############################################################################


def mdb():
    """Download mdb dataset"""
    torchutil.download.targz(
        'https://zenodo.org/record/1481172/files/MDB-stem-synth.tar.gz',
        penn.DATA_DIR)

    # Delete previous directory
    shutil.rmtree(penn.DATA_DIR / 'mdb', ignore_errors=True)

    # Rename directory
    shutil.move(penn.DATA_DIR / 'MDB-stem-synth', penn.DATA_DIR / 'mdb')


def ptdb():
    """Download ptdb dataset"""
    directory = penn.DATA_DIR / 'ptdb'
    directory.mkdir(exist_ok=True, parents=True)
    torchutil.download.zip(
        'https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH_DATA_ZIPPED.zip',
        directory)


def gset():
    """Download GuitarSet dataset"""
    torchutil.download
    directory_annotation = penn.DATA_DIR / 'gset' / 'annotations'
    directory_audio_mono_mic = penn.DATA_DIR / 'gset' / 'audio-mono-mic'

    directory_annotation.mkdir(exist_ok=True, parents=True)
    directory_audio_mono_mic.mkdir(exist_ok=True, parents=True)

    torchutil.download.zip(
        'https://zenodo.org/records/3371780/files/annotation.zip',
        directory_annotation)

    torchutil.download.zip(
        'https://zenodo.org/records/3371780/files/audio_mono-mic.zip',
        directory_audio_mono_mic)
