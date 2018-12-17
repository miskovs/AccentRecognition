import os
import argparse
import librosa
import pickle
import tempfile

MIN_SILENCE_LENGTH_MS = 200
SILENCE_THRESHOLD_DB = -44
SAMPLING_RATE = 22050
MFCC_COUNT = 50

from simple_logger import SimpleLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to root directory of .wav files')
    parser.add_argument('language_class', help='Two characters indicating the language of speakers (e.g. EN for english, CH for chinese)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more stuff')
    args = parser.parse_args()

    logger = SimpleLogger(args.verbose)

    YOUR_ABSOLUTE_FFMPEG_LOCATION = r"C:\Users\micha\Downloads\ffmpeg-20181208-fe0416f-win64-static\ffmpeg-20181208-fe0416f-win64-static\bin"
    os.environ['PATH'] += os.pathsep + YOUR_ABSOLUTE_FFMPEG_LOCATION
    try:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
    except RuntimeWarning:
        logger.log("Please write your ffmpeg location into this file", force=True)

    if not os.path.exists(args.path) or not os.path.isdir(args.path):
        logger.log(f'{args.path} is not a valid directory')

    os.makedirs("out/", exist_ok=True)
    os.makedirs("temp/", exist_ok=True)
    output_file = open(f'out//{args.language_class}.pkl', 'wb')
    logger.log(f"Output file: out//{args.language_class}.pkl")

    for root, dirs, filenames in os.walk(args.path):
        for filename in filenames:
            if ".wav" in filename:
                logger.log(f"Analyzing {filename}..")
                wav_data = AudioSegment.from_wav(os.path.join(root, filename))
                audio_chunks = split_on_silence(wav_data, min_silence_len=MIN_SILENCE_LENGTH_MS, silence_thresh=SILENCE_THRESHOLD_DB)
                if not len(audio_chunks):
                    logger.log(f"{filename} contains silence only. Consider changing silence parameters", force=True)
                logger.log(f"Found {len(audio_chunks)} separate utterances, proceeding to MFCC extraction")
                for chunk in audio_chunks:
                    handle, abs_path = tempfile.mkstemp(prefix='chunk_', dir='temp/')

                    chunk.export(abs_path, format='wav')
                    waveform, sr = librosa.load(abs_path, sr=SAMPLING_RATE, mono=True)

                    os.close(handle)
                    os.remove(abs_path)

                    mfcc = librosa.feature.mfcc(waveform, sr, n_mfcc=MFCC_COUNT)
                    pickle.dump(mfcc, output_file)
            output_file.flush()
    output_file.close()