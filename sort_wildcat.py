import argparse
import os
import shutil
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('wav_root', help='Root dir of Wildcat dataset')
    args = parser.parse_args()

    all_wavs = glob.glob(args.wav_root + r'\Wildcat\audioFiles_*\task*\list*\talker*\*.wav')
    print(f'Found {len(all_wavs)} WAV files')
    if not len(all_wavs):
        exit()

    class_dict = dict()
    for filename in all_wavs:
        metadata = os.path.basename(filename).split('_')
        language_position = 2
        if len(metadata) > 6:
            language_position = 1
        language_class = metadata[language_position][:2]
        if language_class not in class_dict:
            class_dict[language_class] = []
        class_dict[language_class].append(filename)
    print(f'Sorted WAV files into language classes: {class_dict.keys()}')

    for language_class, file_list in class_dict.items():
        os.makedirs(os.path.join(args.wav_root, language_class), exist_ok=True)
        for file in file_list:
            shutil.move(file, os.path.join(args.wav_root, language_class))
