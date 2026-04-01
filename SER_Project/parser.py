import librosa as lr
import numpy as np
import pandas as pd
import os

def _file_name_to_metadata(fn: str):
    # Split the filename by dashes and remove the extension
    parts = fn.split('.')[0].split('-')
    
    meta = {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': parts[2],
        'intensity': parts[3],
        'statement': parts[4],
        'repetition': parts[5],
        'actor': parts[6]
    }

    return meta

def _wav_to_features(path: str):
    y, sr = lr.load(path, sr=48000)

    # MFCC features
    mfcc = lr.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_min = np.min(mfcc, axis=1)
    mfcc_max = np.max(mfcc, axis=1)
    
    # MFCC delta features
    mfcc_delta = lr.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)
    mfcc_delta_min = np.min(mfcc_delta, axis=1)
    mfcc_delta_max = np.max(mfcc_delta, axis=1)

    # Chroma features
    chroma = lr.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    #chroma_min = np.min(chroma, axis=1)
    #chroma_max = np.max(chroma, axis=1)

    # Mel Spectrogram features
    mel_spec = lr.feature.melspectrogram(y=y, sr=sr)
    mel_spec_mean = np.mean(mel_spec, axis=1)
    mel_spec_std = np.std(mel_spec, axis=1)
    #mel_spec_min = np.min(mel_spec, axis=1)
    mel_spec_max = np.max(mel_spec, axis=1)

    # Zero Crossing Rate features
    zcr = lr.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr, axis=1)
    zcr_std = np.std(zcr, axis=1)
    zcr_min = np.min(zcr, axis=1)
    zcr_max = np.max(zcr, axis=1)

    # RMS Energy features
    rms = lr.feature.rms(y=y)
    rms_mean = np.mean(rms, axis=1)
    rms_std = np.std(rms, axis=1)
    rms_min = np.min(rms, axis=1)
    rms_max = np.max(rms, axis=1)

    # Spectral Centroid features
    sp = lr.feature.spectral_centroid(y=y, sr=sr)
    sp_mean = np.mean(sp, axis=1)
    sp_std = np.std(sp, axis=1)
    sp_min = np.min(sp, axis=1)
    sp_max = np.max(sp, axis=1)

    # Spectral Bandwidth features
    sb = lr.feature.spectral_bandwidth(y=y, sr=sr)
    sb_mean = np.mean(sb, axis=1)
    sb_std = np.std(sb, axis=1)
    sb_min = np.min(sb, axis=1)
    sb_max = np.max(sb, axis=1)

    # Spectral Rolloff features
    srf = lr.feature.spectral_rolloff(y=y, sr=sr)
    srf_mean = np.mean(srf, axis=1)
    srf_std = np.std(srf, axis=1)
    srf_min = np.min(srf, axis=1)
    srf_max = np.max(srf, axis=1)


    features = {
        'mfcc_mean': mfcc_mean,
        'mfcc_std': mfcc_std,
        'mfcc_min': mfcc_min,
        'mfcc_max': mfcc_max,
        'mfcc_delta_mean': mfcc_delta_mean,
        'mfcc_delta_std': mfcc_delta_std,
        'mfcc_delta_min': mfcc_delta_min,
        'mfcc_delta_max': mfcc_delta_max,
        'chroma_mean': chroma_mean,
        'chroma_std': chroma_std,
        #'chroma_min': chroma_min,
        #'chroma_max': chroma_max,
        'mel_spec_mean': mel_spec_mean,
        'mel_spec_std': mel_spec_std,
        #'mel_spec_min': mel_spec_min,
        'mel_spec_max': mel_spec_max,
        'zcr_mean': zcr_mean,
        'zcr_std': zcr_std,
        'zcr_min': zcr_min,
        'zcr_max': zcr_max,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'rms_min': rms_min,
        'rms_max': rms_max,
        'sp_mean': sp_mean,
        'sp_std': sp_std,
        'sp_min': sp_min,
        'sp_max': sp_max,
        'sb_mean': sb_mean,
        'sb_std': sb_std,
        'sb_min': sb_min,
        'sb_max': sb_max,
        'srf_mean': srf_mean,
        'srf_std': srf_std,
        'srf_min': srf_min,
        'srf_max': srf_max
    }

    # features = [
    #     mfcc_mean, mfcc_std, mfcc_min, mfcc_max,
    #     mfcc_delta_mean, mfcc_delta_std, mfcc_delta_min, mfcc_delta_max,
    #     chroma_mean, chroma_std,
    #     mel_spec_mean, mel_spec_std, mel_spec_max,
    #     zcr_mean, zcr_std, zcr_min, zcr_max,
    #     rms_mean, rms_std, rms_min, rms_max,
    #     sp_mean, sp_std, sp_min, sp_max,
    #     sb_mean, sb_std, sb_min, sb_max,
    #     srf_mean, srf_std, srf_min, srf_max
    # ]
    # features = np.concatenate(features).tolist()

    return features

def flatten_dict(d: dict):
    flat = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            for i in range(value.shape[0]):
                flat[f"{key}_{i}"] = value[i]
        else:
            flat[key] = value
    return flat

def process_all_wavs(wav_dir: str, csv_dir: str):
    df = pd.DataFrame()
    first = True
    for root, dirs, files in os.walk(wav_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                labels = _file_name_to_metadata(file)
                features = _wav_to_features(wav_path)
                flat_features = flatten_dict(features)
                if first:
                    headers = list(labels.keys()) + list(flat_features.keys())
                    df = pd.DataFrame(columns=headers)
                    first = False
                df.loc[len(df)] = list(labels.values()) + list(flat_features.values())

    df.to_csv(f"{csv_dir}features2.csv", index=False)

    return df

if __name__ == "__main__":
    wav_dir = "SER_Project\\data\\"
    csv_dir = "SER_Project\\"
    process_all_wavs(wav_dir, csv_dir)
