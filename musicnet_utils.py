import numpy as np
import pandas as pd
import os
import librosa
from multiprocessing import Pool

SEED = int(1e9+7e7+17)
np.random.seed(SEED)
default_labels = ['blues']*100 + ['classical']*100 + ['country']*100 + ['disco']*100 + ['hiphop']*100 + ['jazz']*99 + ['metal']*100 + ['pop']*100 + ['reggae']*100 + ['rock']*100
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
features = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean',
       'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var',
       'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
       'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
       'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
       'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
       'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
       'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
       'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
       'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
       'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
       'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
       'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']
musicnet_path = 'musicnet'

def rel_path_to_abs(file, rel_path):
    return os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(file))), rel_path)

# def normalize(track):
#     m,s = track.mean(), track.std()
#     return (track-m)/s

def normalize(track):
    mx,mn = max(track), min(track)
    m = (mx+mn)/2
    return (track-m)/(mx-m)

class Loader:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs>0 else os.cpu_count()
        self.names = None
    
    def load_tracks(self, path, n_jobs=-1, verbose=0, get_names=False, normalize=True):
        n_jobs = self.n_jobs if n_jobs==-1 else n_jobs
        dataset, names = self.__scan_folder__(path, n_jobs, verbose, True, normalize=normalize)
        dataset = np.array(dataset)
        self.names = names
        return (dataset,names) if get_names else dataset

    def __scan_folder__(self, path, n_jobs, verbose, get_names, normalize, blacklist=['jazz.00054.wav']):
        tracks_paths = []
        tmp_paths = []
        tracks = []
        tracks_names = []
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                tmp_paths.append(os.path.join(dirpath, filename))
        tmp_paths.sort()
        for music_path in tmp_paths:
            filename = os.path.split(music_path)[1]
            if filename in blacklist:
                continue
            tracks_names.append(filename)
            tracks_paths.append(music_path)
            if verbose==1:
                print(filename)
        with Pool(n_jobs) as p:
            tracks = p.starmap(self.__load_track__, [(track, verbose, normalize) for track in tracks_paths])
        return (tracks, tracks_names) if get_names else tracks

    def __load_track__(self, path, verbose, _normalize):
        X,sr = librosa.load(path)
        if _normalize:
            X = normalize(X)
        if verbose==2:
            print(os.path.split(path)[1])
        return X



class Cutter:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs>0 else os.cpu_count()
    
    def cut_dataset(self, dataset, durations, sr=22050, n_jobs=-1, default_labels=None, normalize=True):
        n_jobs = self.n_jobs if n_jobs==-1 else n_jobs
        new_dataset = []
        labels = []
        self.normalize = normalize
        for duration in durations:
            if not default_labels:
                new_dataset.extend(self.cut_album_in_pieces(dataset, duration, sr, n_jobs))
            else:
                new_data = self.cut_album_in_pieces(dataset, duration, sr, n_jobs, default_labels)
                new_dataset.extend(new_data[0])
                labels.extend(new_data[1])

        new_dataset = np.array(new_dataset)
        return new_dataset if not default_labels else (new_dataset, labels)

    def cut_album_in_pieces(self, dataset, duration, sr=22050, n_jobs=-1, default_labels=None):
        n_jobs = self.n_jobs if n_jobs==-1 else n_jobs
        subtracks = []
        labels = []
        album = dataset.copy()
        if len(album[0].shape)==0:
            album = album.reshape((1,-1))
        with Pool(n_jobs) as p:
            if not default_labels:
                new_data = p.starmap(self.cut_track_in_pieces, [(track, duration, sr) for track in album])
            else:
                new_data = p.starmap(self.cut_track_in_pieces, [(album[i], duration, sr, default_labels[i]) for i in range(len(album))])
        for new_data_sample in new_data:
            subtracks.extend(new_data_sample[0])
            if not default_labels is None:
                labels.extend([new_data_sample[1]]*len(new_data_sample[0]))

        return subtracks if not default_labels else (subtracks, labels)

    def cut_track_in_pieces(self, track, duration, sr=22050, label=None):
        subtracks = []
        if duration == 0:
            raise Exception("Duration must be non-zero")
        if duration < 0:
            n_pieces = int((-1)/duration)
            duration = track.shape[0]/sr/n_pieces
        else:
            n_pieces = int((track.shape[0]/sr)//duration)
        for i in range(n_pieces):
            _start, _stop = int(i*duration*sr), int((i+1)*duration*sr)
            if self.normalize:
                subtracks.append(normalize(track[_start:_stop]))
            else:
                subtracks.append(track[_start:_stop])

        return (subtracks, label)



class MusicFeaturesExtractor:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs if n_jobs>0 else os.cpu_count()
        self.columns = features
    
    def extract(self, dataset, n_jobs=-1):
        ###################### mono sound ##########################
        n_jobs = self.n_jobs if n_jobs==-1 else n_jobs
        if dataset.shape[0]==1:
            return pd.DataFrame([self.__extract__(dataset[0])], columns=self.columns)
        elif len(dataset[0].shape)==0:
            return pd.DataFrame([self.__extract__(dataset)], columns=self.columns)
        else:
            with Pool(n_jobs) as p:
                self.data_features = p.map(self.__extract__, dataset)#, chunksize=4)
            data_features = pd.DataFrame(self.data_features, columns=self.columns)
            return data_features
        
    def extract_batch(self, data, batch_size=None):
        X = None
        if batch_size is None:
            batch_size=max(1, data.shape[0]//100)
        for start_index in range(0, data.shape[0], batch_size):
            _start, _stop = start_index, start_index+batch_size
            tmpX = self.extract(data[_start:_stop])
            if X is None:
                X = tmpX
            else:
                X = pd.concat((X,tmpX), axis=0, ignore_index=True)
        return X
        
        
    def __extract__(self, audio):
        features = []
        tmp = np.abs(librosa.feature.chroma_stft(audio))
        features.append(tmp.mean())
        features.append(tmp.var())
        tmp = librosa.feature.rms(audio)
        features.append(tmp.mean())
        features.append(tmp.var())
        tmp = librosa.feature.spectral_centroid(audio)
        features.append(tmp.mean())
        features.append(tmp.var())
        tmp = librosa.feature.spectral_bandwidth(audio)
        features.append(tmp.mean())
        features.append(tmp.var())
        tmp = librosa.feature.spectral_rolloff(audio)
        features.append(tmp.mean())
        features.append(tmp.var())
        tmp = librosa.feature.zero_crossing_rate(audio)
        features.append(tmp.mean())
        features.append(tmp.var())
        tmp = librosa.effects.harmonic(audio)
        features.append(tmp.mean())
        features.append(tmp.var())
        tmp = librosa.effects.percussive(audio)
        features.append(tmp.mean())
        features.append(tmp.var())
        tmp = librosa.beat.tempo(audio)[0]
        features.append(tmp)
        tmp = librosa.feature.mfcc(audio)
        for i in range(20):
            features.append(tmp[i].mean())
            features.append(tmp[i].var())
            
        return features
