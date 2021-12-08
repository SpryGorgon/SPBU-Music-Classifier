from musicnet_utils import Cutter, MusicFeaturesExtractor, Loader, genres, rel_path_to_abs
import numpy as np
import pandas as pd
import os
import joblib
from copy import deepcopy
from functools import wraps
from warnings import filterwarnings
from IPython.display import display

def create_music_classifier(path='MusicClassifier.spgn'):
#     path = rel_path_to_abs(__file__, path)
    package = MusicClassifier()
    MusicClassifier.__module__="musicnet.MusicClassifier"
    joblib.dump(package, path, compress=True)

def get_music_classifier(path='MusicClassifier.spgn', n_jobs=-1, filter_warnings=True):
    n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
#     path = rel_path_to_abs(__file__, path)
    try:
        classifier = joblib.load(path)
    except FileNotFoundError:
        create_music_classifier()
        classifier = joblib.load(path)

    if(filter_warnings):
        filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
        filterwarnings("ignore", category=DeprecationWarning)
        filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    return classifier(n_jobs)

class _MusicClassifierContainer:
    def __init__(self):
        self.cutter=Cutter()
        self.extractor=MusicFeaturesExtractor()
        self.loader = Loader()
#         musicClassifierDir = rel_path_to_abs(__file__, 'MusicClassifier.rfc')
        musicClassifierDir = "../input/musicclassifier/MusicClassifier.rfc"
        rfc = joblib.load(musicClassifierDir)
        self.classifier = rfc
        
    def set_params(self, n_jobs):
        self.cutter.n_jobs=n_jobs
        self.extractor.n_jobs=n_jobs
        self.classifier.n_jobs=n_jobs
        self.loader.n_jobs=n_jobs

    def _predict_decorator(func):
        @wraps(func)
        def wrapped(self, X_features=None, X_audios=None, X_path=None, *args, **kwargs):
            if(X_features is None and X_audios is None and X_path is None):
                raise Exception("No X parameter was specified")
            if not X_features is None:
                return func(self, X_features, *args, **kwargs)
            elif not X_audios is None:
                return func(self, self.extractor.extract(X_audios), *args, **kwargs)
            else:
                return func(self, self.extractor.extract(self.loader.load_tracks(X_path)), *args, **kwargs)
        return wrapped

    @_predict_decorator
    def predict(self, X_features=None, X_audios=None, X_path=None):
        return self.classifier.predict(X_features)

    @_predict_decorator
    def predict_proba(self, X_features=None, X_audios=None, X_path=None):
        return self.classifier.predict_proba(X_features)

    @_predict_decorator
    def predict_proba_display(self, X_features=None, X_audios=None, X_path=None, names=None):
        if names is None:
            names = self.loader.names
        results = self.predict_proba(X_features)
        for i in range(X_features.shape[0]):
            results_dataframe = pd.DataFrame([results[i]], columns=genres)
            print(str(i+1) + "\t" + genres[np.argmax(results[i])] + "\t" + names[i])
            display(results_dataframe)



class MusicClassifier:
    def __init__(self):
        self.__container=_MusicClassifierContainer()
        
    def __call__(self, _n_jobs=-1):
        n_jobs=_n_jobs
        newobj=deepcopy(self.__container)
        newobj.set_params(n_jobs)
        return newobj
