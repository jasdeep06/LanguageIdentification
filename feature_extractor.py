from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np


def get_feature_matrix(start_obs,end_obs):
    base_files=["english","hindi"]

    vectors=[]
    for file in base_files:
        for i in range(start_obs,end_obs+1):
            (rate,sig)=wav.read("sound_samples/"+file+str(i)+".wav")
            feature_vector=get_feature_vector(rate,sig)
            feature_vector.reshape(104,1)
            vectors.append(feature_vector)

    return(np.concatenate(vectors).reshape(2*(end_obs-start_obs+1),104))






def get_feature_vector(rate,sig):
    mfcc_feat=mfcc(sig,rate).T
    mean_mfcc_feat = np.mean(mfcc_feat, 1)
    cov_mfcc_feat = np.cov(mfcc_feat)
    uppe_cov_mcc_feat = cov_mfcc_feat[np.triu_indices(13)]
    feature_vector=np.concatenate((mean_mfcc_feat,uppe_cov_mcc_feat))
    return feature_vector



"""""
def get_test_matrix(start_obs,end_obs):
    base_files=["test"]
    vectors = []
    for file in base_files:
        for i in range(start_obs, end_obs + 1):
            (rate, sig) = wav.read("sound_samples/" + file + str(i) + ".wav")
            feature_vector = get_feature_vector(rate, sig)
            feature_vector.reshape(104, 1)
            vectors.append(feature_vector)

    return (np.concatenate(vectors).reshape(1 * (end_obs - start_obs + 1), 104))
"""""
def process_test_file(file):

    (rate, sig) = wav.read(file)
    feature_vector=get_feature_vector(rate,sig)
    return feature_vector
