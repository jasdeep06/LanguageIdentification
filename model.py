from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np
from feature_extractor import get_feature_matrix,process_test_file


X=get_feature_matrix(1,50)
y=np.concatenate((np.ones((1,50)).astype("int32"),2*np.ones((1,50)).astype("int32")),axis=1).T
y=np.ravel(y)

X_test=process_test_file("sound_samples/test1.wav")

clf=SVC(kernel="rbf")
clf.fit(X,y)
#print(clf.predict(X_test))
joblib.dump(clf,"trained_model.pkl")

print(clf.predict(X_test.reshape(1,-1)))
