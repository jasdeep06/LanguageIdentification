from flask import Flask,render_template,request
from flask_uploads import UploadSet,configure_uploads,AUDIO
from sklearn.externals import joblib
from feature_extractor import process_test_file

app=Flask(__name__)

sounds=UploadSet('sounds',AUDIO)
app.config['UPLOADED_SOUNDS_DEST']='static/test_files/'

configure_uploads(app,sounds)

@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method=="POST" and 'sound' in request.files:
        filename=sounds.save(request.files['sound'])
        clf=joblib.load("trained_model.pkl")
        feature_vector=process_test_file("static/test_files/"+filename)
        prediction=clf.predict(feature_vector.reshape(1,-1))
        if prediction[0]==1:
            return "English"
        if prediction[0]==2:
            return "Hindi"

    return render_template('upload.html')


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))

    print("Starting app on port %d" % port)

    app.run(debug=False, port=port, host='0.0.0.0')
