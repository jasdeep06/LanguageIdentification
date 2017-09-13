from pydub import AudioSegment


def generate_audio_files(main_file):

    t_old=0
    t_new=0

    for i in range(1,51):
        t_old=t_new
        t_new=t_old+(10*1000)
        new_audio=AudioSegment.from_wav('sound_samples/'+main_file+".wav")
        new_audio=new_audio[t_old:t_new]
        new_audio.export('sound_samples/'+main_file+ str(i)+'.wav',format='wav')

#generate_audio_files("english")
new_audio=AudioSegment.from_wav("sound_samples/big_test.wav")
new_audio=new_audio[20*1000:30*1000]
new_audio.export("bigg_test.wav",format='wav')