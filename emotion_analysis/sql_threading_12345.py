import time
from emotion_analysis.sql_12345_threading import Emotion_analysis

if __name__ == '__main__':
    print("Start at:{}".format(time.ctime()))
    thresh = Emotion_analysis(n_jobs=1)
    thresh1 = thresh.threading_process()
    print("Done at: {}".format(time.ctime()))
