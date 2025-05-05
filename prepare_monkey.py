import scipy.io as sio
import numpy as np

def process(d,down=2,func=np.mean):
    # As I'm no expert on monkey data, this is brutally basic
    s = d.shape[0]
    m = s%down
    d = np.reshape(d[:s-m,:],(int((d.shape[0]-m)/down),d.shape[1],down))
    d = func(d,axis=2)
    return d


if __name__ == '__main__':
    dat = sio.loadmat('ds/Mihili_CO_VR_2014-03-03.mat')['trial_data'][0]
    numTrials = dat.shape[0]

    # Pick random 8*5 trials (if only I knew how to identify the target)
    np.random.seed(1337)
    numPicks = 120
    downFactor = 3
    picks = np.random.choice(numTrials,numPicks,replace=False)
    data = process(dat[picks[0]][20],down=downFactor,func=np.sum)
    target = process(dat[picks[0]][16],down=downFactor,func=np.mean)


    for i in range(1,numPicks):
        #Could do some pre-processing here!
        d = dat[picks[i]][20]
        t = dat[picks[i]][16]
        d = process(d,down=downFactor,func=np.sum)
        t = process(t,down=downFactor,func=np.mean)
        data = np.concatenate([data,d])
        target =np.concatenate([target,t])

    print(data.shape,target.shape)
    np.save('ds/monkey_feat.npy',data)
    np.save('ds/monkey_target.npy',target[:,0])
    
    # Test decoding to get a feeling whether this is working
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_california_housing, load_diabetes
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_decomposition import CCA 
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
    from sklearn.preprocessing import StandardScaler

    
    X_train, X_test, y_train, y_test = train_test_split(data, target[:,0], test_size=0.5, shuffle=False)
    #Z-Score
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    #PCA
    pca = PCA(n_components=10)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    est=LinearRegression()
    est.fit(X_train,y_train)
    prediction = est.predict(X_test)
    reconstruction = np.corrcoef(prediction,y_test)[0,1]
    print('Within score is %f' % (reconstruction))

    numrands=1000
    rs = []
    for i in range(numrands):
        rs.append(np.corrcoef(y_test,np.random.permutation(y_test))[0,1])
    print(np.sort(rs)[-50])

    