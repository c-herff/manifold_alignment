import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

### Config
nComponents = 8
test_size = 0.5
nRandsCCA = 100

def align(source, target):
    # As CCA is temporally independent, we can maximize alignment with this
    l = np.min((source.shape[0],target.shape[0]))
    # Sort targets
    sIdx = np.argsort(source)
    tIdx = np.argsort(target)
    # Sample to same length
    sIdx = sIdx[np.linspace(0,source.shape[0]-1,l).astype(int)]
    tIdx = tIdx[np.linspace(0,target.shape[0]-1,l).astype(int)]
    return sIdx, tIdx

def pre_proc(feat,lbl):
    X_train, X_test, y_train, y_test = train_test_split(feat, lbl, test_size=test_size, shuffle=False)
    #Z-Score
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    #PCA
    pca = PCA(n_components=nComponents)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train,X_test,y_train,y_test


if __name__ == '__main__':
    est = LinearRegression()
    cca = CCA(n_components=nComponents)
    # Load datasets
    grasp = np.load('ds/kh011_feat.npy')
    graspTarget = (np.load('ds/kh011_lbl.npy')>0).astype(int)
    speech = np.load('ds/sub-03_feat.npy')
    speechTarget = np.mean(np.load('ds/sub-03_spec.npy'),axis=1)
    housing, housingTarget = fetch_california_housing(return_X_y=True)
    monkey = np.load('ds/monkey_feat.npy')
    monkeyTarget = np.load('ds/monkey_target.npy')
    #diabetes, diabetesTarget = load_diabetes(return_X_y=True)
    datasets = [(grasp,graspTarget),(speech,speechTarget),(housing,housingTarget),(monkey,monkeyTarget)] #(diabetes,diabetesTarget),(speech,speechTarget),(speech2,speech2Target),
    # Initialize result arrays
    reconstruction = np.zeros((len(datasets),len(datasets)))
    randReconstruction = np.zeros((len(datasets),len(datasets),nRandsCCA))
    corrs = np.zeros((len(datasets),len(datasets),nComponents))
    corrsRandBaseline = np.zeros((len(datasets),len(datasets),nComponents))
    for sNr, source in enumerate(datasets):
        (source_x, source_y) = source
        Xsource_train, Xsource_test, ysource_train, ysource_test = pre_proc(source_x,source_y)
        # Establish reconstruction baseline on own test data set
        est.fit(Xsource_train,ysource_train)
        prediction = est.predict(Xsource_test)
        reconstruction[sNr,sNr] = np.corrcoef(prediction,ysource_test)[0,1]
        print('Within score of %d is %f' % (sNr,reconstruction[sNr,sNr]))
        # Establish alignment baseline on own data between train and test
        idxS, idxT = align(ysource_train,ysource_test)
        jointSpaceTrain, jointSpaceTest = cca.fit_transform(Xsource_train[idxS],Xsource_test[idxT])
        # Calculate correlations in aligned space
        for dim in range(jointSpaceTrain.shape[1]):
            corrs[sNr,sNr,dim] = np.corrcoef(jointSpaceTrain[:,dim],jointSpaceTest[:,dim])[0,1]
        for tNr, target in enumerate(datasets):
            if sNr == tNr:
                continue
            (target_x, target_y) = target
            # Pre-process and split into training and testing
            Xtarget_train, Xtarget_test, ytarget_train, ytarget_test = pre_proc(target_x,target_y)
            # Train classifier on target
            est.fit(Xtarget_train,ytarget_train)
            # Align data
            idxS, idxT = align(ysource_train,ytarget_train)
            # Fit CCA and transform the training data sets
            jointSpaceSource, jointSpaceTarget = cca.fit_transform(Xsource_train[idxS],Xtarget_train[idxT])
            # Calculate correlations in aligned space
            for dim in range(jointSpaceSource.shape[1]):
                corrs[sNr,tNr,dim] = np.corrcoef(jointSpaceSource[:,dim],jointSpaceTarget[:,dim])[0,1]
            # Transform test data to target space
            dat = cca.predict(Xsource_test)
            prediction = est.predict(dat)
            reconstruction[sNr,tNr] = np.corrcoef(ysource_test,prediction)[0,1]
            print('From %s to %s gives %f' % (sNr,tNr,reconstruction[sNr,tNr]))
            # Non-matching baselines       
            alpha = 0.5
            randCorrs = np.zeros((nComponents,nRandsCCA))
            for rIt in range(nRandsCCA):
                jointSpaceSource, jointSpaceTarget = cca.fit_transform(Xsource_train[idxS],Xtarget_train[np.random.permutation(idxT)])
                # Correlations
                for dim in range(jointSpaceSource.shape[1]):
                    randCorrs[dim,rIt] = np.corrcoef(jointSpaceSource[:,dim],jointSpaceTarget[:,dim])[0,1]
                # Decoding
                dat = cca.predict(Xsource_test)
                prediction = est.predict(dat)
                randReconstruction[sNr,tNr,rIt] = np.corrcoef(ysource_test,prediction)[0,1]
            corrsRandBaseline[sNr,tNr,:] = np.sort(randCorrs,axis=1)[:,-int(nRandsCCA*alpha)]
            print('Max correlation is %f compared to max random correlation of %f' % (np.max(corrs[sNr,tNr,:]),np.max(randCorrs)))

    np.save('ds/reconstruction.npy',reconstruction)
    np.save('ds/randReconstruction.npy',randReconstruction)
    np.save('ds/corrs.npy',corrs)
    np.save('ds/corrsRandBaseline.npy',corrsRandBaseline)