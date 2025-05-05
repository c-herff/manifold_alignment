import numpy as np
import matplotlib.pyplot as plt



# Load Results
reconstruction =np.load('ds/reconstruction.npy')
randReconstruction = np.load('ds/randReconstruction.npy')
corrs = np.load('ds/corrs.npy')
corrsRandBaseline = np.load('ds/corrsRandBaseline.npy')

n_ds = corrs.shape[0]
n_components = corrs.shape[2]

# Visualize Correlation results
rs = np.zeros(n_components)
stds = np.zeros(n_components)
selfRs = np.zeros(n_components)
selfstds = np.zeros(n_components)
randRs = np.zeros(n_components)
randStds = np.zeros(n_components)


for i in range(n_components):
    rs[i] = np.mean(corrs[:,:,i][np.logical_not(np.eye(n_ds))])
    stds[i] = np.std(corrs[:,:,i][np.logical_not(np.eye(n_ds))])
    selfRs[i] = np.mean(corrs[:,:,i][(np.eye(n_ds)>0)])
    selfstds[i] = np.std(corrs[:,:,i][(np.eye(n_ds)>0)])
    randRs[i] = np.mean(corrsRandBaseline[:,:,i][np.logical_not(np.eye(n_ds))])
    randStds[i] = np.std(corrsRandBaseline[:,:,i][np.logical_not(np.eye(n_ds))])

fig,ax = plt.subplots(1,2)
x = range(1,len(rs)+1)
ax[0].plot(x,rs,c='red')
ax[0].fill_between(x,rs-stds,rs+stds,alpha=0.5,color='red')

ax[0].plot(x,selfRs,c='lightgray')
ax[0].fill_between(x,selfRs-selfstds,selfRs+selfstds,alpha=0.5,color='lightgray')

ax[0].plot(x,randRs,c='orange')
ax[0].fill_between(x,randRs-randStds,randRs+randStds,alpha=0.5,color='orange')
ax[0].set_xlabel('Neural Mode')
ax[0].set_ylabel('Correlation')
#Title
#ax.set_title('b',fontsize=20,fontweight="bold")
##Make pretty
plt.setp(ax[0].spines.values(), linewidth=2)
ax[0].set_xticks([1,2,4,6,8])
ax[0].set_yticks([0,0.2,0.4,0.6])
#The ticks
ax[0].xaxis.set_tick_params(width=2)
ax[0].yaxis.set_tick_params(width=2)
ax[0].xaxis.label.set_fontsize(20)
ax[0].yaxis.label.set_fontsize(20)
c = [a.set_fontsize(20) for a in ax[0].get_yticklabels()]
c = [a.set_fontsize(20) for a in ax[0].get_xticklabels()]
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)


# Visualize Decoder results
#fig, ax = plt.subplots()
rands = np.zeros(n_ds)
aligned = np.zeros(n_ds)
for p in range(n_ds):
    idx = np.eye(n_ds)[p,:]
    rand = np.mean(randReconstruction[p,np.logical_not(idx)])
    res = np.mean(reconstruction[p,np.logical_not(idx)])
    self = reconstruction[p,p]
    #ax.plot([0,1,2],[rand,res,self],c='gray',alpha=0.3)
ax[1].errorbar([0], [np.mean(randReconstruction[np.logical_not(np.eye(n_ds))])], yerr=[np.std(randReconstruction[np.logical_not(np.eye(n_ds))])], fmt='o',ms=10)
ax[1].errorbar([1], [np.mean(reconstruction[np.logical_not(np.eye(n_ds))])], yerr=[np.std(reconstruction[np.logical_not(np.eye(n_ds))])], fmt='o',ms=10)
ax[1].errorbar([2], [np.mean(reconstruction[(np.eye(n_ds)>0)])], yerr=[np.std(reconstruction[(np.eye(n_ds)>0)])], fmt='o',ms=10)
ax[1].set_ylabel('Reconstruction correlation')
##Make pretty
plt.setp(ax[1].spines.values(), linewidth=2)
ax[1].set_xticks([0,1,2],['Unaligned','Aligned','Within'])
ax[1].set_yticks([0,0.2,0.4,0.6])
#The ticks
ax[1].xaxis.set_tick_params(width=2)
ax[1].yaxis.set_tick_params(width=2)
ax[1].xaxis.label.set_fontsize(20)
ax[1].yaxis.label.set_fontsize(20)
c = [a.set_fontsize(20) for a in ax[1].get_yticklabels()]
c = [a.set_fontsize(20) for a in ax[1].get_xticklabels()]

ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
plt.tight_layout()
plt.show()