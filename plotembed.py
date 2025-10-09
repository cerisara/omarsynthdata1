import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def loadval():
    with open("aclarcval.lab","r") as f: labs=[int(s) for s in f]
    es=[]
    with open("aclarcval.txt","r") as f:
        for li, l in enumerate(f):
            s=l.replace('[','').strip().split(']')
            vs=[]
            for i in range(len(s)):
                ss=s[i].strip().split(',')
                v=[float(x) for x in ss if len(x)>0]
                if len(v)>0: vs.append(v)
            e = [0.]*len(vs[0])
            for i in range(len(vs)):
                for j in range(len(e)):
                    e[j] += vs[i][j]
            for j in range(len(e)): e[j] /= float(len(vs))
            es.append(e)
    print("evaldata",' '.join([str(sum([1 for l in labs if l==x])) for x in range(max(labs))]))
    return es,labs

def loadsynth(nsamp=320):
    es = []
    with open("ds6.txt","r") as f:
        for li, l in enumerate(f):
            s=l.replace('[','').strip().split(']')
            vs=[]
            for i in range(len(s)):
                ss=s[i].strip().split(',')
                v=[float(x) for x in ss if len(x)>0]
                if len(v)>0: vs.append(v)
            e = [0.]*len(vs[0])
            for i in range(len(vs)):
                for j in range(len(e)):
                    e[j] += vs[i][j]
            for j in range(len(e)): e[j] /= float(len(vs))
            es.append(e)
            if len(es)>nsamp: break

    labs = []
    with open("shuffledds6.csv","r") as f:
        for i,l in enumerate(f):
            if i>=len(es): break
            labs.append(int(l[0]))
    print("synthdata",' '.join([str(sum([1 for l in labs if l==x])) for x in range(max(labs))]))
    return es,labs

def sample(es,labs):
    i = random.randint(0,len(es)-1)
    return es[i],labs[i]

def dist(a,b):
    d=sum([a[i]*b[i] for i in range(len(a))])
    return d

def distances():
    d=0.
    est = []
    for i in range(100):
        e,l = sample(tre,trl)
        est.append(e)
        be,bl = sample(tre,trl)
        d += dist(e,be)
    print("Dsynth",d/100.)

    d=0.
    esc=[]
    for i in range(100):
        e,l = sample(tee,tel)
        esc.append(e)
        be,bl = sample(tre,trl)
        d += dist(e,be)
    print("Dcorp",d/100.)

    d=0.
    for i in range(len(est)):
        d += dist(est[i],esc[i])
    print("Dboth",d/100.)

tre,trl = loadsynth()
tee,tel = loadval()

# - vectors: list of lists or numpy array, shape (n_samples, embedding_dim)
# - labels: list of 0s and 1s (or any two class labels)
cl=5
vectors = np.array([tre[i] for i in range(len(tre)) if trl[i]==cl] + [tee[i] for i in range(len(tee)) if tel[i]==cl])
labels = np.array([0 for i in range(len(tre)) if trl[i]==cl] + [1 for i in range(len(tee)) if tel[i]==cl])

# Compute t-SNE (reduce to 2D)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
vectors_2d = tsne.fit_transform(vectors)

# Plot
plt.figure(figsize=(8,6))
for label in np.unique(labels):
    idx = labels == label
    plt.scatter(vectors_2d[idx, 0], vectors_2d[idx, 1], label=f"Class {label}", alpha=0.7, s=50)

plt.title("t-SNE visualization of embeddings")
plt.legend()
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

