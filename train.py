
import torch
import torch.nn as nn
import random

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
        es.append(torch.tensor(e))
        if len(es)>10200: break

with open("shuffledds6.csv","r") as f:
    for i,l in enumerate(f):
        if i>=len(es): break
        lab = int(l[0])
        es[i] = (lab,es[i])

# on garde toujours le meme test set
testset = es[:200]
es = es[200:]

nsamp = len(es)
dim = es[0][1].shape[0]
nclass = 1+max([x[0] for x in es])
print("data",nsamp,dim,nclass)

mlp = nn.Sequential(
    nn.Linear(dim, 256),
    nn.ReLU(),              # or nn.GELU() for Transformer-style
    nn.Linear(256, 256),
    nn.ReLU(),              
    nn.Linear(256, nclass)
)

labs = [torch.LongTensor([x]) for x in range(nclass)]
lossf = nn.CrossEntropyLoss()
opt = torch.optim.Adam(mlp.parameters(),lr=0.00002)
for ep in range(1000):
    random.shuffle(es)
    xi=0
    for lab,x in es:
        opt.zero_grad()
        y=mlp(x)
        loss = lossf(y.view(1,-1),labs[lab].view(1,))
        print("LOSS",loss.item(),ep,xi)
        loss.backward()
        opt.step()
        xi += 1

    with torch.no_grad():
        nok,ntot=0,0
        nokc = [0]*nclass
        ntotc = [0]*nclass
        for lab,x in testset:
            y=mlp(x)
            cpred = torch.argmax(y)
            if cpred.item()==lab:
                nokc[lab]+=1
                nok+=1            
            ntot += 1
            ntotc[lab]+=1
            print("REC",ep,cpred.item(),lab)
        acc = float(nok)/float(ntot)
        print("ACC",acc,ntot)
        for ci in range(nclass):
            acc = float(nokc[ci])/float(ntotc[ci])
            print("ACL"+str(ci),acc,ntotc[ci])

