
import torch
import torch.nn as nn
import random
 
def loadtrain():
    with open("aclarctrain.lab","r") as f: labs=[int(s) for s in f]
    es=[]
    with open("aclarctrain.txt","r") as f:
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
            # es.append(e)
            es.append(torch.tensor(e))
    print("traindata",' '.join([str(sum([1 for l in labs if l==x])) for x in range(max(labs))]))
    return es,labs
 
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
            es.append(torch.tensor(e))
    return es,labs
 
def loadsynth():
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
            if len(es)>1200: break

    labs=[]
    with open("shuffledds6.csv","r") as f:
        for i,l in enumerate(f):
            if i>=len(es): break
            lab = int(l[0])
            labs.append(lab)
    return es,labs

acles,acllabs = loadval()
# es,labs = loadsynth()
es,labs = loadtrain()
assert len(es)==len(labs)
tridx = [i for i in range(len(es))]

# on garde toujours la meme partie du synth data comme test set
# testset = es[:200]
# es = es[200:]

nsamp = len(es)
dim = es[0].shape[0]
nclass = 1+max(labs])
print("data",nsamp,dim,nclass)

mlp = nn.Sequential(
    nn.Linear(dim, 256),
    nn.ReLU(),              # or nn.GELU() for Transformer-style
    nn.Linear(256, nclass)
)

labs = [torch.LongTensor([x]) for x in range(nclass)]
lossf = nn.CrossEntropyLoss()
opt = torch.optim.Adam(mlp.parameters(),lr=0.00002)
for ep in range(100):
    random.shuffle(tridx)
    for xi in range(len(tridx)):
        lab = torch.LongTensor(labs[tridx[xi]])
        x   = es[tridx[xi]]
        opt.zero_grad()
        y=mlp(x)
        loss = lossf(y.view(1,-1),lab.view(1,))
        print("LOSS",loss.item(),ep,xi)
        loss.backward()
        opt.step()

    if False:
        # only use that to test on part of synth data
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
 
    with torch.no_grad():
        nok,ntot=0,0
        nokc = [0]*nclass
        ntotc = [0]*nclass
        for i in range(len(acles)):
            y=mlp(acles[i])
            cpred = torch.argmax(y)
            lab = acllabs[i]
            if cpred.item()==lab:
                nokc[lab]+=1
                nok+=1
            ntot += 1
            ntotc[lab]+=1
            print("VALREC",cpred.item(),lab)
        acc = float(nok)/float(ntot)
        print("VALACC",acc,ntot)
        for ci in range(nclass):
            acc = float(nokc[ci])/float(ntotc[ci])
            print("VALACL"+str(ci),acc,ntotc[ci])
     
