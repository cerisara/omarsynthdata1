
import torch
import torch.nn as nn
import random
from collections import defaultdict
import arxiv

dev = "cuda"

class Metric:
    def __init__(self):
        # Track true positives, false positives, false negatives per class
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.labels = set()

    def update(self, ypred: int, ugold: int):
        self.labels.update([ypred, ugold])  # Track all observed labels
        if ypred == ugold:
            self.tp[ypred] += 1
        else:
            self.fp[ypred] += 1
            self.fn[ugold] += 1

    def getF1(self):
        f1_scores = {}
        for label in sorted(self.labels):
            tp = self.tp[label]
            fp = self.fp[label]
            fn = self.fn[label]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores[label] = f1
        return f1_scores

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
            es.append(torch.tensor(e).to(dev))
    print("traindata",len(es),' '.join([str(sum([1 for l in labs if l==x])) for x in range(max(labs))]))
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
            es.append(torch.tensor(e).to(dev))
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
            es.append(torch.tensor(e).to(dev))
            if len(es)>1200: break

    labs=[]
    with open("shuffledds6.csv","r") as f:
        for i,l in enumerate(f):
            if i>=len(es): break
            lab = int(l[0])
            labs.append(lab)
    return es,labs

# unsuputts = arxiv.loadArxiv()
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
nclass = 1+max(labs)
print("data",nsamp,dim,nclass)

mlp = nn.Sequential(
    nn.Linear(dim, 256),
    nn.ReLU(),              # or nn.GELU() for Transformer-style
    nn.Linear(256, 2)
).to(dev)

def sft(mlp, cl):
    # 1 class vs. all
    labOK = torch.LongTensor([0]).to(dev)
    labKO = torch.LongTensor([1]).to(dev)
    lossf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(mlp.parameters(),lr=0.001)
    for ep in range(100):
        random.shuffle(tridx)
        # simulate a single batch with all data in
        opt.zero_grad()
        for xi in range(len(tridx)):
            if labs[tridx[xi]]==cl: lab=labOK
            else: lab=labKO
            x   = es[tridx[xi]]
            y=mlp(x)
            loss = lossf(y.view(1,-1),lab.view(1,))
            print("LOSS",loss.item(),ep,xi)
            loss.backward()
        opt.step()

        with torch.no_grad():
            metric = Metric()
            for i in range(len(acles)):
                y=mlp(acles[i])
                cpred = torch.argmax(y)
                lab = acllabs[i]
                metric.update(cpred.item(),lab)
                print("VALREC",cpred.item(),lab)
            f1s = metric.getF1()
            print("clF1s",f1s)
            macrof1 = sum(f1s.values())/len(f1s)
            print("macroF1",macrof1)

sft(mlp,4)

