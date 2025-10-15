import torch
import torch.nn as nn
import random 
from collections import defaultdict
import sys

dev = "cuda"

def embedall():
    from datasets import load_dataset
    from transformers import pipeline

    model_name = "Qwen/Qwen3-14B"
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedder = pipeline("feature-extraction", model=model_name, trust_remote_code=True, device=dev, return_tensors=True)

    ds = load_dataset("hrithikpiyush/acl-arc")
    res = []
    for dsnom in ('train','validation','test'):
        print("data",dsnom)
        val = ds[dsnom]
        labs = []
        for i in range(len(val)):
            l=val['intent'][i]
            labs.append(l)
        embeds = []
        for i in range(len(val)):
            l=val['cleaned_cite_text'][i]
            embeddings = embedder(l).to(dev).to(torch.float32)
            embeds.append(embeddings[0,-1,:])
        res.append((labs,embeds))
    return res
 
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

corpus = embedall()
nsamp = len(corpus[0][0])
dim = corpus[0][1][0].shape[0]
nclass = 1+max(corpus[0][0])
print("data",nsamp,dim,nclass)

mlp = nn.Sequential(
    nn.Linear(dim, 500),
    nn.ReLU(),              # or nn.GELU() for Transformer-style
    nn.Linear(500, 6)
).to(dev)

def sft(mlp):
    vlabs = [torch.LongTensor([i]).to(dev) for i in range(6)]
    lossf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(mlp.parameters(),lr=0.001)
    tridx = [i for i in range(nsamp)]

    allf1s = []
    teallf1s = []
    for ep in range(100):
        random.shuffle(tridx)

        lo = 0.
        for xi in range(len(tridx)):
            opt.zero_grad()
            lab = vlabs[corpus[0][0][tridx[xi]]]
            x   = corpus[0][1][tridx[xi]]
            y=mlp(x)
            loss = lossf(y.view(1,-1),lab.view(1,))
            lo += loss.item()
            print("sampleLOSS",loss.item(),ep,xi)
            loss.backward()
            opt.step()
        lo /= float(len(tridx))
        print("SFTLOSS",lo,ep)

        # evaluation
        with torch.no_grad():
            metric = Metric()
            for i in range(len(corpus[1][0])):
                y=mlp(corpus[1][1][i])
                cpred = torch.argmax(y)
                lab = corpus[1][0][i]
                metric.update(cpred.item(),lab)
                print("VALREC",cpred.item(),lab)
            f1s = metric.getF1()
            print("clF1s",f1s)
            macrof1 = sum(f1s.values())/len(f1s)
            print("macroF1",macrof1)
            allf1s.append(f1s)
 
        # test
        with torch.no_grad():
            metric = Metric()
            for i in range(len(corpus[2][0])):
                y=mlp(corpus[2][1][i])
                cpred = torch.argmax(y)
                lab = corpus[2][0][i]
                metric.update(cpred.item(),lab)
                print("TESTREC",cpred.item(),lab)
            f1s = metric.getF1()
            print("TEclF1s",f1s)
            teallf1s.append(f1s)
 
    # early stopping
    maxf1  = max([x[0] for x in allf1s])
    meanf1 = sum([allf1s[-i][0] for i in range(50)])/50.
    for ep in range(len(allf1s)):
        if allf1s[ep][0]==maxf1:
            bestep = ep
            break
    print("FINALF1",allf1s[bestep][0], allf1s[-1][0], meanf1, maxf1)
    macrof = sum(teallf1s[bestep].values())/float(len(teallf1s[bestep]))
    print("TESTF1",macrof, teallf1s[bestep], bestep)
    return allf1s, macrof

if __name__ == "__main__":
    if len(sys.argv)>1: w0 = float(sys.argv[1])
    else: w0 = 0.

    # attention: il y a du code qui a deja run ci-dessus !

    smeanf1, smaxf1, slastf1, tef1 = 0.,0.,0.,0.
    nruns = 100
    for run in range(nruns):
        _, teallf1 = sft(mlp)
        tef1 += teallf1
        tef1 /= float(run+1)
        print("ALLRUNSF1",tef1,run)
 
