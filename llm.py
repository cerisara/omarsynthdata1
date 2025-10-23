# train en LLM binaire pour reco la classe 5 FUTURE TODO: add unsup risk

import torch
import torch.nn as nn
import random
from collections import defaultdict
import unsuprisk
import sys
import arxiv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda"
dounsup = True

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

ds = load_dataset("hrithikpiyush/acl-arc")
# ds = load_dataset("ialvarenga/acl-arc-revised")
# ds = load_dataset("kejian/ACL-ARC")
txtcol = 'text'
txtcol = 'cleaned_cite_text'
co={}
for xi in range(len(ds['train'])):
    s = ds['train'][txtcol][xi]
    l = ds['train']['intent'][xi]
    if l in co: co[l]+=1
    else: co[l]=1
for xi in range(len(ds['validation'])):
    s = ds['validation'][txtcol][xi]
    l = ds['validation']['intent'][xi]
    if l in co: co[l]+=1
    else: co[l]=1
for xi in range(len(ds['test'])):
    s = ds['test'][txtcol][xi]
    l = ds['test']['intent'][xi]
    if l in co: co[l]+=1
    else: co[l]=1
for i in co.keys():
    print(i,co[i])

if dounsup:
    arxs = arxiv.loadArxiv()
    arxes = [s for s in arxs if len(s)<1500]
    arxs = None

def sft(cl, wp0=1.):
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(dev)
    toker = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokyes = toker(" Yes")['input_ids'][0]
    tokno  = toker(" No")['input_ids'][0]
    print("tokyesno",tokyes,tokno)
    # 1 class vs. all
    labOK = torch.LongTensor([0]).to(dev)
    labKO = torch.LongTensor([1]).to(dev)
    lossf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),lr=0.001)
    okidx = []
    koidx = []
    for xi in range(len(ds['train'])):
        if ds['train'][xi]['intent']==cl: okidx.append(xi)
        else: koidx.append(xi) 
    prior0 = float(len(okidx))/float(len(ds['train']))
    print("PRIOR0",prior0)
    assert prior0>0 and prior0<0.5 # assume tgt = classe minoritaire

    allf1s = []
    teallf1s = []
    for ep in range(400):
        # nko = len(okidx) # gives balanced corpus
        nko = len(okidx)*3
        random.shuffle(koidx)
        tridx = okidx+koidx[:nko]
        random.shuffle(tridx)

        # partie SFT de la loss
        lo = 0.
        m0OK, m0KO = 0., 0.
        nOK, nKO = 0, 0
        supscores = []
        for xi in range(len(tridx)):
            opt.zero_grad()
            if ds['train'][tridx[xi]]['intent']==cl: lab=labOK
            else: lab=labKO
            s = ds['train']['cleaned_cite_text'][tridx[xi]]
            utt = f"{s}. The previous sentence is extracted from a scientific paper. Is @@CITATION used to motivate a potential future work, yes or no? Just answer with a single word, yes or no. Answer:"
            x = toker(utt, return_tensors="pt").to(dev)
            print("supcontextlen",x['input_ids'].shape,len(utt))
            # if lab==0:
            #     # la classe minoritaire a peu de samples, je la regularise ici avec du bruit gaussien
            #     noise = torch.randn_like(x) * 0.1
            #     x=x+noise
            y=model(**x)
            yy = y.logits[0,-1,[tokyes,tokno]]
            print("debugsup",yy)
            sc0 = torch.nn.functional.softmax(yy, dim=-1).view(-1,)[0].item()
            supscores.append((sc0,lab.item()))
            print("SCORESUP",sc0,lab.item(),ep,xi)
            if lab==0:
                nOK += 1
                m0OK += sc0
            else:
                nKO += 1
                m0KO += sc0
            loss = lossf(yy.view(1,-1),lab.view(1,))
            lo += loss.item()
            print("sampleLOSS",loss.item(),ep,xi,"batch",len(tridx))
            loss.backward()
            opt.step()
        lo /= float(len(tridx))
        print("SFTLOSS",lo,ep)
 
        # fix le prior: dans unsuprisk, la classe 0 est la minoritaire
        # j'ai calcule la moyenne m(0,OK) des output 0 du MLP pour la classe 0 et pour la classe 1 m(0,KO)
        # si m(0,OK)>m(0,KO), alors il faut inverser les outputs du MLP
        m0OK /= float(nOK)
        m0KO /= float(nKO)
        if False and m0OK>m0KO:
            # non, car je sais que tokYES a l'idx 0 !!
            out0idx = 1
        else: out0idx = 0
        print("OUT0",out0idx,wp0,m0OK,m0KO,nOK,nKO) 

        if dounsup:
            random.shuffle(arxes)
            ss = arxes[:1024] # pick random batch of 1024 samples: on a 3% ==> 30 samples positifs
            allscores = []
            with torch.no_grad():
                for s in ss:
                    utt = f"{s}. The previous sentence is extracted from a scientific paper. Is @@CITATION used to motivate a potential future work, yes or no? Just answer with a single word, yes or no. Answer:"
                    x = toker(utt, return_tensors="pt").to(dev)
                    y=model(**x)
                    yy = y.logits[0,-1,[tokyes,tokno]]
                    print("debugunsup",yy)
                    sc0 = torch.nn.functional.softmax(yy, dim=-1).view(-1,)[0].item()
                    print("SCOREUNSUP",sc0,ep)
                    allscores.append(sc0)
            allscores.sort()
            xmin, xmax = allscores[0], allscores[-1]
            n = int(prior0 * len(allscores))
            thr = allscores[-n]
            # pourquoi on a des probas(0) ramassees dans un minuscule intervalle, qui change de position a chaque epoch ?
            # c'est comme si notre LLM output une proba independante de la phrase en input !
            # alors que le score sur la partie labeled span tout l'intervalle 0-1
            # ==> car sur la partie SUP, il train en meme temps: le LLM change ses poids, ce qui change l'output:
            # la variabilite du SUP est due au chgt des poids, pas au changement des phrases !!
            # donc le train modifie surtout le biais (= prior) du YES vs NO: TODO: ce n'est pas un bon training !!!
            print("NORMSCORES",xmin, xmax, thr,n,len(allscores))
            if True:
                # check
                cok,cko = [0,0],[0,0]
                for s,lab in supscores:
                    if lab==0:
                        if s<thr: cok[0] += 1
                        else: cok[1] += 1
                    else:
                        if s<thr: cko[0] += 1
                        else: cko[1] += 1
                print("THRSUP",cok[0],cok[1],cko[0],cko[1])
            lo=0.
            for s in ss:
                opt.zero_grad()
                utt = f"{s}. The previous sentence is extracted from a scientific paper. Is @@CITATION used to motivate a potential future work, yes or no? Just answer with a single word, yes or no. Answer:"
                x = toker(utt, return_tensors="pt").to(dev)
                print("ucontextlen",x['input_ids'].shape,len(utt))
                y=model(**x)
                yy = y.logits[0,-1,[tokyes,tokno]]
                sc0 = torch.nn.functional.softmax(yy, dim=-1).view(-1,)[0]
                if sc0>thr: loss = -sc0 * wp0
                else: loss = sc0 * wp0
                lo += loss.item()
                print("sampleunsuploss",loss.item(),torch.cuda.mem_get_info()[0])
                loss.backward()
                opt.step()
            lo /= float(len(ss))
            print("UNSUPLOSS",lo,ep)
            # risk = unsuprisk.UnsupRisk(prior0)
            # uloss = risk(allscores)
            # uloss = uloss * wp0
            # uloss.backward()
            # opt.step()

        # evaluation
        with torch.no_grad():
            metric = Metric()
            cnn=[0,0]
            for i in range(len(ds['validation'])):
                s = ds['validation']['cleaned_cite_text'][i]
                utt = f"{s}. The previous sentence is extracted from a scientific paper. Is @@CITATION used to motivate a potential future work, yes or no? Just answer with a single word, yes or no. Answer:"
                x = toker(utt, return_tensors="pt").to(dev)
                y=model(**x)
                yy = y.logits[0,-1,[tokyes,tokno]]
                cpred = torch.argmax(yy)
                lab = ds['validation'][i]['intent']
                if lab==cl: lab=0
                else: lab=1
                cnn[cpred.item()]+=1
                metric.update(cpred.item(),lab)
                print("VALREC",cpred.item(),lab)
            print("NN",cnn[0],cnn[1])
            f1s = metric.getF1()
            print("clF1s",f1s)
            allf1s.append(f1s)
 
        # test
        with torch.no_grad():
            metric = Metric()
            cnn=[0,0]
            for i in range(len(ds['test'])):
                s = ds['test']['cleaned_cite_text'][i]
                utt = f"{s}. The previous sentence is extracted from a scientific paper. Is @@CITATION used to motivate a potential future work, yes or no? Just answer with a single word, yes or no. Answer:"
                x = toker(utt, return_tensors="pt").to(dev)
                y=model(**x)
                yy = y.logits[0,-1,[tokyes,tokno]]
                cpred = torch.argmax(yy)
                lab = ds['test'][i]['intent']
                if lab==cl: lab=0
                else: lab=1
                cnn[cpred.item()]+=1
                metric.update(cpred.item(),lab)
                print("TESTREC",cpred.item(),lab)
            print("TENN",cnn[0],cnn[1])
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
    print("TESTF1",teallf1s[bestep][0], teallf1s[-1][0], bestep)
    return allf1s, teallf1s[bestep][0]

if __name__ == "__main__":
    if len(sys.argv)>1: w0 = float(sys.argv[1])
    else: w0 = 0.01

    # attention: il y a du code qui a deja run ci-dessus !

    smeanf1, smaxf1, slastf1, tef1 = 0.,0.,0.,0.
    nruns = 10
    for run in range(nruns):
        allf1s, teallf1 = sft(5, w0)
        smeanf1 += sum([allf1s[-i][0] for i in range(50)])/50.
        smaxf1  += max([x[0] for x in allf1s])
        slastf1 += allf1s[-1][0]
        tef1 += teallf1
    tef1 /= float(nruns)
    smeanf1 /= float(nruns)
    smaxf1 /= float(nruns)
    slastf1 /= float(nruns)
    print("ALLRUNSF1",tef1,smeanf1,smaxf1,slastf1)

