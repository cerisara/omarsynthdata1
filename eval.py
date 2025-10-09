import torch
import torch.nn as nn

citation_dict={
        0:"BACKGROUND",
        1:"USES",
        2:"COMPARES OR CONTRASTS",
        3:"MOTIVATION",
        4:"CONTINUATION",
        5:"FUTURE"}

def prepval():
    from datasets import load_dataset
    from transformers import pipeline

    model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedder = pipeline("feature-extraction", model=model_name, trust_remote_code=True, device=-1)

    ds = load_dataset("hrithikpiyush/acl-arc")
    val = ds['validation']
    print("val",val)
    with open("aclarcval.lab","w") as g:
        for i in range(len(val)):
            l=val['intent'][i]
            g.write(str(l)+'\n')
    with open("aclarcval.txt","w") as g:
        for i in range(len(val)):
            l=val['cleaned_cite_text'][i]
            embeddings = embedder(l)
            g.write(str(embeddings)+'\n')
            g.flush()

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

# prepval()
es,labs = loadval()
nclass = max(labs)+1
dim = es[0].shape[0]
print("valdata",len(es),nclass,dim)

mlp = nn.Sequential(
    nn.Linear(dim, 256),
    nn.ReLU(),              # or nn.GELU() for Transformer-style
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, nclass)
)

with torch.no_grad():
    nok,ntot=0,0
    nokc = [0]*nclass
    ntotc = [0]*nclass
    for i in range(len(es)):
        y=mlp(es[i])
        cpred = torch.argmax(y)
        lab = labs[i]
        if cpred.item()==lab:
            nokc[lab]+=1
            nok+=1
        ntot += 1
        ntotc[lab]+=1
        print("REC",cpred.item(),lab)
    acc = float(nok)/float(ntot)
    print("ACC",acc,ntot)
    for ci in range(nclass):
        acc = float(nokc[ci])/float(ntotc[ci])
        print("ACL"+str(ci),acc,ntotc[ci])

