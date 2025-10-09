import random

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

def loadtrain(nsamp=320):
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
    print("traindata",' '.join([str(sum([1 for l in labs if l==x])) for x in range(max(labs))]))
    return es,labs

def sample(es,labs):
    i = random.randint(0,len(es)-1)
    return es[i],labs[i]

def dist(a,b):
    d=sum([a[i]*b[i] for i in range(len(a))])
    return d

tre,trl = loadtrain()
d=0.
est = []
for i in range(100):
    e,l = sample(tre,trl)
    est.append(e)
    be,bl = sample(tre,trl)
    d += dist(e,be)
print("Dsynth",d/100.)

tee,tel = loadval()
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

