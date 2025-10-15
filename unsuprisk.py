import math
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

iscuda = True

def getacc(scores, labs):
    sc = scores.detach().cpu().numpy()
    ys = labs.detach().cpu().numpy()
    assert len(sc)==len(ys)
    nok=0
    for i in range(len(sc)):
        # attention: les scores sont avant la sigmoid
        #     sigm = 1. / (1. + np.exp(-x))
        s,y=sc[i][0],ys[i]
        if s<0.5 and y==0.: nok+=1
        elif s>=0.5 and y==1.: nok+=1
    acc=float(nok)/float(len(sc))
    if acc<0.5: acc=1.-acc
    return acc

def plothisto(scores, prefix):
    hist, binedges = np.histogram(scores, bins=30)
    nsamp = len(scores)
    # binedges contient un elt de plus que hist
    centers = [(binedges[i]+binedges[i+1])/2. for i in range(len(binedges)-1)]
    for i in range(len(hist)): print(prefix+" %f %f" % (centers[i], hist[i]))

def binrisk(mu0, mu1, var0, var1, prior0):
    # we must have a single "sample" here !
    with torch.set_grad_enabled(True):
        sq2 = torch.tensor(math.sqrt(2.))
        if iscuda: sq2.cuda()
        sigma0 = torch.sqrt(var0)
        sigma1 = torch.sqrt(var1)
        nor0 = torch.distributions.normal.Normal(mu0,sigma0)
        mor0 = torch.exp(nor0.log_prob(-1.))
        nor1 = torch.distributions.normal.Normal(mu1,sigma1)
        mor1 = torch.exp(nor1.log_prob(1.))
        prior1 = 1.-prior0

        m = mu0+1.
        r = torch.mul(prior0/2.,m)
        mm = -mu0-1.
        nn = torch.mul(sq2,sigma0)
        mm = torch.div(mm,nn)
        mm = torch.erf(mm)
        mm = 1.-mm
        term1 = torch.mul(r,mm)
        r = term1

        term2 = torch.mul(prior0,var0)
        term2 = torch.mul(term2,mor0)
        r = r+term2

        m3 = 1.-mu1
        term3 = torch.mul(prior1/2.,m3)
        nn3 = torch.mul(sq2,sigma1)
        mm3 = torch.div(m3,nn3)
        mm3 = 1. + torch.erf(mm3)
        term3 = torch.mul(term3,mm3)
        r = r+term3

        term4 = torch.mul(prior1,var1)
        term4 = torch.mul(term4,mor1)
        r = r+term4
        return r

# version du loss sans GMM explicite
# aucun parametre !
class UnsupRisk(nn.Module):
    def __init__(self, prior0=0.5):
        super(UnsupRisk,self).__init__()
        # le prior est suppose connu et fixe
        self.p0 = torch.tensor(prior0, requires_grad=False)
        if iscuda: self.p0.cuda()
        self.logpriors = torch.tensor([np.log(prior0), np.log(1.-prior0)], requires_grad=False)
        self.numcall=0

    def plothisto(self, x):
        plothisto(x.cpu().detach().numpy(),"HISTO"+str(self.numcall))
        self.numcall+=1

    def forward(self, x):
        # normalize, otherwise an easy way to decrese loss is to scale scores towards infinity
        xmin, xmax = torch.min(x), torch.max(x)
        xmax = xmax - xmin
        xn = (x-xmin)/xmax

        xx,_ = torch.sort(xn.view(-1))
        # on veut que les xx les plus petits diminuent avec le loss, donc il suffit que le loss soit lineaire(xx)
        n = self.p0 * x.size(0)
        n = n.int()
        lbas = torch.mean(xx[0:n])
        # inversement, on veut que les xx les plus grands augmentent, donc lineaire(-xx)
        lhaut = torch.mean(xx[n:])
        if False:
            # version basique: on veut separer au plus les 2 gaussiennes
            # donne a peu pres les memes res que la version avec le risk exact, mais en bcp plus rapide
            l = lbas - lhaut
        else:
            # version normale: on utilise la formule du risk
            sigbas = torch.std(xx[0:n])
            sighaut = torch.std(xx[n:])
            l = binrisk(lbas,lhaut,sigbas,sighaut,self.p0)

        # TODO ajouter un sigmoid plutot que de normaliser ?
        # x is a batch of scores/scalars (batch,1)
        # self.plothisto(x)
        # print("lossunsup %f %f" % (l.detach().cpu().numpy(),acc))

        return l

# version modifie du risque unsup pour prendre en compte des X annotes: on veut separer les distribs des 2 classes, plutot que de separer par le quantile
class UnsupLabeledRisk(nn.Module):
    def __init__(self, prior0=0.5):
        super(UnsupLabeledRisk,self).__init__()
        # le prior est suppose connu et fixe
        self.p0 = torch.tensor(prior0, requires_grad=False)
        self.logpriors = torch.tensor([np.log(prior0), np.log(1.-prior0)], requires_grad=False)
        self.numcall=0

    def forward(self, x, y):
        # normalize, otherwise an easy way to decrese loss is to scale scores towards infinity
        xmin, xmax = torch.min(x), torch.max(x)
        xmax = xmax - xmin
        xn = (x-xmin)/xmax
        xn = xn.view(-1)

        # TODO: must be another way to better do that
        lbas  = torch.tensor([0.])
        lhaut = torch.tensor([0.])
        if iscuda:
            lbas.cuda()
            lhaut.cuda()
        n0,n1=0,0
        for i in range(xn.size(0)):
            if y[i]==0:
                lbas = lbas + xn[i]
                n0+=1
            else:
                lhaut = lhaut + xn[i]
                n1+=1
        lbas = lbas / float(n0)
        lhaut = lhaut / float(n1)
        if lbas>lhaut:
            tmp=lbas
            lbas=lhaut
            lhaut=tmp
        l = lbas - lhaut

        return l

# =======================================================================

# main class:
# - takes as input a tensor X that comes from a previous DNN
# - adds on top of X a set of linear layers, each one associated with a given prior
# - adds at the very end a set of unsuprisk loss functions for each prior, equally combined into a single loss function to optimize
# - IMPORTANT: works ONLY with batches of large enough size (typically >=128)
class UnsupClassif(nn.Module):

    def __init__(self,nins):
        super(UnsupClassif, self).__init__()
        self.mlp10 = nn.Linear(nins,1)
        self.mlp20 = nn.Linear(nins,1)
        self.mlp30 = nn.Linear(nins,1)
        self.mlp40 = nn.Linear(nins,1)
        self.mlp50 = nn.Linear(nins,1)
        self.loss10 = UnsupRisk(0.1)
        self.loss20 = UnsupRisk(0.2)
        self.loss30 = UnsupRisk(0.3)
        self.loss40 = UnsupRisk(0.4)
        self.loss50 = UnsupRisk(0.5)

    def forward(self,oneXbatch):
        y10 = self.mlp10(oneXbatch)
        y20 = self.mlp20(oneXbatch)
        y30 = self.mlp30(oneXbatch)
        y40 = self.mlp40(oneXbatch)
        y50 = self.mlp50(oneXbatch)
        # y10 of size (batch, 1)
        l10 = self.loss10(y10)
        l20 = self.loss20(y20)
        l30 = self.loss30(y30)
        l40 = self.loss40(y40)
        l50 = self.loss50(y50)
        losses = torch.stack([l10,l20,l30,l40,l50])
        loss = torch.mean(losses)
        return loss

