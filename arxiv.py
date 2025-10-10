import json
import random
import re

def loadArxiv():
    # 100k utts from Kaggle arXiv
    utts = []
    with open("arxiv.txt","r") as f:
        i=0
        for l in f:
            o = json.loads(l)
            # print(o.keys())
            # print(len(o['sections']))
            # print(len(o['section_names']))
            nutts = len(o['article_text'])
            for ui in range(nutts):
                s = o['article_text'][ui]
                if '@xcite' in s:
                    cites = [m.start() for m in re.finditer('@xcite', s)]
                    tgt = random.choice(cites)
                    ss = s[0:cites[0]]
                    for j in range(len(cites)-1):
                        if cites[j]==tgt:
                            ss+='@@CITATION'
                        else:
                            ss+="(Zhu, 2025)"
                        ss+=s[cites[j]+6:cites[j+1]]
                    if cites[-1]==tgt: ss+='@@CITATION'
                    ss+=s[cites[-1]+6:]
                    utts.append(ss)
            i+=1
    return utts


