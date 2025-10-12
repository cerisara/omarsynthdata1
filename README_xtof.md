
-----------
Sun Oct 12 07:26:49 CEST 2025

J'ai 2 idees en cours:
- ajouter arxiv corpus via unsuprisk
- augmenter ACLARC-train en demande au LLM de garder les memes phrases mais de changer le topic
  pour que le MLP apprenne que la classif doit etre indep du topic

mais je teste en cours simplement un classif binaire sur classe 4, train sur ACLARC-train:
macroF1(class 0) = 79%
macroF1(class 4) avec 50% de NOK = 58% (on a le moins de data ici, il faudrait early stop)
macroF1(class 4) avec 75% de NKO = 59%
macroF1(class 4) avec tous les NOK = 49%
macroF1(class 4) avec NKO=3NOK = 59%
macroF1(class 4) avec NKO=4NOK = 59%
note: pour la classe 0, la loss ne converge pas ! est-ce donc qu'il faut regulariser ?!
je teste en ajoutant du bruit aux embeddings 
(stddev=0.01): F1(4) = 58%
(stddev=0.1): F1(4) = 57%
(stddev=0.001): F1(4) = 59.5%
(stddev=0.1 only to NOK): F1(4) = 63%
conclusion: regularizing by adding noise to rare classes helps generalize a bit

J'ajoute ma loss unsup:

F1(4) = 61% ca doit etre un peu mieux en vrai, mais attention a la variabilite et convergence
ATTENTION: sur seulement la classe 4, la macro-F1 ne vaut rien, il faut regarder F1 de la classe 0 !!!
avec unsuprisk:
w=0

w=1
clF1s {0: 0.22222222222222224, 1: 0.9680365296803652}
w=6
clF1s {0: 0.25, 1: 0.9727272727272728}



-----------
sam. 11 oct. 2025 07:17:47 CEST

dans tSNE, les points synth vs. ACLARC sont peut-etre separes parce que le LLM
genere des topics tres differents, alors que ACLARC est focus sur un seul topic.
Donc cela serait l'inverse de ce que je croyais (i.e. LLM n'est pas assez variable);
si c'est le cas, alors TODO: augmenter le ACLARC train dataset en demandant au LLM de
preserver la forme de la phrase, la syntaxe, mais de changer le domaine scientifique,
afin de train le MLP pour qu'il apprenne que la classif doit etre indep du topic.

-----------
Fri Oct 10 21:51:51 CEST 2025

je teste une baseline en 1 vs all pour preparer le unsuprisk.

avec cl=4, un classifieur binaire trained sur un corpus balanced 50% donne F1=25% sur ACLARC
avec cl=0, on est a F1=80%

ceci correspond a peu pres a ce qu'on avait avec un N-class classifier; c'est normal.
TODO: ajouter le corpus unsup

-----------
Thu Oct  9 11:53:24 AM CEST 2025

Methode: synth data -> embedding -> MLP -> classif en 6 citation types

QR1: est-ce que les synth data permettent de differencier les classes entre elles ?
    ==> Oui: en decoupant les synth data en train/test, on a ACC=87% avec 10k utts de train
        Il est surement possible d'avoir plus avec plus de data, MLP plus gros, etc. mais ce n'est pas urgent
    ==> conclusion: les synth data representent bien des classes differentes

QR2: est-ce que les synth data "couvrent" bien les classes du corpus ?
    ==> Non: ACC sur corpus tres faible

XP: etudier la "couverture" des embeddings, entre le corpus et les synth data
==> tSNE montre que les embeddings synth et du corpus sont separes
mais c'est peut-etre normal: car ils ont une semantique differente. Mais en fait non, car
ils devraient etre repartis dans tout l'espace ! Or, ils sont groupes et separes des autres.
==> conclusion: les synth data representent mal les data du corpus

TODO: embed ACL-train sur ADAE
TODO: QR = est-ce que train sur une distrib synth tres differente de la distrib ACL-ARC limite les perfs ?

TODO:
SFT la generation de data synth pour matcher la distribution de ACL ARC

"You are ... generate ..." ───> LLM ───> y1...yT ───> E3 ───> Loss <─── E3 <─── ACLARC data

obj = SFT le LLM par soft-prompt ou LoRA très léger
La Loss peut etre de minimiser la dist entre les Y generes et les Z les plus proches de ACLARC et de maximiser la variance des Y
note:
- MSE(mean(Y)-mean(Z)) + MSE(var(Y)-var(Z)) pas bon car on peut avoir tous les Z a gauche et tous les Y a droite
- min_i MSE(Y-Z_i) pas bon car on peut avoir tous les Y sur un seul Z_i, aucune variance ni couverture !
- donc plutot min_i MSE(Y-Z_i) - var(Y)


