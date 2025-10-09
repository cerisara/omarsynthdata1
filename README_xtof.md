
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


