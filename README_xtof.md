
-----------
dim. 26 oct. 2025 07:34:46 CET

- au cours du training, on se retrouve avec 2 pics de scores a 0 et 1: ce n'est pas bon, car pour ameliorer, il
  faut faire 'sauter' des samples d'un pic à l'autre, cela devient de la RO
- le score = Le LLM retourne une proba qui peut etre vue comme un silver label (le LLM est-il calibrated ?)
  donc mon unsup training peut etre vu comme du self-training: il ne faudrait donc pas train tous les samples
  mais seulement ceux dont on est moyennement sur, i.e., dont la proba est a mi-chemin entre 0.5 et 0/1.

-----------
sam. 25 oct. 2025 07:18:43 CEST

- le testF1 atteint un max (50% à 65%) entre 100 et 150 epochs sur chaque run, puis il diminue; mais tres variable !
- c'est vrai aussi du devF1, mais il est plus haut (80%) et il reste bcp plus longtemps au max
- les supscores sont bien separes dans les 2 classes car ils sont revus a chaque epoch ==> overfitting sur supscores
- est-ce que les unsupscores sont bien separes en 2 classes ? ==> OUI
- quel est le posterior de la repartition ?
    ==> unsup class 0: 9% ==> trop grand !
    ==> sup class 0: 25% ce qui est parfait

==> tout ceci indique clairement un overfitting
==> une repartition qui classe bien les labeled samples dans les classes 0 et 1, mais par overfitting et ne respecte
    pas le prior !!

la loss sup est instables mais parfois a 1E-6, malgre une repartition stable de 25% !
C'est normal !!!!!!!! car on train en oversampling 3x samples positifs

TODO:
- ajouter loss pour forcer posterior
- diminuer le LR ou ajouter regul

-----------
24 oct 2025

- OK, je suis quasi sur qu'il n'y a plus de bugs dans le code unsup.
- il converge vers une repartition des labeled future/no-future quasi parfaite par rapport au seuil du prior0
- mais il ne generalise pas: le F1 sur le test reste < 30%: est-ce le arxiv dataset qui est trop different du TEST ?
- note: les scores-unsup sont tres tres proches de 0 ou de 1, alors qu'ils ne l'etaient pas au debut

TODO:
- est-ce qu'on overfit sur ARXIV ? tester les hyper-parms ? comment mesurer l'overfitting sur ARXIV, avec un dev ?
- tester avec le dataset de TEST a la place de ARXIV ?
- ou mieux: des qu'on a un ex. de test, on recup 1024 ex. de ARXIV proches

22 oct 2025

XP: train un Qwen3-0.6B en mode generation-clasif binaire:
"is the @@CITATION in the previous sentence future, yes or no?"
train en GD (batch=tout le train set) sur 400 epochs, early stopping

average sur 9 runs:
- les perfs sur dev:  F1=85.7%
- les perfs sur test: F1=32.4%

J'essaye de run avec le unsuprisk, mais ca plante OOM, meme avec
seulement 128 batch (qui est le minimum possible, avec seulement 3 samples positifs en moyenne)
==> TODO: version simplifiee en 2 passes: 1) calcul du threshold; 2) unsup train per sample et non plus per batch

J'ai aussi plot les distribs des scores: les scores sup et unsup suivent la meme distrib,
mais il n'y a aucune separation class 0/1; et les scores des lab=0 dans sup sont distribs partout;
==> TODO: faire qqs epochs de SFT avant d'injecter le unsuprisk

-----------
Mon Oct 13 09:37:09 AM CEST 2025

unsuprisk avec arxiv: impact du weights de la unsup loss:

==> trunsup0..log <==
ALLRUNSF1 0.28772650151133083 0.23325247419247408 0.3356519591519589 0.246258297258297

==> trunsup10..log <==
ALLRUNSF1 0.3308894597834222 0.23564610559375265 0.33056998556998535 0.23535281385281362

==> trunsup100..log <==
ALLRUNSF1 0.3179313157602628 0.2451130014662418 0.3080112665112662 0.24789033189033172

==> trunsup200..log <==
ALLRUNSF1 0.3252608799405673 0.24391421620431808 0.30975646575646537 0.25345743145743116

==> trunsup500..log <==
ALLRUNSF1 0.27487615566408113 0.23590208511750266 0.31729191295535314 0.23481125161388292

==> trunsup1000..log <==
ALLRUNSF1 0.1811569507105002 0.23056675309955388 0.32935716442140545 0.22583885558885541

Je relance, et j'observe toujours une variabilité importante: pourquoi, alors que je run 100 XPs supposément indep ?
est-ce que +/-3% est norma sur 100 runs ?
==> il n'y a pas de patterns temporels evidents, juste une grand var de chaque run, surement due
au faible nb de classe 4 dans le corpus de test, et au early-stopping: je pourrais peut-etre diminuer la variabilite
sur ce dernier point en detectant les mauvaises convergence du train, ou en moyennant les MLP sur plusieurs epoch
(Stochastic weight averaging), ou en empechant de stop trop tot.

ALLRUNSF1 0.3016899113957935 0.23125508772273487 0.3301796536796533 0.2390295815295813
xtof@adae:~/git/omar_synthdata$ tail -1 trunsup10..log 
ALLRUNSF1 0.30059382124226713 0.23505968423733145 0.31623088023088 0.23823809523809505
xtof@adae:~/git/omar_synthdata$ tail -1 trunsup100..log 
ALLRUNSF1 0.3210954836357309 0.24438256730197913 0.3065733155733153 0.25303318903318883
xtof@adae:~/git/omar_synthdata$ tail -1 trunsup200..log 
ALLRUNSF1 0.3459759767732445 0.24380815458730562 0.31014382839382804 0.24590476190476174


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
clF1s {0: 0.28571428571428575, 1: 0.9773755656108598}
w=1
clF1s {0: 0.22222222222222224, 1: 0.9680365296803652}
w=6
clF1s {0: 0.25, 1: 0.9727272727272728}
w=10
clF1s {0: 0.25, 1: 0.9727272727272728}

bugfix: je gardais les 2x MLP outputs

w=10
clF1s {0: 0.25, 1: 0.9727272727272728}
w=100
clF1s {0: 0.33333333333333337, 1: 0.9819819819819819}
clF1s {0: 0.22222222222222224, 1: 0.9680365296803652}

je rerun 10x: ci-dessous, moyenné sur 10 runs, le avg sur 50 dernieres epoch, max sur toutes epochs, last F1:
w=100
ALLRUNSF1 0.266803219003219 0.3285714285714286 0.2551587301587302
w=0
ALLRUNSF1 0.2307539711750238 0.3545454545454546 0.26352092352092354

j'ajoute early stopping

w=0
ALLRUNSF1 0.2887084226403112 0.22772683982683986 0.3431818181818182 0.21358585858585855
w=10
ALLRUNSF1 0.37434173669467785 0.256487105051811 0.39989898989898987 0.2546031746031746
w=100
ALLRUNSF1 0.37503968253968256 0.2799884526820697 0.41715728715728717 0.275
ALLRUNSF1 0.4196581196581198 0.2708586306166952 0.3238095238095239 0.2964285714285715
==> il y a encore bcp de variabilite TODO: tester avec 100 runs


TODO: test other LR, batchsize>1024, sans gaussian noise



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


