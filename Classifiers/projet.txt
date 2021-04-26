from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from utils import AbstractClassifier
import utils
from scipy.stats import chi2_contingency
import numpy as np
 

def getPrior(train):
    """cette fonction retourne un dictionnaire contenant 3 clés 'estimation'= probabilité a priori de la classe train, 'min5pourcent'et 'max5pourcent' qui sont les bornes de l'intervalle de confiance à 95% pour l'estimation de cette probabilité  """
    nom=0
    denom=0
    variance=0
    res={"estimation":0,"min5pourcent":0,"max5pourcent":0}
    for d in train["target"]:
        nom+=d
        denom+=1
    #on calcule la moyenne
    moyenne=nom/denom
    res["estimation"]=moyenne
    #on calcule la variance
    for d in train["target"]:
        variance+=(d-moyenne)**2/denom
    #on utilise la variance pour calculer min5pourcent et max5pourcent
    res["min5pourcent"]=moyenne-1.96*sqrt(variance)/sqrt(denom)
    res["max5pourcent"]=moyenne+1.96*sqrt(variance)/sqrt(denom)
    return res

class APrioriClassifier(AbstractClassifier):
    def __init__(self):
        pass
    def estimClass(self,classe):
        """
        retourne une estimation très simple de la classe de chaque individu par la classe majoritaire
        quelle que soit la classe on a le même resultat calculé à partir de la moyenne de train
        """
        a=getPrior(pd.read_csv("train.csv"))
        if(a["estimation"]>0.5):
            return 1
        return 0
       
    def statsOnDF(self,df):
        """Retourne le nombre d'individus en fonction du target=0 ou 1, et de la classe prévue=0 ou 1 """
        res={"VP":0,"VN":0,"FP":0,"FN":0,"precision":0,"rappel":0}
        for t in df.itertuples():
            dic=t._asdict()
            estim=self.estimClass(dic)
            test=dic["target"]
            if(estim==0):
                if(test==0):
                    res["VN"]+=1
                elif(test==1):
                    res["FN"]+=1
            elif(estim==1):
                if(test==0):
                    res["FP"]+=1
                elif(test==1):
                    res["VP"]+=1
        res["precision"]=res["VP"]/(res["VP"]+res["FP"])
        res["rappel"]=res["VP"]/(res["VP"]+res["FN"])
        return res
    

def P2D_l(df,attr):
    """prend en argument le dataframe df ainsi qu'un attribu attr
    retourne 𝑃(𝑎𝑡𝑡𝑟=𝑎|𝑡𝑎𝑟𝑔𝑒𝑡=𝑡) a etant les différentes valeurs prises par l'attribu et t=0 ou 1 (valeurs prises par target """
    res={0:{},1:{}}
    proba={0:0,1:0}
    nb=0
    for t in df.itertuples():
        dic=t._asdict()
        tar=dic["target"]
        a=dic[attr]
        if (a not in res[tar]):
            res[tar].update({a:0})
        res[tar][a]+=1
        proba[tar]+=1
        nb+=1
    for i in [0,1]:
        for j in res[i]:
            res[i][j]=res[i][j]/proba[i]
    return res
   
def P2D_p(df,attr):
    """prend en argument le dataframe df ainsi qu'un attribu attr
    retourne 𝑃(𝑡𝑎𝑟𝑔𝑒𝑡=𝑡|𝑎𝑡𝑡𝑟=𝑎) a etant les différentes valeurs prises par l'attribu et t=0 ou 1 (valeurs prises par target """
    res={}
    proba={}
    ensemble_a=[]
    for t in df.itertuples():
        dic=t._asdict()
        a=dic[attr]
        tar=dic["target"]
        if(a not in proba):
            proba.update({a:0})
            ensemble_a.append(a)
        if(a not in res):
            res.update({a:{0:0,1:0}})
        res[a][tar]+=1
        proba[a]+=1
    for a in ensemble_a:
        for t in [0,1]:
            res[a][t]=res[a][t]/proba[a]
    return res

class ML2DClassifier(APrioriClassifier):
    """utilise une procédure de maximum de vraisemblance pour estimer la classe d'un individu: retourne la valeur de target de probabilité max """
    def __init__(self,d,c):
        self.dataf=d
        self.classe=c
    def estimClass(self,dico):
        t=dico[self.classe]
        P2DL=P2D_l(self.dataf,self.classe)
        proba_0=P2DL[0][t]
        proba_1=P2DL[1][t]
        if(proba_1>proba_0):
            return 1
        return 0
   
class MAP2DClassifier(APrioriClassifier):
    """utilise une procédure de maximum à postériori pour estimer la classe d'un individu: retourne la valeur de target de probabilité max """
    def __init__(self,d,c):
        self.dataf=d
        self.classe=c
       
    def estimClass(self,dico):
        t=dico[self.classe]
        P2DP=P2D_p(self.dataf,self.classe)
        proba_0=P2DP[t][0]
        proba_1=P2DP[t][1]
        if(proba_1>proba_0):
            return 1
        return 0
       
def nbParams(df,params=['target','age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']):
    """ prend en paramètre le dataframe, une liste das attirbuts contenant target et calcul la taille mémoire nécessaire pour représenter les tables de probabilité
    on suppose qu'un float est représenté sur 8octets
    """
    nb_float=0
    res={}
    for attr in params:
        res.update({attr:[]})
    for t in df.itertuples():
        dic=t._asdict()
        for attr in params:
            a=dic[attr]
            if(a not in res[attr]):
                res[attr].append(a)
    nb_float=1
    for i in res:
        nb_float*=len(res[i])
           
    print(len(res)," variables :", nb_float*8," octets")

def nbParamsIndep(df):
    """prend en paramètre le dataframe, et calcul la taille mémoire nécessaire pour représenter les tables de probabilité en supposant l'indépendance des variables
    on suppose qu'un float est représenté sur 8octets
    """
    nb_float=0
    res={}
    for t in df.itertuples():
        dic=t._asdict()
        for attr in dic.keys():
            if(attr!='Index'):
                a=dic[attr]
                if(attr not in res):
                    res.update({attr:[]})
                if(a not in res[attr]):
                    res[attr].append(a)
    for i in res:
        nb_float+=len(res[i])
           
    print(len(res)," variables :", nb_float*8," octets")
    
def drawNaiveBayes(df,col):
    """ prend en paramètre un dataframe, le nom d'une colonne du dataframe, retourne un graphe où le noeud col est parent de tous les attributs restant du dataframe """
    res=""
    tmp=0
    for k in df.keys():
        if(k!=col):
            if(tmp==0):
                res+=str(col)+"->"+str(k)
                tmp=1
            else:
                res+=";"+str(col)+"->"+str(k)
    return utils.drawGraph(res)

def nbParamsNaiveBayes(df,col,liste_param=['target','age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']):
    """prend en paramètre le dataframe, le nom d'une colonne, une liste d'attributs, et calcul la taille mémoire nécessaire pour représenter les tables de probabilité en utilisant l'hypothèse du Naive Bayes
    on suppose qu'un float est représenté sur 8octets
    """
    dico={col:[]}
    for param in liste_param:
        if(param!=col):
            dico.update({param:[]})
    for t in df.itertuples():
        dic=t._asdict()
        if(dic[col] not in dico[col]):
            dico[col].append(dic[col])
        for param in liste_param:
            if(param!=col):
                if(dic[param] not in dico[param]):
                    dico[param].append(dic[param])
    somme_taille_param=0
    for param in liste_param:
        if(param!=col):
            somme_taille_param+=len(dico[param])
    n=len(dico[col])        
    taille=n*(somme_taille_param+1)
    print(str(len(liste_param))+" variables : "+str(taille*8)+" octets")
       
class MLNaiveBayesClassifier(APrioriClassifier):
    """ utilise le maximum de vraisemblance (ML)pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes. """
    def __init__(self,df):
        self.df=df
               
        self.dico={0:{},1:{}}
        for k in self.df.keys():
            if(k!='target'):
                self.dico[0].update({k:{}})
                self.dico[1].update({k:{}})
            else:
                self.dico[0].update({k:0})
                self.dico[1].update({k:0})
        for t in self.df.itertuples():
            dic=t._asdict()
            for k in dic.keys():
                if(k!='Index'):
                    if(k=='target'):
                        self.dico[dic['target']][k]+=1
                    else:
                        if(dic[k] not in self.dico[dic['target']][k]):
                            self.dico[dic['target']][k].update({dic[k]:0})
                        self.dico[dic['target']][k][dic[k]]+=1
               
    def estimProbas(self,ligne):
        """  calcule la vraisemblance """
        res={0:1,1:1}
        for k in self.df.keys():
            for tar in [0,1]:
                if(k!='target' and k!='Index'):
                    if(ligne[k] not in self.dico[tar][k]):
                        res[tar]=0
                    elif(res!=0):
                        res[tar]*=self.dico[tar][k][ligne[k]]/self.dico[tar]['target']
        return res
          
    def estimClass(self,ligne):
        """  utilise estimProbas et retourne la classe avec la plus grande probabilité """
        res=self.estimProbas(ligne)
        if(res[0]>=res[1]):
            return 0
        return 1
    
class MAPNaiveBayesClassifier(APrioriClassifier):
    """ utilise le maximum a posteriori (MAP) pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes. """
    def __init__(self,df):
        self.df=df
       
        self.dico={0:{},1:{}}
        for k in self.df.keys():
            if(k!='target'):
                self.dico[0].update({k:{}})
                self.dico[1].update({k:{}})
            else:
                self.dico[0].update({k:0})
                self.dico[1].update({k:0})
        for t in self.df.itertuples():
            dic=t._asdict()
            for k in dic.keys():
                if(k!='Index'):
                    if(k=='target'):
                        self.dico[dic['target']][k]+=1
                    else:
                        if(dic[k] not in self.dico[dic['target']][k]):
                            self.dico[dic['target']][k].update({dic[k]:0})
                          
                        self.dico[dic['target']][k][dic[k]]+=1
       
    def estimProbas(self,ligne):
        """  calcule la vraisemblance """
        res={0:1,1:1}
        for k in self.df.keys():
            for tar in [0,1]:
                if(k!='target' and k!='Index'):
                    if(ligne[k] not in self.dico[tar][k]):
                        res[tar]=0
                    elif(res!=0):
                        res[tar]*=self.dico[tar][k][ligne[k]]/self.dico[tar]['target']
                      
        proba_target=self.dico[0]['target']/(self.dico[0]['target']+self.dico[1]['target'])
        res[0]=proba_target*res[0]
        proba_target=self.dico[1]['target']/(self.dico[0]['target']+self.dico[1]['target'])
        res[1]=proba_target*res[1]
        if(res[0]+res[1]!=0.):
            res[0]=res[0]/(res[0]+res[1])
            res[1]=1-res[0]
        return res
       
    def estimClass(self,ligne):
        """  utilise estimProbas et retourne la classe avec la plus grande probabilité """
        res=self.estimProbas(ligne)
        if(res[0]>=res[1]):
            return 0
        return 1

def isIndepFromTarget(df,attr,x):
    """ retoune si attr est indépendant de target au seuil de x% """
    dico_np={0:{},1:{}}
    for t in df.itertuples():
        dic=t._asdict()
        if(dic[attr] not in dico_np[dic['target']]):
            dico_np[dic['target']].update({dic[attr]:0})
        if(dic[attr] not in dico_np[1-dic['target']]):
            dico_np[1-dic['target']].update({dic[attr]:0})
           
        dico_np[dic['target']][dic[attr]]+=1
       
    liste_np=[[],[]]
    for i in [0,1]:
        for d in dico_np[i].keys():
            liste_np[i].append(dico_np[i][d])
    stat, p, dof, expected = chi2_contingency(np.array(liste_np))
    if(p>x):
        return True
    return False

class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    """ utilisent le maximum de vraisemblance (ML) pour estimer la classe d'un individu sur un modèle Naïve Bayes préalablement optimisé grâce à des tests d'indépendance au seuil de 𝑥% """
    def __init__(self,df,x):
        liste=[]
        for attr in df.keys():
            if(attr!='target' and attr!='Index'):
                if(isIndepFromTarget(df,attr,x)):
                    liste.append(attr)

        self.df=df.drop(labels=liste,axis=1)
        MLNaiveBayesClassifier.__init__(self,self.df)
      
    def draw(self):
        return drawNaiveBayes(self.df,"target")

class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    """ utilisent le maximum a posteriori (MAP) pour estimer la classe d'un individu sur un modèle Naïve Bayes préalablement optimisé grâce à des tests d'indépendance au seuil de 𝑥% """
    def __init__(self,df,x):
        liste=[]
        for attr in df.keys():
            if(attr!='target' and attr!='Index'):
                if(isIndepFromTarget(df,attr,x)):
                    liste.append(attr)

        self.df=df.drop(labels=liste,axis=1)
        MAPNaiveBayesClassifier.__init__(self,self.df)
      
    def draw(self):
        """  dessine le Naïve Bayes réduit utilisé.  """
        return drawNaiveBayes(self.df,"target")

def mapClassifiers(dic,df):
    """ prend en paramètre un dictionnaire {nom:instance de classifier} représente graphiquement les classifiers dans l'espace (𝑝𝑟é𝑐𝑖𝑠𝑖𝑜𝑛,𝑟𝑎𝑝𝑝𝑒𝑙) """
    x=[]
    y=[]
    liste_nom=[]
    for clas in dic.keys():
        if(clas!='Index'):
            liste_nom.append(clas)
            d=dic[clas].statsOnDF(df)
            x.append(d["precision"])
            y.append(d["rappel"])

    plt.plot(x,y,'x', color='red')
    plt.show()

def MutualInformation(df,x,y):
    """  calcule une distance entre la distribution des variables x et y et la distribution si ces 2 variables étaient indépendantes """
    res={}
    px={}
    py={}
    n=0
    for t in df.itertuples():
        dic=t._asdict()
        if(dic[x] not in res):
            res.update({dic[x]:{}})
        if(dic[y] not in res[dic[x]]):
            res[dic[x]].update({dic[y]:0})
        if(dic[y] not in py):
            py.update({dic[y]:0})
        if(dic[x] not in px):
            px.update({dic[x]:0})
        px[dic[x]]+=1
        py[dic[y]]+=1
        res[dic[x]][dic[y]]+=1
        n+=1
 
    somme=0
    for cle_x in res.keys():
        for cle_y in res[cle_x].keys():
            xy=res[cle_x][cle_y]/n
            x=px[cle_x]/n
            y=py[cle_y]/n
            somme+=xy*np.log2(xy/(x*y))
    return somme
           
       
def ConditionalMutualInformation(df,x,y,z):
    """  calcul des informations mutuelles conditionnelles voir formule donnée  """
    
    res={}
    pz={}
    pzy={}
    pzx={}
    n=0
    for t in df.itertuples():
        dic=t._asdict()
        n+=1
        if(dic[z] not in res):
            res.update({dic[z]:{}})
            pz.update({dic[z]:0})
            pzy.update({dic[z]:{}})
            pzx.update({dic[z]:{}})
        if(dic[y] not in res[dic[z]]):
            res[dic[z]].update({dic[y]:{}})
        if(dic[x] not in res[dic[z]][dic[y]]):
            res[dic[z]][dic[y]].update({dic[x]:0})
        if(dic[y] not in pzy[dic[z]]):
            pzy[dic[z]].update({dic[y]:0})
        if(dic[x] not in pzx[dic[z]]):
            pzx[dic[z]].update({dic[x]:0})
        pz[dic[z]]+=1
        pzx[dic[z]][dic[x]]+=1
        pzy[dic[z]][dic[y]]+=1
        res[dic[z]][dic[y]][dic[x]]+=1
       
    somme=0
    for cle_z in res.keys():
        for cle_y in res[cle_z].keys():
            for cle_x in res[cle_z][cle_y].keys():
                z=pz[cle_z]/n
                zx=pzx[cle_z][cle_x]/n
                zy=pzy[cle_z][cle_y]/n
                xyz=res[cle_z][cle_y][cle_x]/n
                somme+=xyz*np.log2((z*xyz)/(zx*zy))
    return somme
   
def MeanForSymetricWeights(a):
    """ calcule la moyenne des poids pour une matrice a symétrique de diagonale nulle. """
    i=1
    somme=0
    n=0
    for ligne in a:
        for j in range(i,len(ligne)):
            somme+=ligne[j]
            n+=1
        i+=1    
    return somme/n

def SimplifyConditionalMutualInformationMatrix(a):
    """  annule toutes les valeurs plus petites que la moyenne des poids pour une matrice a symétrique de diagonale nulle. """
    moyenne=MeanForSymetricWeights(a)
    for ligne in a:
        for i_case in range(len(ligne)):
            if ligne[i_case]<moyenne:
                ligne[i_case]=0.
    return a

#les trois fonctions suivantes servent à définir Kruskal
def aretePoidsMaximale(df,a):
    max=0
    attr1Max=""
    attr2Max=""
    i_max=0
    j_max=0
    key=df.keys()
    c=1
    for i in range(len(a)):
        attr1=key[i]
        for j in range(c,len(a[i])):
            attr2=key[j]
            if(a[i][j]>max):
                max=a[i][j]
                i_max=i
                j_max=j
                attr1Max=attr1
                attr2Max=attr2
        c+=1
    return [[i_max,j_max],(attr1Max,attr2Max,max)]

def fin(a):
    c=1
    for i in range(len(a)):
        for j in range(c,len(a[i])):
            if(a[i][j]!=0):
                return False
        c+=1
    return True

def cycle(ar,res):
    k1=0
    k2=0
    for r in res:
        if(ar[0]==r[0] or ar[0]==r[1]):
            k1=1
        if(ar[1]==r[0] or ar[1]==r[1]):
            k2=1
    return (k1==1 and k2==1)


def Kruskal(df,a):
    """ propose la liste des arcs à ajouter dans notre classifieur sous la forme d'une liste de triplet (𝑎𝑡𝑡𝑟1,𝑎𝑡𝑡𝑟2,𝑝𝑜𝑖𝑑𝑠) """
    res=[]
    while(not fin(a)):
        arete=aretePoidsMaximale(df,a)
        while(cycle(arete[1],res)):
            x=arete[0][0]
            y=arete[0][1]
            a[x][y]=0
            arete=aretePoidsMaximale(df,a)

        res.append(arete[1])
        x=arete[0][0]
        y=arete[0][1]
        a[x][y]=0
    return res      
   
def ConnexSets(liste_arcs):
    """ renvoie une liste d'ensemble d'attributs connectés  """
    res=[]
    for l in liste_arcs:
        k=0
        for elem in res:
            if((l[0] in elem) or (l[1] in elem)):
                k=1
                elem.add(l[1])
                elem.add(l[0])
        if(k==0):
            res.append({l[0],l[1]})
    return res