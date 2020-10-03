# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:53:43 2020

@author: ADJARIAN Stéphan
"""
import numpy as np
import time

"""La classe vecteur 4D permet les manipulations élémentaires (à l'aide de surcharge de méthode) sur un vecteur 
de dimension 4. C'est également dans cette classe que l'on calcule la distance euclienne ou la distance de Manhattan."""
class Vect4D: 
    def __init__(self,x=0.,y=0.,z=0.,t=0.):
        if x=='':self.x=0.
        else:self.x=float(x)  
        if y=='':self.y=0.
        else:self.y=float(y)
        if z=='':self.z=0.
        else:self.z=float(z)
        if t=='': self.t=0.
        else:self.t=float(t)
        self.liste=[self.x,self.y,self.z,self.t]        
    def __str__(self):
        return "[{}, {}, {}, {}]".format(self.x,self.y,self.z,self.t)    
    def __abs__(self):
        return (self.x**2+self.y**2+self.z**2+self.t**2)**0.5    
    def __add__(self,other):
        return Vect4D(self.x+other.x,self.y+other.y,self.z+other.z,self.t+other.t)    
    def __sub__(self,other):
        return Vect4D(self.x-other.x,self.y-other.y,self.z-other.z,self.t-other.t)    
    def __equal__(self,other):
        return self.x==other.x and self.y==other.y and self.z==other.z and self.t==other.t    
    def __mul__(self,other):
        if isinstance(other,Vect4D):
            return self.x*other.x+self.y*other.y+self.z*other.z+self.t*other.t
        elif isinstance(other,int) or isinstance(other,float):         
            return Vect4D(self.x*other,self.y*other,self.z*other,self.t*other)
    def __getitem__(self,i):
        assert i in range(4)
        if i==0: return self.x
        if i==1: return self.y
        if i==2: return self.z
        if i==3: return self.t             
    def __setitem__(self,i,val):
        assert i in range(4)
        if i==0: self.x=val
        if i==1: self.y=val
        if i==2: self.z=val
        if i==3: self.t=val
    def distance(self, other):   #on calcule la distance entre l'obet actuel et toutes les données du dataset.
        return (abs(self.x-other.x) + abs(self.y-other.y) + abs(self.z-other.z) + abs(self.t-other.t))  #Manhattan distance
        #return ((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2 +(self.t-other.t)**2)**0.5 #Euclidean Distance
    
"""Cette fonction permet de charger les données dans une matrice
Les données sont stockés sous cette forme : [[Vect4D, label],[Vect4D, label],...]
Il faut passer en paramètre le nom du fichier se trouvant dans le même dossier que le fichier .py"""
def Acquisition(nom_fichier):
    s=str(open(nom_fichier, 'r').read()).replace(";", " ").split()
    matrice=[]
    k=0
    for i in range(len(s)//5):
        matrice.append([Vect4D(float(s[k]),float(s[k+1]),float(s[k+2]),float(s[k+3])),s[k+4]])
        k+=5
    return matrice       

"""Même fonction mais adaptée pour le fichier final n'ayant pas de label"""
def AcquisitionFichierFinal(nom_fichier):
    s=str(open(nom_fichier, 'r').read()).replace(";", " ").split()
    matrice=[]
    k=0
    for i in range(len(s)//4):
        matrice.append([Vect4D(float(s[k]),float(s[k+1]),float(s[k+2]),float(s[k+3]))])
        k+=4
    return matrice

"""Recherchekvoisin retourne les k plus proches voisins, triés dans l'ordre, de l'objet vect."""
def rechercheKvoisins(mat, VectToTest, k): 
    matrice=[]
    for vect in mat: #mat=matrice contenant les vecteurs4D
        matrice.append([VectToTest.distance(vect[0]),vect[1]]) #on stock dans  une matrice    
    #tri de la matrice dans l'ordre croissant des distances et on retourne les k premiers
    return sorted(matrice, key=lambda x : x[0])[0:k] 

"""labelPossible retourne toutes les classes que peuvent prendre nos objets du dataset."""
def LabelPossible(matrice_entrainement): 
    labelPossible=[]
    for i in matrice_entrainement:
        if i[1] not in labelPossible:
            labelPossible.append(i[1]) 
    return labelPossible

"""La fonction KNN est le corps de l'algorithme. Elle retourne une prédiction de la classe de l'objet.
On lui passe en paramètre l'objet dont on doit prédire la classe,
la matrice contenant les "voisins" (les éléments dont on connait la classe) 
et enfin le nombre de voisins k."""
def KNN(vect,  mat, k, affichage):
    #Affiche = True or False si on veut afficher ou non le texte (utile pour la fonction MatriceConfusion)
    if affichage:
        print("Vous voulez prédire la classe de l'objet aux paramètres suivants :", vect, '\n')
    #Récupération des classes possibles
    labelPossible=LabelPossible(mat)     
    #liste des k plus proches voisins
    listeVoisin=rechercheKvoisins(mat, vect, k)
    #On récupère juste les classes et pas la distance.
    labelVoisin=[]
    for i in listeVoisin:
        labelVoisin.append(i[1])
    #Calcul du mode : la classe la plus fréquente
    maxi=0
    labelmax=''
    for label in labelPossible:
        #on compte le nombre d'occurence de chaque classe
        temp=labelVoisin.count(label)
        if affichage:
            print("Parmi les", k, "plus proches voisins, on a trouvé", temp, label)
        if temp>maxi:
            #on met à jour la classe la plus fréquente dans labelmax
            maxi=temp
            labelmax=label
    if affichage:
        print("\nOn pense donc que l'objet passé en paramètre a pour label", labelmax)     
    return labelmax
          
"""Cette fonction permet de tester notre modèle sur un dataset. 
Elle calcule et affiche la matrice de confusion du modèle. """
def MatriceConfusion(k,matrice_test):
    #On fait passer tous les objets à tester du jeu de données dans le KNN
    #Début du décompte du temps
    start_time = time.time()
    matrice_entrainement=Acquisition("data.csv") #+Acquisition("preTest.csv")
    #Récupération des labels possibles
    labelPossible=LabelPossible(matrice_entrainement)
    #Création de la matrice de confusion en fonction du nombre de classe possible.
    matrice=np.zeros((len(labelPossible),len(labelPossible)))
    #Remplissage matrice de confusion
    for X in matrice_test:  
        labelmax=KNN(X[0], matrice_entrainement, k, False)
        temp=0
        ligne=0
        colonne=0
        #temp vient ici prendre la valeur de l'indice du label correspondant ex: "A" temp=0 ou "J" temp=9
        #On récupère la colonne et la ligne de la matrice de confusion à incrémenter
        for label in labelPossible:
            if X[1]==label:
                ligne=temp
            if labelmax==label:
                colonne=temp
            temp+=1
        #On rajoute 1 dans la bonne case dans la matrice. ex: "H"=="H" matrice[9][9]+=1 
        matrice[ligne][colonne]+=1
    #Partie affichage
    print("\nLa matrice de confusion pour k = %i est donnée par :\n    " % k,end="")
    for i in range(len(matrice)):
        print("", labelPossible[i], " ", end="")
    print("\n",matrice.astype(int))
    #On stock le nombre de bon résultat
    nbrBonResultat=0
    for i in range(len(matrice)):
        nbrBonResultat+=matrice[i][i]
    print("On obtient une accuracy de {}%".format(np.round(nbrBonResultat/len(matrice_test)*100,2)))
    # Affichage du temps d execution
    print("Temps d'execution : %f secondes." % (time.time() - start_time))

"""Cette fonction créée le fichier de sortie contenant chaque prédiction"""
def Prédiction():
    k=8
    fichier_test=AcquisitionFichierFinal("finalTest.csv")
    file = open("NEBOUT_ADJARIAN.txt","w") 
    prédiction=[]
    matrice_entrainement=Acquisition("data.csv")+Acquisition("preTest.csv")
    for vect in fichier_test:  
        #stock dans le fichier de sortie chaque prédiction
        file.write(KNN(vect[0],matrice_entrainement, k, False) + "\n")
        #stockage sous python        
        prédiction.append(KNN(vect[0], matrice_entrainement, k, False))   
    file.truncate(len(prédiction)*3-2)
    file.close() 

#Pour le test d'un seul objet sur le dataset data.csv:
VectToTest=Vect4D(-0.6407110283787365,-0.23944620508552883,-0.9968821134099406,9.17806806490537) #Par exemple
KNN(VectToTest, Acquisition("data.csv"), 5, True) #k=5 et on affiche le résultat

#Pour la matrice de confusion de preTest.csv
MatriceConfusion(8,Acquisition("preTest.csv"))

#Evalutation final
Prédiction()

#Pour la fréquence de chaque classe
#MatriceConfusion(1,Acquisition("data.csv"))