#coding: utf-8

from __future__ import division
from math import sqrt
import random as rnd
from itertools import product
import pickle
import numpy as np
import matplotlib.pyplot as plt

#tuples a prendre en compte
ordre=2

#association 'numéro coup' et son nom 
list_coup  = {'0' : 'pierre', '1' : 'feuille', '2' : 'ciseau'}

#variables de jeu
gains, nuls, pertes, tot_pred, pred_juste, taux= 0,0,0,0,0,0.0

#dictionnaire de règles du jeu. la clé représente un coup et sa valeur est le coup qui le bat
versus={'0' : '1', '1' : '2', '2' : '0'}

#fonction de création du dictionnaire (matrice de transition)
def dictioComb(taille):
	dct = dict()
	for e in list(product(['0','1','2'], repeat=taille)) :
		dct[e]=1
	return dct

#toutes combinaisons de sequences de (ordre+1) coups
comb=dictioComb(ordre+1)

#derniers coups de l'user
coups2=('x',) * ordre

#variable servant à collectionner les taux de prédiction après chaque coup d'une partie
list_taux=[]  

#variable servant à connaitre l'évolution du score de la machine
bin_score=[]
		
#fonction principale de jeu		
def modele_markov(x):
	global ordre
	global coups2
	global comb
	global list_coup
	global gains
	global nuls
	global pertes
	global tot_pred
	global pred_juste
	global list_taux
	global bin_score, test4

	
	#choisir 'ordre' coups aléatoires pour le début 
	if coups2[0]=='x':
		y = rnd.choice(['0','1','2'])

	#predire
	else: 
		#nombre total de prédictions tentées
		tot_pred+=1
		  
 		#effectifs partiels pour calcul de probabilités
		nb_p = comb[coups2 + ('0',)]
		nb_f = comb[coups2 + ('1',)]
		nb_c = comb[coups2 + ('2',)]

		#effectif total
		nb_tot = nb_p+nb_f+nb_c

		#calcul des probabilités de transition
		prob = [nb_p/nb_tot, nb_f/nb_tot, 1-(nb_p/nb_tot)-(nb_f/nb_tot)]
		
		#choix du coup ayant la plus grande probabilité
		z=str(prob.index(max(prob)))
		
		y=versus[z]

		#mise à jour du dictionnaire
		comb[coups2 + (x,)] += 1

	#prise en compte du dernier coup de l'adversaire
	coups2 = coups2[1:]+(x,)

	#calcul résultat du round
	if versus[x] == y:
		pertes += 1
		bin_score.append(1) #pour graphe4
		if tot_pred>0:
		  	pred_juste+=1  	
	elif x==y:
		nuls   += 1
		bin_score.append(0)
	else:
		gains   += 1
		bin_score.append(0)

	if tot_pred>0:
	  	list_taux.append(round(((pred_juste/tot_pred))*100,2))
	else:
		list_taux.append(0.0)

#fonction de lecture des données de fichiers
def recup_data():
	with open("data1.txt", "rb") as fp :   
		dataG= pickle.load(fp)
	return dataG

#fonction d'initialisation des variables de jeu
def init():
	global ordre
	global coups2
	global comb
	global list_coup
	global gains
	global nuls
	global pertes
	global tot_pred
	global pred_juste
	global list_taux
	gains, nuls, pertes, tot_pred, pred_juste, taux= 0,0,0,0,0,0.0
	coups2=('x',) * ordre
	comb=dictioComb(ordre+1)
	list_taux=[]

def test():
	global ordre
	global list_taux

	#liste des taux de prédiction obtenus pour chaque ordre en moyenne
	taux_pred=[]

	#liste des taux de victoire obtenus pour chaque ordre en moyenne
	taux_vic=[]

	#liste des pourcentages de nul obtenus pour chaque ordre en moyenne
	taux_nul=[]

	#liste des taux de prédiction obtenus apres chaque coup pour la partie 1 (jusqu'à ordre 4)
	taux_partie_1 =[]

	#somme totale des taux de chaque partie pour calculer après la moyenne
	sum_taux=0.0

	#le nombre total de parties jouées
	tot_part=0

	#nombre de victoires de la machine
	win=0

	#nombre de nuls de la machine
	draw=0
	
	i=1
	while i<=8 : #pour ordre de 1 à 8
		partie_1=True
		ordre=i
		game_data=recup_data()
		for partie in game_data:
			init() #(ré)initialisation des vars
			tot_part+=1
			for coup in partie:
				modele_markov(coup[1])
			if pertes>gains:
				win+=1
			elif gains==pertes:
				draw+=1
			
			if tot_pred>0:
				taux=(pred_juste/tot_pred)*100 	
				sum_taux+=taux	

			#on prend en compte que la partie 1 (étude de l'évolution du taux de prédiction)
			if partie_1 and ordre<=4: 
				taux_partie_1.append(list_taux)
				partie_1=False

		#calcul pourcentages moyens de victoires et de nuls de la machine
		vix=(win/tot_part)*100
		nix=(draw/tot_part)*100
		
		taux_pred.append(round(sum_taux/tot_part,2))
		taux_vic.append(round(vix,2))
		taux_nul.append(round(nix,2))
		i+=1

	return (taux_pred, taux_vic, taux_nul, taux_partie_1)


#Courbes des taux moyen de prédiction et de victoires en fonction de l’ordre
def graphe1(y,v): 
	x= [1,2,3,4,5,6,7,8]
	plt.plot(x, y, label='taux moyen prediction')
	plt.plot(x, v, label='moyenne de victoires')
	plt.ylabel('pourcentage')
	plt.xlabel('ordre')
	plt.legend()
	plt.show()


#Histogramme des taux de victoires et de nuls en fonction de l’ordre
def graphe2(y,v,n):
	ind = np.arange(len(v))   
	width = 0.35 
	p1 = plt.bar(ind, v, width, color='#d62728')
	p2 = plt.bar(ind, n, width, bottom=v)
	plt.ylabel('pourcentage')
	plt.xlabel('ordre')
	plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8'))
	plt.yticks(np.arange(0, 101, 10))
	plt.legend((p1[0], p2[0]), ('Victoire', 'Nul'))
	plt.show()


#Evolution du taux de réussite de prédiction dans une partie donnée
def graphe3(t):
	x= [i for i in range(len(t[0]))]
	for i in range(len(t)):
		list_taux='ordre'+str(i+1)
		plt.plot(x, t[i], label=list_taux)
	plt.ylabel('taux de prediction')
	plt.xlabel('rounds')
	plt.legend()
	plt.show()


#Evolution de la détection de changement de stratégie en fonction de l’ordre
def graphe4():
	global bin_score
	global ordre
	coups=['0', '1', '2', '0', '1', '2', '0', '1', '2', '0', '1', '2', '1', '0', '2', '1', '0', '2', '1', '0', '2', '1', '0', '2', '1', '0', '2', '1', '0', '2','1', '0', '2', '1', '0', '2', '1', '0', '2','1', '0', '2', '1', '0', '2', '1', '0', '2']
	bin_score=[]
	nb_coups_list=[]


	#calcul du nombre de coups nécessaires pour détecter le changement de stratégie
	k=1
	while k<=8 : #pour ordre de 1 à 8
		bin_score=[]
		ordre=k
		init()
		for coup in coups:
			modele_markov(coup)
		bin_score.reverse()
		i=0
		for b in bin_score:
			if b==0:
				break
			i+=1
		nb_coups=len(bin_score)-i-12

		nb_coups_list.append(nb_coups)
		k+=1

	x= [1,2,3,4,5,6,7,8]
	plt.plot(x, nb_coups_list)
	plt.ylabel('coups')
	plt.xlabel('ordre')
	plt.legend()
	plt.yticks(np.arange(4, 16, 1))
	plt.show()
	test4=False
	bin_score=[]
	
#programme principal
if __name__ == "__main__":
	y,v,n,t = test()
	graphe1(y,v)
	graphe2(y,v,n)
	graphe3(t)
	graphe4()



	
