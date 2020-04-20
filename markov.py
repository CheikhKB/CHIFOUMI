from __future__ import division
from math import sqrt
import random as rnd


#ttes combinaisons de sequences de 3 coups
comb = { '000' : 1, '001' : 1, '002' : 1, '010' : 1, '011' : 1, '012' : 1, '020' : 1, '021' : 1, '022' : 1, '100' : 1, '101' : 1, '102' : 1, '110' : 1, '111' : 1, '112' : 1, '120' : 1, '121' : 1, '122' : 1, '200' : 1, '201' : 1, '202' : 1, '210' : 1, '211' : 1, '212' : 1, '220' : 1, '221' : 1, '222' : 1 }
list_coup  = {'0' : 'p', '1' : 'f', '2' : 'c'}
gains, nuls, pertes, tot_pred, pred_juste, taux= 0,0,0,0,0,0.0
last2 = '33' #2 derniers coups de l'user

def score(a, b):
    if a == '0' and b == '1' or a == '1' and b == '2' or a == '2' and b == '0':
        return -1 #on perd
    elif a == b:
        return 0 #nul
    else:
        return 1 #on gagne
        
def modele_markov(x):
    global last2
    global comb
    global list_coup
    global gains
    global nuls
    global pertes
    global tot_pred
    global pred_juste
    global taux

    if(last2[0] == '3'):
        y = str(rnd.randint(0,2)) #1er et 2e coup ordi 
    else: #predire
        tot_pred+=1
          
        nb_p = comb[last2 + '0']
        nb_f = comb[last2 + '1']
        nb_c = comb[last2 + '2']

        nb_tot = nb_p+nb_f+nb_c

        prob = [nb_p/nb_tot, nb_f/nb_tot, 1-(nb_p/nb_tot)-(nb_f/nb_tot)]
        
        res = [max(prob[2]-prob[1],0), max(prob[0]-prob[2],0), max(prob[1]-prob[0],0)]
        norme = sqrt(res[0]*res[0] + res[1]*res[1] + res[2]*res[2])
        res = [res[0]/norme, res[1]/norme, 1-(res[0]/norme)-(res[1]/norme)]
        
        y = rnd.uniform(0,1)


        if y <= res[0]: #'ciseau' predit
            y = '0' #joue 'pierre'
        elif y <= res[0] + res[1]: #'pierre' predit
            y = '1' #joue 'feuille'
        else: #'feuille' predit
            y = '2' #joue 'ciseau'

        #maj dictio
        comb[last2+x] += 1

    last2 = last2[1] + x

    if score(x,y) == -1:
        pertes += 1
        if tot_pred>0:
          pred_juste+=1
    elif score(x,y) == 0:
        nuls   += 1
    elif score(x,y) == 1:
        gains   += 1

    if tot_pred>0:
        taux=(pred_juste/tot_pred)*100 		
    taux=round(taux,2)

    dt=[x,y]
    
    return dt


def data():
    global gains
    global pertes
    global taux

    dt=[gains, pertes, taux]

    return dt


def renitialiser():
    global gains
    global pertes
    global taux
    gains,pertes,taux=0,0,0