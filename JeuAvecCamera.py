# uncompyle6 version 3.6.5
# Python bytecode 3.5 (3351)
# Decompiled from: Python 2.7.11 (default, Sep 29 2016, 13:33:00) 
# [GCC 5.3.1 20160406 (white Hat 5.3.1-6)]
# Warning: this version whiteproblems handling the Python 3 byte type in contants properly.

# Embedded file name: /home/aSara/Documents/TRAVAIL/ING2/PROJET/Travail/ELABORATION1/125061_FILES/REPTILE/TRAVAIL/IHM_sara/FINAL_SASSOU/test_phpadmin/JeuSansCamera.py
# Compiled at: 2020-04-19 12:23:07
# Size of source mod 2**32: 18964 bytes
import tkinter as tk, string as st
from random import randint, choice
from tkinter.filedialog import askopenfilename
from imghdr import what
import os, tkinter.font
from InterfaceJeu import InterfaceJeu
import AccueilJoueur
from database import *
import pickle
from markov import *
from PIL import Image, ImageTk
import cv2
from CNN import *
import ModeJeu



#OU se situe la frame de la camera: self.frame2.frame22.frame221.webcam

game_data = []

class JeuAvecCamera(InterfaceJeu):

    def __init__(self, session):
        self.sessionLogin = session
        self.num = 0
        self.play=0
        self.time=0
        self.play_pause=0
        self.idPartie = insererPartie(self.sessionLogin, 1)
        if os.path.isfile('data.txt'):
            with open('data.txt', 'rb') as (fp):
                self.dataG = pickle.load(fp)
            self.id_partie = self.dataG[(-1)][(-1)][0] + 1
        else:
            self.dataG = []
            self.id_partie = 0
        InterfaceJeu.__init__(self)

    def addContenuJeu(self):
        self.setTitleJeu() 
        wy=400
        self.frame1.bouton_trad.pack_forget()
        self.frame1.label_score.pack(side=tk.LEFT, padx=200, pady=15, fill='both')


        self.frame2.frame22.frame221 = tk.Frame(self.frame2.frame22, bg="white", height=125, width=1200) 
        self.frame2.frame22.frame222 = tk.Frame(self.frame2.frame22, bg="white", height=425, width=1200)
        self.frame2.frame22.frame221.grid(row=0, column=0)
        self.frame2.frame22.frame222.grid(row=1, column=0)
        self.frame2.frame22.frame221.grid_propagate(0)
        self.frame2.frame22.frame221.pack_propagate(0)
        self.frame2.frame22.frame222.grid_propagate(0)
        self.frame2.frame22.frame222.pack_propagate(0)

        #PARTIE DU HAUT
        self.frame2.frame22.frame221.frame2211 = tk.Frame(self.frame2.frame22.frame221 , bg= "white", height=125, width=400)
        self.frame2.frame22.frame221.frame2212 = tk.Frame(self.frame2.frame22.frame221 , bg= "white", height=125, width=400)
        self.frame2.frame22.frame221.frame2213 = tk.Frame(self.frame2.frame22.frame221 , bg= "white", height=125, width=400)
        self.frame2.frame22.frame221.frame2211.grid(row=0, column=0)
        self.frame2.frame22.frame221.frame2212.grid(row=0, column=1)
        self.frame2.frame22.frame221.frame2213.grid(row=0, column=2)
        self.frame2.frame22.frame221.frame2211.grid_propagate(0)
        self.frame2.frame22.frame221.frame2211.pack_propagate(0)
        self.frame2.frame22.frame221.frame2212.grid_propagate(0)
        self.frame2.frame22.frame221.frame2212.pack_propagate(0)
        self.frame2.frame22.frame221.frame2213.grid_propagate(0)
        self.frame2.frame22.frame221.frame2213.pack_propagate(0)
        #score
        self.frame2.frame22.frame221.frame2211.label_score = tk.Label(self.frame2.frame22.frame221.frame2211, bg="white", text='Score: ', fg='#0040d9', font=('Courier',25,'bold'), justify='center')
        self.frame2.frame22.frame221.frame2211.label_score.pack(pady=20)
        #bouton du lancement
        self.frame2.frame22.frame221.frame2213.bouton_lancement=tk.Button(self.frame2.frame22.frame221.frame2213, bg="white", fg="black", text=self.aff_jouer(), command=self.play_paus, width=10)
        self.frame2.frame22.frame221.frame2213.bouton_lancement.pack(pady=40)
        ###timer
        self.imgt = tk.PhotoImage(file='images/0.gif')
        self.label3 = tk.Label(self.frame2.frame22.frame221.frame2212, image=self.imgt, bg='white')
        self.label3.pack()

        #PARTIE DU BAS
        self.frame2.frame22.frame222.frame_gauche = tk.Frame(self.frame2.frame22.frame222, bg="white", height=425, width=600)
        self.frame2.frame22.frame222.frame_droite = tk.Frame(self.frame2.frame22.frame222, bg="white", height=425, width=600)
        self.frame2.frame22.frame222.frame_gauche.grid(row=0, column=0)
        self.frame2.frame22.frame222.frame_droite.grid(row=0, column=1)
        self.frame2.frame22.frame222.frame_droite.grid_propagate(0)
        self.frame2.frame22.frame222.frame_droite.pack_propagate(0)
        self.frame2.frame22.frame222.frame_gauche.grid_propagate(0)
        self.frame2.frame22.frame222.frame_gauche.pack_propagate(0)

        #Partie caméra
        self.frame2.frame22.frame222.frame_gauche.frame_cam = tk.Frame(self.frame2.frame22.frame222.frame_gauche, bg="white", height=400, width=500)
        self.frame2.frame22.frame222.frame_gauche.frame_cam.pack(pady=12.25)
        self.frame2.frame22.frame222.frame_gauche.frame_cam.pack_propagate(0)
        self.frame2.frame22.frame222.frame_gauche.frame_cam.grid_propagate(0)



        #Partie droite: coups
        self.frame2.frame22.frame222.frame_droite.frame_jeu = tk.Frame(self.frame2.frame22.frame222.frame_droite, bg="white", height=400, width=500)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.pack(pady=12.25)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.pack_propagate(0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.grid_propagate(0)

        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut = tk.Frame(self.frame2.frame22.frame222.frame_droite.frame_jeu, bg="white", height=300, width=500)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas = tk.Frame(self.frame2.frame22.frame222.frame_droite.frame_jeu, bg="white", height=100, width=500)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.grid(row=0, column=0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.grid(row=1, column=0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.grid_propagate(0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.pack_propagate(0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.grid_propagate(0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.pack_propagate(0)



        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_joueur = tk.Frame(self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut, bg="white", height=150, width=200)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_ia = tk.Frame(self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut, bg="white", height=150, width=200)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_joueur.grid(row=0, column=0, padx=25, pady=75)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_ia.grid(row=0, column=1, padx=25, pady=75)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_joueur.grid_propagate(0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_joueur.pack_propagate(0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_ia.grid_propagate(0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_ia.pack_propagate(0)

        #image joueur
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_joueur.canvas_joueur = tk.Canvas(self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_joueur, bg='white', height=150, width=200, highlightthickness=0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_joueur.canvas_joueur.pack()
        image_joueur = self.rien
        self.image_joueur = self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_joueur.canvas_joueur.create_image(10, 10, anchor=tk.NW, image=image_joueur)
        #image machine
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_ia.canvas_ia = tk.Canvas(self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_ia, bg='white', height=150, width=200, highlightthickness=0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_ia.canvas_ia.pack()
        image_ia = self.rien
        self.image_ia = self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_ia.canvas_ia.create_image(15, 15, anchor=tk.NW, image=image_ia)

        #nom des joueurs
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_joueur = tk.Frame(self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas, width=200, height=75, bg ="white")
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_ia = tk.Frame(self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas, width=200, height=75, bg="white")
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_joueur.grid(row=0, column=0, padx=25, pady=12.25)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_ia.grid(row=0, column=1, padx=25, pady=12.25)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_joueur.grid_propagate(0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_joueur.pack_propagate(0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_ia.grid_propagate(0)
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_ia.pack_propagate(0)

        #nom joueur
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_joueur.label_joueur = tk.Label(self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_joueur,text=self.sessionLogin, bg='white', fg='#0040d9', font=('Courier', 20, 'bold', 'underline'), justify='center')
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_joueur.label_joueur.pack(fill='both', pady=20)
        #nom machine : 
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_ia.label_machine = tk.Label(self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_ia, text="Machine", bg='White', fg='#0040d9', font=('Courier', 20, 'bold', 'underline'), justify='center')
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_bas.frame_nom_ia.label_machine.pack(fill='both', pady=20)
        
        

        
        self.cap = cv2.VideoCapture(0)
        self.affichage = tk.Label(self.frame2.frame22.frame222.frame_gauche.frame_cam)
        self.affichage.pack(fill='both')
        self.addCam()
        self.mainloop()

    def setTitleJeu(self):
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            self.setTitle('Jeu Chifoumi: Markov')
        else:
            self.setTitle("Chifoumi's Game: Markov")

    def aff_jouer(self):
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang=="fr":
            return "Jouer"
        else:
            return "Play"



    def aff_titre_jeu(self):
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            return 'JEU MARKOV'
        else:
            return 'MARKOV GAME'

    def addCam(self):
        _, frame = self.cap.read()
        roi = frame[0:400, 0:500]
        cv2image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        #cv2.imshow(" ",cv2image)
        img1 = cv2.resize(cv2image, (227, 227))
        move=image2move(img1)
        #print(move)
        if self.time == 4 and self.play==1 :
                if move=="p":
                    self.pierre()
                elif move=="f":
                    self.feuille()
                elif move=="c":
                    self.ciseau() 
                self.play=0      	
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.affichage.imgtk = imgtk #Shows frame for display 1
        self.affichage.configure(image=imgtk)
        self.after(10, self.addCam) 


    def updateIHM(self, dt2):
        global list_coup
        dt1 = data()
        self.frame2.frame21.label_texte.config(text=self.aff_text_gagnant(dt2), wraplength=500)
        self.frame2.frame22.frame221.frame2211.label_score.config(text='Score :\n' + str(dt1[0]) + ' - ' + str(dt1[1]))
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_joueur.canvas_joueur.itemconfigure(self.image_joueur, image=self.aff_image_joueur(dt2))
        self.frame2.frame22.frame222.frame_droite.frame_jeu.frame_haut.frame_image_ia.canvas_ia.itemconfigure(self.image_ia, image=self.aff_image_machine(dt2))   

    def aff_text_gagnant(self, dt2):
        dt1 = data()
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            if score(dt2[0], dt2[1]) == -1:
                return 'La machine gagne'
            if score(dt2[0], dt2[1]) == 0:
                return 'Match nul'
            if score(dt2[0], dt2[1]) == 1:
                return self.sessionLogin + ' gagne'
        else:
            if score(dt2[0], dt2[1]) == -1:
                return self.sessionLogin + ' lose'
            if score(dt2[0], dt2[1]) == 0:
                return 'Tie game ? !'
            if score(dt2[0], dt2[1]) == 1:
                return self.sessionLogin + ' win'

    def aff_commande(self, entier):
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            return self.text_commande_fr[entier]
        else:
            return self.text_commande_eng[entier]

    # def aff_coup_machine(self, dt2):
    # 	fichier = open('langage.txt', 'r')
    # 	lang = fichier.read()
    # 	fichier.close()
    # 	if lang == 'fr':
    # 		if list_coup[dt2[1]] == 'p':
    # 			return self.text_commande_fr[0]
    # 		else:
    # 			if list_coup[dt2[1]] == 'f':
    # 				return self.text_commande_fr[1]
    # 			return self.text_commande_fr[2]
    # 	else:
    # 		if list_coup[dt2[1]] == 'p':
    # 			return self.text_commande_eng[0]
    # 		else:
    # 			if list_coup[dt2[1]] == 'f':
    # 				return self.text_commande_eng[1]
    # 			return self.text_commande_eng[2]

    # def aff_coup_joueur(self, dt2):
    # 	fichier = open('langage.txt', 'r')
    # 	lang = fichier.read()
    # 	fichier.close()
    # 	if lang == 'fr':
    # 		if list_coup[dt2[0]] == 'p':
    # 			return self.text_commande_fr[0]
    # 		else:
    # 			if list_coup[dt2[0]] == 'f':
    # 				return self.text_commande_fr[1]
    # 			return self.text_commande_fr[2]
    # 	else:
    # 		if list_coup[dt2[0]] == 'p':
    # 			return self.text_commande_eng[0]
    # 		else:
    # 			if list_coup[dt2[0]] == 'f':
    # 				return self.text_commande_eng[1]
    # 			return self.text_commande_eng[2]

    def finpartie(self):
        dt = data()
        updatePartie(self.idPartie, dt[0], dt[1], dt[2])
        renitialiser()
        self.cap.release()  
        self.destroy()
        x = ModeJeu.ModeJeu(self.sessionLogin)

        

    def traduction_page(self):
        self.destroy()
        self.choix_lang()
        x = JeuAvecCamera(self.sessionLogin)

    def aff_joueur(self):
        return self.sessionLogin

    def aff_machine(self):
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            return 'Machine RNN'
        else:
            return 'RNN Machine'

    def aff_image_machine(self, dt2):
        if list_coup[dt2[1]] == 'p':
            return self.coups_machine[0]
        else:
            if list_coup[dt2[1]] == 'f':
                return self.coups_machine[1]
            return self.coups_machine[2]

    def aff_image_joueur(self, dt2):
        if list_coup[dt2[0]] == 'p':
            return self.coups_joueur[0]
        else:
            if list_coup[dt2[0]] == 'f':
                return self.coups_joueur[1]
            return self.coups_joueur[2]
        


    def timer(self):
        if self.play_pause==1:
            if self.time==0:
               self.time=4
            else:
                self.time-=1
                if self.time==0:
                    imgt='images/'+str(self.time)+'.gif'
                    self.imgt.config(file=imgt)
                    self.play=1
                else:
                    imgt='images/'+str(self.time)+'.gif'
                    self.imgt.config(file=imgt)        
            self.after(1000,self.timer)     

    def play_paus(self):
        if self.play_pause==0:
            self.play_pause=1
            self.frame2.frame22.frame221.frame2213.bouton_lancement.config(text="Pause")
            self.timer()
        elif self.play_pause==1: 
            self.play_pause=0
            self.time=0
            self.play=0
            self.frame2.frame22.frame221.frame2213.bouton_lancement.config(text="Jouer")


    def pierre(self,event=""):
        dt = modele_markov('0')
        self.num+=1
        insererCoup(self.num, self.idPartie, 'p', list_coup[dt[1]])
        self.updateIHM(dt)

    def feuille(self,event=""):
        dt = modele_markov('1')
        self.num+=1
        insererCoup(self.num, self.idPartie, 'f', list_coup[dt[1]])
        self.updateIHM(dt)

    def ciseau(self,event=""):
        dt = modele_markov('2')
        self.num+=1
        insererCoup(self.num, self.idPartie, 'c', list_coup[dt[1]])
        self.updateIHM(dt)



if __name__ == "__main__":
    bienvenue = JeuAvecCamera("badi")
