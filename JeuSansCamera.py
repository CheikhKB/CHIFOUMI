# uncompyle6 version 3.6.5
# Python bytecode 3.5 (3351)
# Decompiled from: Python 2.7.11 (default, Sep 29 2016, 13:33:00) 
# [GCC 5.3.1 20160406 (Red Hat 5.3.1-6)]
# Warning: this version has problems handling the Python 3 byte type in contants properly.

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
import ModeJeu
from database import *
from markov import *
game_data = []

class JeuSansCamera(InterfaceJeu):

    def __init__(self, session):
        self.sessionLogin = session
        self.num = 0
        self.idPartie = insererPartie(self.sessionLogin, 1)
        InterfaceJeu.__init__(self)

    def addContenuJeu(self):
        self.setTitleJeu()
        self.frame1.bouton_trad.pack_forget()
        self.frame1.label_score.pack(side=tk.LEFT, padx=200, pady=15, fill='both')

                #FRAME JEU SANS CAMERA
        self.frame2.frame22.frame221 = tk.Frame(self.frame2.frame22, bg="white", height=550, width=600)
        self.frame2.frame22.frame222 = tk.Frame(self.frame2.frame22, bg="white", height=550, width=600)
        self.frame2.frame22.frame221.grid(row=0, column=0)
        self.frame2.frame22.frame222.grid(row=0, column=1)
        self.frame2.frame22.frame221.grid_propagate(0)
        self.frame2.frame22.frame221.pack_propagate(0)
        self.frame2.frame22.frame222.grid_propagate(0)
        self.frame2.frame22.frame222.pack_propagate(0)
        
        self.frame2.frame22.frame221.frame2211 = tk.Frame(self.frame2.frame22.frame221, bg='white', height=550, width=350)
        self.frame2.frame22.frame221.frame2212 = tk.Frame(self.frame2.frame22.frame221, bg='white', height=550, width=250)
        self.frame2.frame22.frame221.frame2211.grid(row=0, column=0)
        self.frame2.frame22.frame221.frame2212.grid(row=0, column=1)
        self.frame2.frame22.frame221.frame2211.grid_propagate(0)
        self.frame2.frame22.frame221.frame2211.pack_propagate(0)
        self.frame2.frame22.frame221.frame2212.grid_propagate(0)
        self.frame2.frame22.frame221.frame2212.pack_propagate(0)
        self.frame2.frame22.frame221.frame2211.frame22111 = tk.Frame(self.frame2.frame22.frame221.frame2211, bg='white', height=100, width=350)
        self.frame2.frame22.frame221.frame2211.frame22112 = tk.Frame(self.frame2.frame22.frame221.frame2211, bg='white', height=350, width=350)
        self.frame2.frame22.frame221.frame2211.frame22113 = tk.Frame(self.frame2.frame22.frame221.frame2211, bg='white', height=100, width=350)
        self.frame2.frame22.frame221.frame2211.frame22111.grid(row=0, column=0)
        self.frame2.frame22.frame221.frame2211.frame22112.grid(row=1, column=0)
        self.frame2.frame22.frame221.frame2211.frame22113.grid(row=2, column=0)
        self.frame2.frame22.frame221.frame2211.frame22111.grid_propagate(0)
        self.frame2.frame22.frame221.frame2211.frame22111.pack_propagate(0)
        self.frame2.frame22.frame221.frame2211.frame22112.grid_propagate(0)
        self.frame2.frame22.frame221.frame2211.frame22112.pack_propagate(0)
        self.frame2.frame22.frame221.frame2211.frame22113.grid_propagate(0)
        self.frame2.frame22.frame221.frame2211.frame22113.pack_propagate(0)
        self.frame2.frame22.frame221.frame2211.frame22111.label_score = tk.Label(self.frame2.frame22.frame221.frame2211.frame22111, text='Score ', bg='white', fg='#0040d9', font=('Courier',
                                                                                                                                                                                   25,
                                                                                                                                                                                   'bold'), justify='center')
        self.frame2.frame22.frame221.frame2211.frame22111.label_score.pack(fill='both', pady=20)
        self.bind('p', self.pierre)
        self.bind('0', self.pierre)
        self.bind('f', self.feuille)
        self.bind('1', self.feuille)
        self.bind('c', self.ciseau)
        self.bind('2', self.ciseau)
        self.frame2.frame22.frame221.frame2211.frame22112.button_p = tk.Button(self.frame2.frame22.frame221.frame2211.frame22112, bg='white', fg='black', text=self.aff_commande(0), command=self.pierre, width=20, height=5)
        self.frame2.frame22.frame221.frame2211.frame22112.button_f = tk.Button(self.frame2.frame22.frame221.frame2211.frame22112, bg='white', fg='black', text=self.aff_commande(1), command=self.feuille, width=20, height=5)
        self.frame2.frame22.frame221.frame2211.frame22112.button_c = tk.Button(self.frame2.frame22.frame221.frame2211.frame22112, bg='white', fg='black', text=self.aff_commande(2), command=self.ciseau, width=20, height=5)
        self.frame2.frame22.frame221.frame2211.frame22112.button_p.grid(row=0, column=0, pady=15, padx=60)
        self.frame2.frame22.frame221.frame2211.frame22112.button_f.grid(row=1, column=0, pady=15, padx=60)
        self.frame2.frame22.frame221.frame2211.frame22112.button_c.grid(row=2, column=0, pady=15, padx=60)
        self.frame2.frame22.frame221.frame2212.frame22121 = tk.Frame(self.frame2.frame22.frame221.frame2212, bg='white', height=100, width=250)
        self.frame2.frame22.frame221.frame2212.frame22122 = tk.Frame(self.frame2.frame22.frame221.frame2212, bg='white', height=350, width=250)
        self.frame2.frame22.frame221.frame2212.frame22123 = tk.Frame(self.frame2.frame22.frame221.frame2212, bg='white', height=100, width=250)
        self.frame2.frame22.frame221.frame2212.frame22121.grid(row=0, column=0)
        self.frame2.frame22.frame221.frame2212.frame22122.grid(row=1, column=0)
        self.frame2.frame22.frame221.frame2212.frame22123.grid(row=2, column=0)
        self.frame2.frame22.frame221.frame2212.frame22121.grid_propagate(0)
        self.frame2.frame22.frame221.frame2212.frame22121.pack_propagate(0)
        self.frame2.frame22.frame221.frame2212.frame22122.grid_propagate(0)
        self.frame2.frame22.frame221.frame2212.frame22122.pack_propagate(0)
        self.frame2.frame22.frame221.frame2212.frame22123.grid_propagate(0)
        self.frame2.frame22.frame221.frame2212.frame22123.pack_propagate(0)
        self.frame2.frame22.frame221.frame2212.frame22123.label_user = tk.Label(self.frame2.frame22.frame221.frame2212.frame22123, text=self.aff_joueur(), bg='white', fg='#0040d9', font=('Courier',
                                                                                                                                                                                           20,
                                                                                                                                                                                           'bold',
                                                                                                                                                                                           'underline'), justify='center')
        self.frame2.frame22.frame221.frame2212.frame22123.label_user.pack(fill='both', pady=10)
        self.frame2.frame22.frame221.frame2212.frame22122.canvas_joueur = tk.Canvas(self.frame2.frame22.frame221.frame2212.frame22122, bg='white', height=290, width=250, highlightthickness=0)
        self.frame2.frame22.frame221.frame2212.frame22122.canvas_joueur.pack(fill='both', pady=30)
        image_joueur = self.rien
        self.image_joueur = self.frame2.frame22.frame221.frame2212.frame22122.canvas_joueur.create_image(20, 50, anchor=tk.NW, image=image_joueur)
        self.frame2.frame22.frame221.frame2212.frame22121.label_coup_joueur = tk.Label(self.frame2.frame22.frame221.frame2212.frame22121, text='', bg='white', fg='#0040d9', font=('Courier',
                                                                                                                                                                                   22,
                                                                                                                                                                                   'bold',
                                                                                                                                                                                   'underline'), justify='center')
        self.frame2.frame22.frame221.frame2212.frame22121.label_coup_joueur.pack(fill='both', pady=20)
        self.frame2.frame22.frame222.frame2211 = tk.Frame(self.frame2.frame22.frame222, bg='white', height=550, width=250)
        self.frame2.frame22.frame222.frame2212 = tk.Frame(self.frame2.frame22.frame222, bg='white', height=550, width=350)
        self.frame2.frame22.frame222.frame2211.grid(row=0, column=0)
        self.frame2.frame22.frame222.frame2212.grid(row=0, column=1)
        self.frame2.frame22.frame222.frame2211.grid_propagate(0)
        self.frame2.frame22.frame222.frame2211.pack_propagate(0)
        self.frame2.frame22.frame222.frame2212.grid_propagate(0)
        self.frame2.frame22.frame222.frame2212.pack_propagate(0)
        self.frame2.frame22.frame222.frame2211.frame22111 = tk.Frame(self.frame2.frame22.frame222.frame2211, bg='white', height=100, width=250)
        self.frame2.frame22.frame222.frame2211.frame22112 = tk.Frame(self.frame2.frame22.frame222.frame2211, bg='white', height=350, width=250)
        self.frame2.frame22.frame222.frame2211.frame22113 = tk.Frame(self.frame2.frame22.frame222.frame2211, bg='white', height=100, width=250)
        self.frame2.frame22.frame222.frame2211.frame22111.grid(row=0, column=0)
        self.frame2.frame22.frame222.frame2211.frame22112.grid(row=1, column=0)
        self.frame2.frame22.frame222.frame2211.frame22113.grid(row=2, column=0)
        self.frame2.frame22.frame222.frame2211.frame22111.grid_propagate(0)
        self.frame2.frame22.frame222.frame2211.frame22111.pack_propagate(0)
        self.frame2.frame22.frame222.frame2211.frame22112.grid_propagate(0)
        self.frame2.frame22.frame222.frame2211.frame22112.pack_propagate(0)
        self.frame2.frame22.frame222.frame2211.frame22113.grid_propagate(0)
        self.frame2.frame22.frame222.frame2211.frame22113.pack_propagate(0)
        self.frame2.frame22.frame222.frame2211.frame22111.label_coup_machine = tk.Label(self.frame2.frame22.frame222.frame2211.frame22111, text='', bg='white', fg='#0040d9', font=('Courier',
                                                                                                                                                                                    22,
                                                                                                                                                                                    'bold',
                                                                                                                                                                                    'underline'), justify='center')
        self.frame2.frame22.frame222.frame2211.frame22111.label_coup_machine.pack(fill='both', pady=20)
        self.frame2.frame22.frame222.frame2211.frame22112.canvas_ia = tk.Canvas(self.frame2.frame22.frame222.frame2211.frame22112, bg='white', height=290, width=250, highlightthickness=0)
        self.frame2.frame22.frame222.frame2211.frame22112.canvas_ia.pack(fill='both', pady=30)
        image_machine = self.rien
        self.image_machine = self.frame2.frame22.frame222.frame2211.frame22112.canvas_ia.create_image(20, 50, anchor=tk.NW, image=image_machine)
        self.frame2.frame22.frame222.frame2211.frame22113.label_machine = tk.Label(self.frame2.frame22.frame222.frame2211.frame22113, text=self.aff_machine(), bg='white', fg='#0040d9', font=('Courier',
                                                                                                                                                                                               20,
                                                                                                                                                                                               'bold',
                                                                                                                                                                                               'underline'), justify='center')
        self.frame2.frame22.frame222.frame2211.frame22113.label_machine.pack(fill='both', pady=10)
        self.frame2.frame22.frame222.frame2212.frame22121 = tk.Frame(self.frame2.frame22.frame222.frame2212, bg='white', height=100, width=350)
        self.frame2.frame22.frame222.frame2212.frame22122 = tk.Frame(self.frame2.frame22.frame222.frame2212, bg='white', height=350, width=350)
        self.frame2.frame22.frame222.frame2212.frame22123 = tk.Frame(self.frame2.frame22.frame222.frame2212, bg='white', height=100, width=350)
        self.frame2.frame22.frame222.frame2212.frame22121.grid(row=0, column=0)
        self.frame2.frame22.frame222.frame2212.frame22122.grid(row=1, column=0)
        self.frame2.frame22.frame222.frame2212.frame22123.grid(row=2, column=0)
        self.frame2.frame22.frame222.frame2212.frame22121.grid_propagate(0)
        self.frame2.frame22.frame222.frame2212.frame22121.pack_propagate(0)
        self.frame2.frame22.frame222.frame2212.frame22122.grid_propagate(0)
        self.frame2.frame22.frame222.frame2212.frame22122.pack_propagate(0)
        self.frame2.frame22.frame222.frame2212.frame22123.grid_propagate(0)
        self.frame2.frame22.frame222.frame2212.frame22123.pack_propagate(0)
        self.frame2.frame22.frame222.frame2212.frame22121.label_gagnant = tk.Label(self.frame2.frame22.frame222.frame2212.frame22121, text='', bg='white', fg='#0040d9', font=('Courier',
                                                                                                                                                                               24,
                                                                                                                                                                               'bold'))
        self.frame2.frame22.frame222.frame2212.frame22121.label_gagnant.pack(fill='both', pady=20)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221221 = tk.Frame(self.frame2.frame22.frame222.frame2212.frame22122, bg='white', height=175, width=350)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221222 = tk.Frame(self.frame2.frame22.frame222.frame2212.frame22122, bg='white', height=175, width=350)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221221.grid(row=0, column=0)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221222.grid(row=1, column=0)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221221.grid_propagate(0)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221221.pack_propagate(0)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221222.grid_propagate(0)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221222.pack_propagate(0)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221221.label_nom_ia = tk.Label(self.frame2.frame22.frame222.frame2212.frame22122.frame221221, text='', bg='white', fg='#004ce5', font=('Courier',
                                                                                                                                                                                                      18,
                                                                                                                                                                                                      'bold',
                                                                                                                                                                                                      'underline'), wraplength=340)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221221.label_text_ia = tk.Label(self.frame2.frame22.frame222.frame2212.frame22122.frame221221, text='', bg='white', fg='#0559f2', justify='center', font=('Courier',
                                                                                                                                                                                                                         15,
                                                                                                                                                                                                                         'bold'), wraplength=340)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221221.label_nom_ia.grid(row=0, column=0, pady=20)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221221.label_text_ia.grid(row=1, column=0, pady=10)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221222.label_text_fin = tk.Label(self.frame2.frame22.frame222.frame2212.frame22122.frame221222, text='', bg='white', fg='#0559f2', justify='center', font=('Courier',
                                                                                                                                                                                                                          15,
                                                                                                                                                                                                                          'bold'), wraplength=340)
        self.frame2.frame22.frame222.frame2212.frame22122.frame221222.label_text_fin.grid(row=0, column=0, pady=40)
        self.mainloop()

    def setTitleJeu(self):
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            self.setTitle('Jeu Chifoumi: Markov')
        else:
            self.setTitle("Chifoumi's Game: Markov")

    def updateIHM(self, dt2):
        dt1 = data()
        self.frame2.frame21.label_texte.config(text='')
        self.frame2.frame22.frame221.frame2212.frame22121.label_coup_joueur.config(text=self.aff_coup_joueur(dt2))
        self.frame2.frame22.frame222.frame2211.frame22111.label_coup_machine.config(text=self.aff_coup_machine(dt2))
        self.frame2.frame22.frame221.frame2211.frame22111.label_score.config(text='Score :\n' + str(dt1[0]) + ' - ' + str(dt1[1]))
        self.frame2.frame22.frame222.frame2212.frame22121.label_gagnant.config(text=self.aff_text_gagnant(dt2))
        self.frame2.frame22.frame222.frame2212.frame22122.frame221221.label_nom_ia.config(text='IA:')
        self.frame2.frame22.frame221.frame2212.frame22122.canvas_joueur.itemconfigure(self.image_joueur, image=self.aff_image_joueur(dt2))
        self.frame2.frame22.frame222.frame2211.frame22112.canvas_ia.itemconfigure(self.image_machine, image=self.aff_image_machine(dt2))
        self.frame2.frame22.frame222.frame2212.frame22122.frame221221.label_text_ia.config(text=self.aff_text_machine(dt2))
        self.frame2.frame22.frame222.frame2212.frame22122.frame221222.label_text_fin.config(text=self.aff_text_comment(dt2))

    def aff_text_comment(self, dt2):
        dt1 = data()
        i = randint(0, 2)
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            if score(dt2[0], dt2[1]) == -1:
                return self.text_jeu_comment_lose_fr[i]
            if score(dt2[0], dt2[1]) == 0:
                return u'E\u0301galite\u0301 !'
            if score(dt2[0], dt2[1]) == 1:
                return self.text_jeu_comment_win_fr[i]
        else:
            if score(dt2[0], dt2[1]) == -1:
                return self.text_jeu_comment_lose_eng[i]
            if score(dt2[0], dt2[1]) == 0:
                return 'Equality? !'
            if score(dt2[0], dt2[1]) == 1:
                return self.text_jeu_comment_win_eng[i]

    def aff_text_machine(self, dt2):
        dt1 = data()
        i = randint(0, 2)
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            if score(dt2[0], dt2[1]) == -1:
                return self.text_jeu_machine_win_fr[i]
            if score(dt2[0], dt2[1]) == 0:
                return ' ..... '
            if score(dt2[0], dt2[1]) == 1:
                return self.text_jeu_machine_lose_fr[i]
        else:
            if score(dt2[0], dt2[1]) == -1:
                return self.text_jeu_machine_win_eng[i]
            if score(dt2[0], dt2[1]) == 0:
                return 'Equality? !'
            if score(dt2[0], dt2[1]) == 1:
                return self.text_jeu_machine_lose_eng[i]

    def aff_titre_jeu(self):
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            return 'JEU MARKOV'
        else:
            return 'MARKOV GAME'

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

    def aff_coup_machine(self, dt2):
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            if list_coup[dt2[1]] == 'p':
                return self.text_commande_fr[0]
            else:
                if list_coup[dt2[1]] == 'f':
                    return self.text_commande_fr[1]
                return self.text_commande_fr[2]
        else:
            if list_coup[dt2[1]] == 'p':
                return self.text_commande_eng[0]
            else:
                if list_coup[dt2[1]] == 'f':
                    return self.text_commande_eng[1]
                return self.text_commande_eng[2]

    def aff_coup_joueur(self, dt2):
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            if list_coup[dt2[0]] == 'p':
                return self.text_commande_fr[0]
            else:
                if list_coup[dt2[0]] == 'f':
                    return self.text_commande_fr[1]
                return self.text_commande_fr[2]
        else:
            if list_coup[dt2[0]] == 'p':
                return self.text_commande_eng[0]
            else:
                if list_coup[dt2[0]] == 'f':
                    return self.text_commande_eng[1]
                return self.text_commande_eng[2]

    def finpartie(self):
        dt = data()
        updatePartie(self.idPartie, dt[0], dt[1], dt[2])
        renitialiser()
        self.destroy()
        x = ModeJeu.ModeJeu(self.sessionLogin)

    def traduction_page(self):
        self.destroy()
        self.choix_lang()
        x = JeuSansCamera(self.sessionLogin)

    def aff_machine(self):
        fichier = open('langage.txt', 'r')
        lang = fichier.read()
        fichier.close()
        if lang == 'fr':
            return 'Machine Markov'
        else:
            return 'Markov Machine'

    def aff_joueur(self):
        return self.sessionLogin

    def aff_score(self):
        pass

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

    def pierre(self, envent=''):
        dt = modele_markov('0')
        self.num += 1
        self.updateIHM(dt)
        insererCoup(self.num, self.idPartie, 'p', list_coup[dt[1]])

    def feuille(self, envent=''):
        dt = modele_markov('1')
        self.num += 1
        self.updateIHM(dt)
        insererCoup(self.num, self.idPartie, 'f', list_coup[dt[1]])

    def ciseau(self, envent=''):
        dt = modele_markov('2')
        self.num += 1
        self.updateIHM(dt)
        insererCoup(self.num, self.idPartie, 'c', list_coup[dt[1]])