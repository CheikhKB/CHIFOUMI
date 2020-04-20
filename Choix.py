#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
import tkinter.font
from Interface import Interface
from database import * 
import AccueilJoueur
import Login
import ModeJeu
import JeuSansCameraRnn 
class Choix(Interface):
	"""Classe hérite de la classe Interface"""

	def __init__(self, session):
		self.sessionLogin = session
		Interface.__init__(self)


	def aff_texte(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_choix_fr[entier]
		else:
			return self.text_choix_eng[entier]

	def traduction_page(self):
		self.destroy()
		self.choix_lang()
		x = Choix(self.sessionLogin)


	def setTitleChoix(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			self.setTitle("Choix de l'adversaire")

		else:
			self.setTitle("Choose your opponent")

	def addContenuChoix(self):
		self.frame1.bouton_menu.pack_forget()
		self.setTitleChoix()
		#frame21
		self.frame2.frame21.canvas21 = tk.Canvas(self.frame2.frame21, bg="white", height=590, width=600, highlightthickness=0) 
		self.frame2.frame21.canvas21.grid(row=0, column=0)
		self.frame2.frame21.canvas21.create_image(46,14, anchor=tk.NW, image=self.choix_image_cfm("interne"))
		#on divise en 4 la frame 22
		self.frame2.frame22.frame221 = tk.Frame(self.frame2.frame22, bg="white", height=197, width=600) 
		# #Placement du titre dans le frame221
		self.frame2.frame22.frame221.label_titre = tk.Label(self.frame2.frame22.frame221, text=self.aff_texte(0), justify='center', font=("helvetica", 32, 'bold'), bg='white', fg='#1266ff')
		self.frame2.frame22.frame221.label_titre.pack(side=tk.RIGHT, fill='both')
		self.frame2.frame22.frame221.grid(row=0, column=0, ipady=40)

		self.frame2.frame22.frame222 = tk.Frame(self.frame2.frame22, bg="white", height=196.5, width=600)
		# #Placement des deux boutons dans la frame222
		self.frame2.frame22.frame222.button_markov = tk.Button(self.frame2.frame22.frame222, text=self.aff_texte(1), font='Courier 15 bold', width=20, command=self.markov)
		self.frame2.frame22.frame222.button_rn = tk.Button(self.frame2.frame22.frame222, text=self.aff_texte(2), font='Courier 15 bold', width=20, command=self.rnn)
		self.frame2.frame22.frame222.button_markov.grid(row=0, column=0, pady=20)
		self.frame2.frame22.frame222.button_rn.grid(row=1, column=0, pady=20)
		self.frame2.frame22.frame222.grid(row=1, column=0, pady=10)


		self.frame2.frame22.frame223 = tk.Frame(self.frame2.frame22, bg="white", height=196.5, width=600)
		# #Placement du bouton retour dans la frame223
		self.frame2.frame22.frame223.button_retour = tk.Button(self.frame2.frame22.frame223, text=self.aff_texte(3), justify='center', font=("helvetica", 20, 'bold'), bg='white', fg='black', command=self.menu_joueur)
		self.frame2.frame22.frame223.frame_vide = tk.Frame(self.frame2.frame22.frame223, bg="white", height=196.5, width=400)
		self.frame2.frame22.frame223.frame_vide.grid(row=0, column=0)
		self.frame2.frame22.frame223.button_retour.grid(row=0, column=1)
		self.frame2.frame22.frame223.grid(row=2, column=0)

							
		self.mainloop()
		
	def addContenuSession(self):
		pass

	def menu_joueur(self):
		self.destroy()
		x = AccueilJoueur.AccueilJoueur(self.sessionLogin)
		
	def markov(self):
		self.destroy()
		x = ModeJeu.ModeJeu(self.sessionLogin)

		#mkv = partieMarkov.partieMarkov()
		#mkv.sessionLogin=self.sessionLogin
		#mkv.addContenuSession()

	def rnn(self):
		self.destroy()
		x = JeuSansCameraRnn.JeuSansCameraRnn(self.sessionLogin)
		#rn.sessionLogin = self.sessionLogin
		#rn.addContenuSession()



#a voir pour les touches entrées: entree.bind('<Return>', evaluer)