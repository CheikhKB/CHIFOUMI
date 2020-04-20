#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
from database import *
import tkinter.font
from Interface import Interface

class Accueil(Interface):
	"""Classe hérite de la classe Interface, c'est une coquille vide, elle ne doit pas être implémentée!!"""

	def __init__(self):
		Interface.__init__(self)
		self.addContenuAccueilJoueur()
		self.addContenuAccueilAdmin()
					
	def addContenuAccueil(self):
		self.frame1.bouton_menu.pack_forget()
		#frame21
		self.frame2.frame21.canvas21 = tk.Canvas(self.frame2.frame21, bg="white", height=590, width=600, highlightthickness=0) 
		self.frame2.frame21.canvas21.grid(row=0, column=0)
		self.frame2.frame21.canvas21.create_image(46,14, anchor=tk.NW, image=self.choix_image())

		self.frame2.frame22.frame221 = tk.Frame(self.frame2.frame22, bg="white", height=130, width=600) 
		self.frame2.frame22.frame222 = tk.Frame(self.frame2.frame22, bg="white", height=250, width=600)
		self.frame2.frame22.frame223 = tk.Frame(self.frame2.frame22, bg="white", height=110, width=600)
		self.frame2.frame22.frame221.grid(row=0, column=0)
		self.frame2.frame22.frame222.grid(row=1, column=0)
		self.frame2.frame22.frame223.grid(row=2, column=0)	
		#Frame 221
		self.frame2.frame22.frame221.label_titre = tk.Label(self.frame2.frame22.frame221, text=self.aff_welcome(), justify='center', font=("Helvetica", 32, 'bold'), bg='white', fg='#0040d9')
		self.frame2.frame22.frame221.label_titre.pack(side=tk.RIGHT, fill='both', pady=40)
		#Frame222
		self.frame2.frame22.frame222.button_jouer = tk.Button(self.frame2.frame22.frame222, text=self.aff_texte(0), font='Courier 15 bold', command=self.bouton_un, width=20)
		self.frame2.frame22.frame222.button_res = tk.Button(self.frame2.frame22.frame222, text=self.aff_texte(1), font='Courier 15 bold', command=self.bouton_deux, width=20)
		self.frame2.frame22.frame222.button_notice = tk.Button(self.frame2.frame22.frame222, text=self.aff_texte(2), font='Courier 15 bold', command=self.bouton_trois, width=20)
		self.frame2.frame22.frame222.button_jouer.grid(row=0, column=0, pady=20)
		self.frame2.frame22.frame222.button_res.grid(row=1, column=0, pady=0)
		self.frame2.frame22.frame222.button_notice.grid(row=2, column=0, pady=20)
		#frame223
		self.frame2.frame22.frame223.button_exit = tk.Button(self.frame2.frame22.frame223, text=self.aff_texte(3), font='Courier 15 bold', command=self.deconnexion)
		self.frame2.frame22.frame223.frame_vide = tk.Frame(self.frame2.frame22.frame223, bg="white", height=110, width=350)
		self.frame2.frame22.frame223.frame_vide.grid(row=0, column=0)
		self.frame2.frame22.frame223.button_exit.grid(row=0, column=1)

		self.mainloop()

	def aff_welcome(self):
		pass

	def aff_texte(self, entier): 
		pass

	def choix_image(self):
		pass

	def bouton_un(self):
		pass

	def bouton_deux(self):
		pass

	def bouton_trois(self):
		pass

	def deconnexion(self):
		pass

	def addContenuAccueilAdmin(self):
		pass

	def addContenuAccueilJoueur(self):
		pass

	def addContenuSession(self):
		pass


