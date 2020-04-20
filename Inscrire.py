#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
import tkinter.font
from InterfaceSaisie import InterfaceSaisie
from tkinter import messagebox
from database import *
import Login
import Bienvenue
import AccueilJoueur

class Inscrire(InterfaceSaisie):
	"""Classe hérite de la classe InterfaceSaisie"""

	def __init__(self):
		InterfaceSaisie.__init__(self)

	def aff_texte(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_inscrire_fr[entier]
		else:
			return self.text_inscrire_eng[entier]

	def aff_image_logo(self):
		return self.images[5]

	def aff_image(self):
		return self.choix_image_cfm("externe")

	def traduction_page(self):
		self.destroy()
		self.choix_lang()
		x = Inscrire()

	def setTitleInscrire(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			self.setTitle("Inscription")
		else:
			self.setTitle("Inscription")

	def addContenuInscrire(self):
		self.setTitleInscrire()
		self.frame2.frame22.frame222.frame2221.label_ps_conf = tk.Label(self.frame2.frame22.frame222.frame2221, text=self.aff_texte(6), justify='center', font=("Courier", 12, 'bold'), bg='white', fg='#1266ff')
		self.frame2.frame22.frame222.frame2221.password_conf = tk.Entry(self.frame2.frame22.frame222.frame2221,show='*', font='Courier 12')
		self.frame2.frame22.frame222.frame2221.label_ps_conf.grid(row=2, column=0, pady=15, padx=15)
		self.frame2.frame22.frame222.frame2221.password_conf.grid(row=2, column=1, pady=15, padx=15)
		self.mainloop()

	def bouton_un(self, envent=""):
		user=self.frame2.frame22.frame222.frame2221.pseudo.get()
		pwd1=self.frame2.frame22.frame222.frame2221.password.get()
		pwd2=self.frame2.frame22.frame222.frame2221.password_conf.get()
		# verification
		if user=="" or pwd1=="" or pwd2=="":
			messagebox.showinfo("Infos",self.aff_boite_dial(0))
		elif loginNonUsed(user) :
			if pwd1 != pwd2 :
				messagebox.showinfo("Infos", self.aff_boite_dial(1))
			else:
				insererUtilisateur(user,pwd1,"user")
				messagebox.showinfo("Infos", self.aff_boite_dial(7))
				self.destroy()
				x = AccueilJoueur.AccueilJoueur(user)
		else:
			messagebox.showinfo("Infos", self.aff_boite_dial(3))


	def bouton_deux(self):
		self.destroy()
		x = Login.Login()
	
	def menuGeneral(self):
		self.destroy()
		x = Bienvenue.Bienvenue()

	def addContenuSession(self):
		pass
