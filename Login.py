#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
import tkinter.font
from InterfaceSaisie import InterfaceSaisie
from tkinter import messagebox
from database import *
import Inscrire 
import Bienvenue
import AccueilAdmin
import AccueilJoueur

class Login(InterfaceSaisie):
	"""Classe hérite de la classe InterfaceSaisie"""

	def __init__(self):
		InterfaceSaisie.__init__(self)

	def aff_texte(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_login_fr[entier]
		else:
			return self.text_login_eng[entier]


	def aff_image_logo(self):
		return self.images[6]

	def aff_image(self):
		return self.choix_image_cfm("externe")

	def traduction_page(self):
		self.destroy()
		self.choix_lang()
		x = Login()

	def setTitleLogin(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.setTitle("Page Connexion")
		else:
			return self.setTitle("Login Page")

	def addContenuLogin(self):
		self.setTitleLogin()
		self.mainloop()

	def bouton_un(self, envent=""):
		userN=self.frame2.frame22.frame222.frame2221.pseudo.get()
		pwd=self.frame2.frame22.frame222.frame2221.password.get()
	
		#verification
		if userN == "":
			messagebox.showinfo("Infos",self.aff_boite_dial(4))
		elif pwd == "":
			messagebox.showinfo("Infos", self.aff_boite_dial(8))
		else:
			user=getUtilisateur(userN, pwd)
			if user: #ok 
				messagebox.showinfo("Infos", self.aff_boite_dial(5))
				self.destroy()
				if user[2] == "admin":
					x = AccueilAdmin.AccueilAdmin(userN) 
				else : 
					x = AccueilJoueur.AccueilJoueur(userN)
			else:
				messagebox.showinfo("Infos", self.aff_boite_dial(6))
		
	def bouton_deux(self):
		self.destroy()
		x = Inscrire.Inscrire()

	def menuGeneral(self):
		self.destroy()
		x = Bienvenue.Bienvenue()
	
	def addContenuSession(self):
		pass
	

#a voir pour les touches entrées: entree.bind('<Return>', evaluer)