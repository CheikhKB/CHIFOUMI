#import des bibliotheques
import tkinter as tk
from random import randint, choice
import tkinter.font
from database import *
from Interface import Interface
from Accueil import Accueil
import Notice
import Choix
import HistoriqueJoueur
import Login

class AccueilJoueur(Accueil):
	"""Classe hérite de la classe Accueil"""

	def __init__(self, sl):
		self.sessionLogin = sl
		self.text_wlj_fr = "Bienvenue " + str(self.sessionLogin) + " !"
		self.text_wlj_eng = "Welcome " + str(self.sessionLogin) + " !"
		Accueil.__init__(self)
		

	def setTitleAccueil(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			self.setTitle("Menu Joueur")
		else:
			self.setTitle("User Page")


	def choix_image(self):
		return self.choix_image_cfm("interne")

	def aff_welcome(self):
		fichier = open("langage.txt", "r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_wlj_fr
		else:
			return self.text_wlj_eng

	def aff_texte(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_acceuiljoueur_fr[entier]
		else:
			return self.text_acceuiljoueur_eng[entier]

	def traduction_page(self):
		self.destroy()
		self.choix_lang()
		x = AccueilJoueur(self.sessionLogin)

	def bouton_un(self):
		"""lancement dune partie"""
		self.destroy()
		x = Choix.Choix(self.sessionLogin)

	def bouton_deux(self):
		"""affichage des resultats"""
		self.destroy()
		x = HistoriqueJoueur.HistoriqueJoueur(self.sessionLogin)

	def bouton_trois(self):
		"""affichage de la notice """
		self.destroy()
		x = Notice.Notice(self.sessionLogin)

	def deconnexion(self):
		self.destroy()
		log = Login.Login()

