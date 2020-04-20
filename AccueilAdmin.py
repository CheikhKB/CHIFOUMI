#import des bibliotheques
import tkinter as tk
import string as st
import tkinter.font
from Accueil import Accueil
from database import *
from tkinter import messagebox
import HistoriqueAdmin
import AddAdmin
import Liste
import Login

class AccueilAdmin(Accueil):
	"""Classe hérite de la classe Accueil"""
	
	def __init__(self, sl):
		self.sessionLogin = sl
		self.text_wla_fr = "Bienvenue " + str(self.sessionLogin) + " !"
		self.text_wla_eng = "Welcome " + str(self.sessionLogin) + " !"
		Accueil.__init__(self)
		
	def addContenuSession(self):
		pass

	def setTitleAccueil(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			self.setTitle("Menu Admin")
		else:
			self.setTitle("Admin Page")

	def choix_image(self):
		return self.choix_image_cfm("admin")

	def bouton_un(self):
		self.destroy()
		x = HistoriqueAdmin.HistoriqueAdmin(self.sessionLogin)

	def bouton_deux(self):
		self.destroy()
		x = Liste.Liste(self.sessionLogin)

	def bouton_trois(self):
		self.destroy()
		x = AddAdmin.AddAdmin(self.sessionLogin)
		
	def deconnexion(self):
		self.destroy()
		log = Login.Login()

	def aff_welcome(self):
		fichier = open("langage.txt", "r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_wla_fr
		else:
			return self.text_wla_eng

	def aff_texte(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_acceuiladmin_fr[entier]
		else:
			return self.text_acceuiladmin_eng[entier]

	def traduction_page(self):
		self.destroy()
		self.choix_lang()
		x = AccueilAdmin(self.sessionLogin)
