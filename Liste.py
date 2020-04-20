#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
from database import *
import tkinter.font
from InterfaceInformation import InterfaceInformation
import AccueilAdmin
import AddAdmin

class Liste(InterfaceInformation):
	"""Classe hérite de la classe InterfaceInformation"""

	def __init__(self, session):
		self.sessionLogin = session
		self.text_liste_information_fr = ["English", "Ajouter un Admin", "Retour"]
		self.text_liste_information_eng = ["Français","Add Admin", "Go Back"]
		self.text_liste_fr= ["LISTE UTILISATEURS", "Pseudo", "Rôle"]
		self.text_liste_eng = ["LIST OF USERS", "Name", "Function"]
		InterfaceInformation.__init__(self)

	def addContenuListe(self):		
		#PLACEMENT DES TITRES DANS 2 COLONNES DU FRAME_LABEL (qui contient le titre des colonnes)
		self.frame2.frame_general.frame_label.frame21 = tk.Frame(self.frame2.frame_general.frame_label, bg="white", height=50, width=575)
		self.frame2.frame_general.frame_label.frame22 = tk.Frame(self.frame2.frame_general.frame_label, bg="white", height=50, width=575) 
		self.frame2.frame_general.frame_label.frame21.grid_propagate(0)
		self.frame2.frame_general.frame_label.frame21.pack_propagate(0)
		self.frame2.frame_general.frame_label.frame22.grid_propagate(0)
		self.frame2.frame_general.frame_label.frame22.pack_propagate(0)
		self.frame2.frame_general.frame_label.frame21.grid(row=0, column=0)
		self.frame2.frame_general.frame_label.frame22.grid(row=0, column=1)
		self.frame2.frame_general.frame_label.frame21.label_pseudo = tk.Label(self.frame2.frame_general.frame_label.frame21, text=self.aff_texte(1), fg="#1266ff", font=("Courier", 16, 'bold', 'underline'), bg="white")
		self.frame2.frame_general.frame_label.frame22.label_role = tk.Label(self.frame2.frame_general.frame_label.frame22, text=self.aff_texte(2), fg="#1266ff", font=("Courier", 16, 'bold', 'underline'), bg="white")
		self.frame2.frame_general.frame_label.frame21.label_pseudo.pack(pady=10)
		self.frame2.frame_general.frame_label.frame22.label_role.pack(pady=10)
		self.Colonne = []
		for i in range(0, 2):
			self.Colonne.append(tk.Frame(self.frame2.frame_general.frame_data, bg = "white", height=500, width=200))
			self.Colonne[i].grid_propagate(0)
			self.Colonne[i].pack_propagate(0)
			self.Colonne[i].grid(row=0, column=i, padx=187.5)
		self.setTitleListe()
		self.addContenuSessionListe()
		self.mainloop()

	def aff_texte(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_liste_fr[entier]
		else:
			return self.text_liste_eng[entier]

	def traduction_page(self):
		self.destroy()
		self.choix_lang()
		x = Liste(self.sessionLogin)

	def aff_texte_interface_information(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_liste_information_fr[entier]
		else:
			return self.text_liste_information_eng[entier]

	def setTitleListe(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.setTitle("Liste Utilisateurs")
		else:
			return self.setTitle("Users's Liste")

	def retour(self):
		self.destroy()
		x = AccueilAdmin.AccueilAdmin(self.sessionLogin) 

	def addContenuSessionListe(self):
		compteur = 0
		self.Frame_Data = [[[] for i in range(0,2)] for j in range(0,len(allUsers()))]
		self.Label_Data = [[[] for i in range(0,2)] for j in range(0,len(allUsers()))]
		for row in allUsers():
			for i in range(0,2):
				self.Frame_Data[compteur][i] = tk.Frame(self.Colonne[i], bg= "white", height=30, width=180)
				self.Frame_Data[compteur][i].grid_propagate(0)
				self.Frame_Data[compteur][i].pack_propagate(0)
				self.Frame_Data[compteur][i].grid(row=compteur, column=i, padx=5, pady=0)
				self.Label_Data[compteur][i] = tk.Label(self.Frame_Data[compteur][i], text=row[i], fg="black",  font=("Courier", 12, 'bold'), bg="white")
				self.Label_Data[compteur][i].grid_propagate(0)
				self.Label_Data[compteur][i].pack_propagate(0)
				self.Label_Data[compteur][i].pack(pady=5, padx=10)
			compteur+=1

	def bouton_deux(self):
		self.destroy()
		x = AddAdmin.AddAdmin(self.sessionLogin)




#a voir pour les touches entrées: entree.bind('<Return>', evaluer)