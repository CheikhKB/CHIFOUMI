	#import des bibliotheques
import tkinter as tk
import string as st
from tkinter import messagebox
from random import randint, choice
from database import * 
import tkinter.font
import csv
from Historique import Historique
import AccueilAdmin

class HistoriqueAdmin(Historique):
	"""Classe hérite de la classe Historique"""

	def __init__(self, session):
		self.sessionLogin = session
		Historique.__init__(self)

	def aff_texte(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_historique_a_fr[entier]
		else:
			return self.text_historique_a_eng[entier]

	def traduction_page(self):
		self.destroy()
		self.choix_lang()
		x = HistoriqueAdmin(self.sessionLogin)

	def aff_texte_interface_information(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_interface_information_a_fr[entier]
		else:
			return self.text_interface_information_a_eng[entier]

	def setTitleHistorique(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			self.setTitle("Page Historique")
		else:
			self.setTitle("History Page")


	def bouton_deux(self):
		fichier = open("langage.txt", "r")
		lang = fichier.read()
		fichier.close()
		fname1 = "Data_"+self.sessionLogin+"_PARTIES.csv"
		fname2 = "Data_"+self.sessionLogin+"_COUPS.csv"
		file1 = open(fname1, "w")
		compteur = 0
		try:
			writer = csv.writer(file1)
			if lang == "fr":
				writer.writerow(('ID', 'Utilisateur', 'Date', 'Score Utilisateur', 'Score IA', 'Taux de réussite', 'Modèle'))
			else:
				writer.writerow(('ID', 'User', 'Date', 'User Score', 'AI Score', 'Sucess Rate', 'Model'))
			for row in allParties():
				writer.writerow(row)
		finally:
			file1.close()
			if lang=="fr":
				messagebox.showinfo("Infos","Le fichier "+fname1+" a été créé avec succès!")
			else:
				messagebox.showinfo("Infos","The file "+fname1+" has been created successfully!")
		file2 = open(fname2, "w")
		try:
			writer = csv.writer(file2)
			if lang == "fr":
				writer.writerow(('ID','num','Coup Joueur', 'Coup Machine'))
			else:
				writer.writerow(('ID','num' ,'Move Player', 'Move Computer'))
			for row in getCoups():
				#for row_coups in getParties(row[0]):
				writer.writerow((str(row[0]),str(row[1]) ,str(row[2]), str(row[3])))
		finally:
			file2.close()
			if lang=="fr":
				messagebox.showinfo("Infos","Le fichier "+fname2+" a été créé avec succès!")
			else:
				messagebox.showinfo("Infos","The file "+fname2+" has been created successfully!")



	def retour(self):
		self.destroy()
		x = AccueilAdmin.AccueilAdmin(self.sessionLogin)

	def aff_texte_data_admin(self, ligne, colonne):
		if colonne == 0:
			return str(allParties()[ligne][1])
		elif colonne ==1:
			return str(allParties()[ligne][3])
		elif colonne == 2:
			return str(allParties()[ligne][4])
		elif colonne == 3:
			return str(allParties()[ligne][5]) + "%"
		elif colonne == 4:
			if allParties()[ligne][6] == 1:
				return "Markov"
			else:
				return "RNN"

	def addContenuSessionHistorique(self):
		self.Colonne = []
		for i in range(0, 5):
			self.Colonne.append(tk.Frame(self.frame2.frame_general.frame_data, bg = "white", height=500, width=210))
			self.Colonne[i].grid_propagate(0)
			self.Colonne[i].pack_propagate(0)
			self.Colonne[i].grid(row=0, column=i, padx=10)
		#Ajout données
		compteur = 0
		self.Frame_Data = [[[] for i in range(0,5)] for j in range(0,len(allParties()))]
		self.Label_Data = [[[] for i in range(0,5)] for j in range(0,len(allParties()))]
		for row in allParties():
			for i in range(0,5):
				self.Frame_Data[compteur][i] = tk.Frame(self.Colonne[i], bg= "white", height=30, width=200)
				self.Frame_Data[compteur][i].grid_propagate(0)
				self.Frame_Data[compteur][i].pack_propagate(0)
				self.Frame_Data[compteur][i].grid(row=compteur, column=i, padx=5, pady=5)
				self.Label_Data[compteur][i] = tk.Label(self.Frame_Data[compteur][i], text=self.aff_texte_data_admin(compteur, i), fg="black",  font=("Courier", 10, 'bold'), bg="white")
				self.Label_Data[compteur][i].grid_propagate(0)
				self.Label_Data[compteur][i].pack_propagate(0)
				self.Label_Data[compteur][i].pack(pady=5, padx=5)
			compteur+=1
		self.mainloop()


#	historiqueadmin.addContenuSession()


