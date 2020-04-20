import os
import tkinter as tk
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from tkinter import messagebox
import sys
import csv
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import string as st
from random import randint, choice
from database import *
import tkinter.font
from Historique import Historique
import AccueilJoueur

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))


class HistoriqueJoueur(Historique):
	"""Classe hérite de la classe Interface"""

	def __init__(self, session):
		self.sessionLogin = session
		Historique.__init__(self)


	def aff_texte(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_historique_j_fr[entier]
		else:
			return self.text_historique_j_eng[entier]

	def traduction_page(self):
		self.destroy()
		self.choix_lang()
		x = HistoriqueJoueur(self.sessionLogin)

	def aff_texte_interface_information(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_interface_information_j_fr[entier]
		else:
			return self.text_interface_information_j_eng[entier]

	def setTitleHistorique(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			self.setTitle("Historique des parties")
		else:
			self.setTitle("Parties played")

	def recuperation(self):
		fichier = open("langage.txt", "r")
		lang = fichier.read()
		fichier.close()
		fname1 = "DataGame_"+self.sessionLogin+"_PARTIES.csv"
		file1 = open(fname1, "w")
		compteur = 0
		try:
			writer = csv.writer(file1)
			if lang == "fr":
				writer.writerow(('ID', 'Utilisateur', 'Date', 'Score Utilisateur', 'Score IA', 'Taux de réussite', 'Modèle'))
			else:
				writer.writerow(('ID', 'User', 'Date', 'User Score', 'AI Score', 'Sucess Rate', 'Model'))
			for row in getParties(self.sessionLogin):
				writer.writerow(row)
		finally:
			file1.close()
		
	def bouton_deux(self):
		fichier = open("langage.txt", "r")
		lang = fichier.read()
		fichier.close()
		self.recuperation()
		try:
			filein = "DataGame_"+self.sessionLogin+"_PARTIES.csv"
			fileout = "DataGame_"+self.sessionLogin+".pdf"
			txt = open(filein, 'r').read()
			docpdf = SimpleDocTemplate(fileout, pagesize = A4)
			style = getSampleStyleSheet()
			style.add(ParagraphStyle(name='Chinese',
									 fontName='STSong-Light',
									 fontSize=12,
									 leading=14,
									 wordWrap = 'CJK'))
			style.add(ParagraphStyle(name='Titre',
									 fontName='Courier',
									 fontSize=30,
									 leading=14,
									 wordWrap = 'CJK', 
									 textColor="#0040d9"))
			style.add(ParagraphStyle(name='Sous Titre',
									 fontName='Courier',
									 fontSize=25,
									 leading=14,
									 wordWrap = 'CJK', 
									 textColor="#0040d9"))
			story = []
			story.append(Paragraph("CHIFOUMI    INTELLIGENT", style["Titre"]))
			story.append(Spacer(0, cm * 1.3))
			story.append(Paragraph("\n\nJoueur : " + self.sessionLogin + "\n\n\n", style["Sous Titre"]) )
			story.append(Spacer(0, cm * .8))
			paragraphs = txt.split("\n")
			for para in paragraphs:
				story.append(Paragraph(para, style["Chinese"]))
				story.append(Spacer(0, cm * .3))
			docpdf.build(story)
			if lang=="fr":
				messagebox.showinfo("Infos","Le fichier "+fileout+" a été créé avec succès!")
			else:
				messagebox.showinfo("Infos","The file "+fileout+" has been created successfully!")
			os.remove(filein)
		finally:
			pass

	def retour(self):
		self.destroy()
		x = AccueilJoueur.AccueilJoueur(self.sessionLogin)

	def aff_texte_data_joueur(self, ligne, colonne):
		if colonne == 0:
			return str(getParties(self.sessionLogin)[ligne][2])
		elif colonne ==1:
			return str(getParties(self.sessionLogin)[ligne][3])
		elif colonne == 2:
			return str(getParties(self.sessionLogin)[ligne][4])
		elif colonne == 3:
			return str(getParties(self.sessionLogin)[ligne][5]) + "%"
		elif colonne == 4:
			if getParties(self.sessionLogin)[ligne][6] == 1:
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
		compteur = 0
		self.Frame_Data = [[[] for i in range(0,5)] for j in range(0,len(getParties(self.sessionLogin)))]
		self.Label_Data = [[[] for i in range(0,5)] for j in range(0,len(getParties(self.sessionLogin)))]
		for row in getParties(self.sessionLogin):
			for i in range(0,5):
				self.Frame_Data[compteur][i] = tk.Frame(self.Colonne[i], bg= "white", height=30, width=200)
				self.Frame_Data[compteur][i].grid_propagate(0)
				self.Frame_Data[compteur][i].pack_propagate(0)
				self.Frame_Data[compteur][i].grid(row=compteur, column=i, padx=5, pady=0)
				self.Label_Data[compteur][i] = tk.Label(self.Frame_Data[compteur][i], text=self.aff_texte_data_joueur(compteur, i), fg="black",  font=("Courier", 10, 'bold'), bg="white")
				self.Label_Data[compteur][i].grid_propagate(0)
				self.Label_Data[compteur][i].pack_propagate(0)
				self.Label_Data[compteur][i].pack(pady=5, padx=5)
			compteur+=1
		self.mainloop()



