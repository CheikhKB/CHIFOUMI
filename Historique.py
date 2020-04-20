#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
from database import *
import os
import tkinter.font
from InterfaceInformation import InterfaceInformation

class Historique(InterfaceInformation):
	"""Classe hérite de la classe Interface"""

	def __init__(self):
		self.text_interface_information_j_fr = ["English", "Exporter en PDF", "Retour"]
		self.text_interface_information_j_eng = ["Français","Export to PDF", "Go Back"]
		self.text_interface_information_a_fr = ["English", "Exporter en CSV", "Retour"]
		self.text_interface_information_a_eng = ["Français", "Export to CSV", "Go Back"]
		self.text_historique_j_fr= ["RÉSULTATS", "Date", "Ton Score", "Score IA", "Taux", "Modèle"]
		self.text_historique_j_eng = ["RESULTS", "Date", "Your Score", "AI's Score", "Success's Rate", "AI's Model"]
		self.text_historique_a_fr = ["PARTIES ENREGISTRÉES", "Joueur", "Score_J", "Score_IA", "Taux", "Modèle"]
		self.text_historique_a_eng = ["SAVEED GAMES", "User", "User's Score", "AI Score", "Success Rate", "AI's Model"]
		InterfaceInformation.__init__(self)

	def addContenuHistorique(self):

		#PLACEMENT DES TITRES DANS 5 COLONNES DU FRAME_LABEL (qui contient le titre des colonnes)
		self.frame2.frame_general.frame_label.frame21 = tk.Frame(self.frame2.frame_general.frame_label, bg="white", height=50, width=230)
		self.frame2.frame_general.frame_label.frame22 = tk.Frame(self.frame2.frame_general.frame_label, bg="white", height=50, width=230) 
		self.frame2.frame_general.frame_label.frame23 = tk.Frame(self.frame2.frame_general.frame_label, bg="white", height=50, width=230) 
		self.frame2.frame_general.frame_label.frame24 = tk.Frame(self.frame2.frame_general.frame_label, bg="white", height=50, width=230) 
		self.frame2.frame_general.frame_label.frame25 = tk.Frame(self.frame2.frame_general.frame_label, bg="white", height=50, width=230) 
		self.frame2.frame_general.frame_label.frame21.grid_propagate(0)
		self.frame2.frame_general.frame_label.frame21.pack_propagate(0)
		self.frame2.frame_general.frame_label.frame22.grid_propagate(0)
		self.frame2.frame_general.frame_label.frame22.pack_propagate(0)
		self.frame2.frame_general.frame_label.frame23.grid_propagate(0)
		self.frame2.frame_general.frame_label.frame23.pack_propagate(0)
		self.frame2.frame_general.frame_label.frame24.grid_propagate(0)
		self.frame2.frame_general.frame_label.frame24.pack_propagate(0)
		self.frame2.frame_general.frame_label.frame25.grid_propagate(0)
		self.frame2.frame_general.frame_label.frame25.pack_propagate(0)
		self.frame2.frame_general.frame_label.frame21.grid(row=0, column=0)
		self.frame2.frame_general.frame_label.frame22.grid(row=0, column=1)
		self.frame2.frame_general.frame_label.frame23.grid(row=0, column=2) 
		self.frame2.frame_general.frame_label.frame24.grid(row=0, column=3)
		self.frame2.frame_general.frame_label.frame25.grid(row=0, column=4)
		self.frame2.frame_general.frame_label.frame21.label_date = tk.Label(self.frame2.frame_general.frame_label.frame21, text=self.aff_texte(1), fg="#1266ff", font=("Courier", 16, 'bold', 'underline'), bg="white")
		self.frame2.frame_general.frame_label.frame22.label_score_j = tk.Label(self.frame2.frame_general.frame_label.frame22, text=self.aff_texte(2), fg="#1266ff", font=("Courier", 16, 'bold', 'underline'), bg="white")
		self.frame2.frame_general.frame_label.frame23.label_score_ia = tk.Label(self.frame2.frame_general.frame_label.frame23, text=self.aff_texte(3), fg="#1266ff", font=("Courier", 16, 'bold', 'underline'), bg="white")
		self.frame2.frame_general.frame_label.frame24.label_taux = tk.Label(self.frame2.frame_general.frame_label.frame24, text=self.aff_texte(4), fg="#1266ff", font=("Courier", 16, 'bold', 'underline'), bg="white")
		self.frame2.frame_general.frame_label.frame25.label_model = tk.Label(self.frame2.frame_general.frame_label.frame25, text=self.aff_texte(5), fg="#1266ff", font=("Courier", 16, 'bold', 'underline'), bg="white")
		self.frame2.frame_general.frame_label.frame21.label_date.pack(pady=10)
		self.frame2.frame_general.frame_label.frame22.label_score_j.pack(pady=10)
		self.frame2.frame_general.frame_label.frame23.label_score_ia.pack(pady=10)
		self.frame2.frame_general.frame_label.frame24.label_taux.pack(pady=10)
		self.frame2.frame_general.frame_label.frame25.label_model.pack(pady=10)
		self.setTitleHistorique()
		self.addContenuSessionHistorique()
		self.mainloop()

	def setTitleHistorique(self):
		pass

	def aff_texte(self, entier): 
		pass
	
	def addContenuSessionHistorique(self):
		pass