#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
from database import *
import textwrap as tw
import tkinter.font
from Interface import Interface
import AccueilJoueur
import Bienvenue

class Notice(Interface):
	"""Classe hérite de la classe InterfaceSans"""

	def __init__(self, session):
		self.sessionLogin = session
		Interface.__init__(self)

	def aff_texte(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_notice_fr[entier]
		else:
			return self.text_notice_eng[entier]

	def traduction_page(self):
		self.destroy()
		self.choix_lang()
		x = Notice(self.sessionLogin)
	
	def addContenuSession(self):
		pass

	def setTitleNotice(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.setTitle("Notice")
		else:
			return self.setTitle("Manual Page")

	def addContenuNotice(self):
		self.frame1.bouton_menu.pack_forget()
		self.setTitleNotice()

		#frame 21
		self.frame2.frame21.frame211 = tk.Frame(self.frame2.frame21, bg="white", height=120, width=600)
		self.frame2.frame21.frame212 = tk.Frame(self.frame2.frame21, bg="white", height=270, width=600)
		self.frame2.frame21.frame213 = tk.Frame(self.frame2.frame21, bg="white", height=200, width=600)
		self.frame2.frame21.frame211.grid(row=0, column=0)
		self.frame2.frame21.frame212.grid(row=1, column=0)
		self.frame2.frame21.frame213.grid(row=2, column=0)
		#frame 211
		# #Placement du titre dans le frame221
		self.frame2.frame21.frame211.label_titre = tk.Label(self.frame2.frame21.frame211, text=self.aff_texte(0), font=("helvetica", 40, 'bold'), bg='white', fg='#0040d9')
		self.frame2.frame21.frame211.label_titre.pack(side=tk.RIGHT, fill='both', ipady=10)
		#frame 212
		# placement des texte partis gauche
		self.frame2.frame21.frame212.label_titre_explication = tk.Label(self.frame2.frame21.frame212, text=self.aff_texte(1), justify='center', font=("helvetica", 17, 'bold', 'underline'), bg='white', fg='#004ce5')
		self.frame2.frame21.frame212.label_corps_explication = tk.Label(self.frame2.frame21.frame212, text=self.aff_texte(4), justify='left', font=("courier", 12, 'bold'), bg='white', fg='#1266ff', wraplength=550)
		self.frame2.frame21.frame212.label_corps_explication_p = tk.Label(self.frame2.frame21.frame212, text=self.aff_texte(5), font=("courier", 12, 'bold'), bg='white', fg='#1266ff')
		self.frame2.frame21.frame212.label_corps_explication_f = tk.Label(self.frame2.frame21.frame212, text=self.aff_texte(6), font=("courier", 12, 'bold'), bg='white', fg='#1266ff')
		self.frame2.frame21.frame212.label_corps_explication_c = tk.Label(self.frame2.frame21.frame212, text=self.aff_texte(7), font=("courier", 12, 'bold'), bg='white', fg='#1266ff')
		self.frame2.frame21.frame212.label_titre_explication.grid(row=0, column=0, sticky='nw', padx=15)
		self.frame2.frame21.frame212.label_corps_explication.grid(row=1, column=0, padx=25, pady=5)
		self.frame2.frame21.frame212.label_corps_explication_p.grid(row=2, column=0, sticky='nw', padx=35)
		self.frame2.frame21.frame212.label_corps_explication_f.grid(row=3, column=0, sticky='nw', padx=35)
		self.frame2.frame21.frame212.label_corps_explication_c.grid(row=4, column=0, sticky='nw', padx=35)
		#frame213 
		#placement des images
		self.frame2.frame21.frame213.label_p = tk.Label(self.frame2.frame21.frame213, text=self.aff_texte(11), justify='center', font=("courier", 12, 'bold'), bg='white', fg='#1266ff', wraplength=87)
		self.frame2.frame21.frame213.label_f = tk.Label(self.frame2.frame21.frame213, text=self.aff_texte(12), justify='center', font=("courier", 12, 'bold'), bg='white', fg='#1266ff', wraplength=87)
		self.frame2.frame21.frame213.label_c = tk.Label(self.frame2.frame21.frame213, text=self.aff_texte(13), justify='center', font=("courier", 12, 'bold'), bg='white', fg='#1266ff', wraplength=86)
		self.frame2.frame21.frame213.canvas_p = tk.Canvas(self.frame2.frame21.frame213,  bg="white", height=200, width=110, highlightthickness=0)
		self.frame2.frame21.frame213.canvas_f = tk.Canvas(self.frame2.frame21.frame213,  bg="white", height=200, width=100, highlightthickness=0)
		self.frame2.frame21.frame213.canvas_c = tk.Canvas(self.frame2.frame21.frame213,  bg="white", height=200, width=130, highlightthickness=0)
		self.frame2.frame21.frame213.canvas_p.create_image(0,0, anchor=tk.NW, image=self.images[9])
		self.frame2.frame21.frame213.canvas_f.create_image(0,0, anchor=tk.NW, image=self.images[10])
		self.frame2.frame21.frame213.canvas_c.create_image(0,0, anchor=tk.NW, image=self.images[11])
		self.frame2.frame21.frame213.canvas_p.grid(row=0, column=1, padx=15, pady=5)
		self.frame2.frame21.frame213.canvas_f.grid(row=0, column=2, padx=15, pady=5)
		self.frame2.frame21.frame213.canvas_c.grid(row=0, column=3, padx=15, pady=5)	
		self.frame2.frame21.frame213.label_p.grid(row=1, column=1, padx=15, pady=0)
		self.frame2.frame21.frame213.label_f.grid(row=1, column=2, padx=15, pady=0)
		self.frame2.frame21.frame213.label_c.grid(row=1, column=3, padx=15, pady=0)
		#frame 22
		self.frame2.frame22.frame221 = tk.Frame(self.frame2.frame22, bg="white", height=250, width=600)
		self.frame2.frame22.frame222 = tk.Frame(self.frame2.frame22, bg="white", height=240, width=600)
		self.frame2.frame22.frame223 = tk.Frame(self.frame2.frame22, bg="white", height=100, width=600)
		self.frame2.frame22.frame221.grid(row=0, column=0, padx=15, pady=10)
		self.frame2.frame22.frame222.grid(row=1, column=0, padx=15, pady=10)
		self.frame2.frame22.frame223.grid(row=2, column=0, padx=15, pady=10)
		#frame221
		# placement des texte partis droite
		self.frame2.frame22.frame221.label_titre_deroulement = tk.Label(self.frame2.frame22.frame221, text=self.aff_texte(2), justify='center', font=("helvetica", 17, 'bold', 'underline'), bg='white', fg='#004ce5')
		self.frame2.frame22.frame221.label_corps_explication = tk.Label(self.frame2.frame22.frame221, text=self.aff_texte(8), justify='left', font=("courier", 12, 'bold'), bg='white', fg='#1266ff', wraplength=530)
		self.frame2.frame22.frame221.label_titre_deroulement.grid(row=0, column=0, sticky='nw', padx=5)
		self.frame2.frame22.frame221.label_corps_explication.grid(row=1, column=0, pady=10, padx=20)
		#frame222
		#placement du texte données
		self.frame2.frame22.frame222.label_titre_donnees = tk.Label(self.frame2.frame22.frame222, text=self.aff_texte(3), justify='center', font=("helvetica", 17, 'bold', 'underline'), bg='white', fg='#004ce5')
		self.frame2.frame22.frame222.label_corps_donnees = tk.Label(self.frame2.frame22.frame222, text=self.aff_texte(9), justify='left', font=("courier", 12, 'bold'), bg='white', fg='#1266ff', wraplength=530)
		self.frame2.frame22.frame222.label_titre_donnees.grid(row=0, column=0, sticky='nw', padx=5)
		self.frame2.frame22.frame222.label_corps_donnees.grid(row=1, column=0, pady=10, padx=20)
		#frame223
		#placement du bouton menu
		self.frame2.frame22.frame223.label_text_fin = tk.Label(self.frame2.frame22.frame223, text=self.aff_texte(10), justify='center', font=("Courier", 17, 'italic'), bg='white', fg='#004ce5', wraplength=520)
		self.frame2.frame22.frame223.bouton = tk.Button(self.frame2.frame22.frame223, text=self.aff_texte(14), justify='center', font=("helvetica", 17, 'bold'), bg='white', fg='#1266ff', command=self.retour)
		self.frame2.frame22.frame223.label_text_fin.grid(row=0, column=0, padx=10, pady=20)
		self.frame2.frame22.frame223.bouton.grid(row=1, column=0, sticky='ne', padx=20, pady=20)
		self.mainloop()

	def retour(self):
		self.destroy()
		x = AccueilJoueur.AccueilJoueur(self.sessionLogin)


	
