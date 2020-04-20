#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
from tkinter.filedialog import askopenfilename
from imghdr import what
import os
import tkinter.font
from tkinter import messagebox
from Interface import Interface
from database import *

class InterfaceSaisie(Interface):
	"""Classe hérite de la classe Interface"""

	def __init__(self):
		Interface.__init__(self)
		self.mainloop()


	def aff_image_logo(self):
		pass

	def menuGeneral(self):
		pass

	def aff_texte(self, entier): 
		pass

	def aff_image(self):
		pass

	def bouton_un(self, envent=""):
		pass

	def bouton_deux(self):
		pass

	def frame_trois(self):
		pass

	def traduction_page(self):
		pass
		#self.destroy()
		# self.choix_lang()
		# x = InterfaceSaisie()

	def addContenuInterfaceSaisie(self):
		#frame21
		self.frame2.frame21.canvas21 = tk.Canvas(self.frame2.frame21, bg="white", height=590, width=600, highlightthickness=0) 
		self.frame2.frame21.canvas21.grid(row=0, column=0)
		self.frame2.frame21.canvas21.grid_propagate(0)
		self.frame2.frame21.canvas21.pack_propagate(0)
		self.frame2.frame21.canvas21.create_image(46,14, anchor=tk.NW, image=self.aff_image())
		#on divise en 3 la frame 22
		self.frame2.frame22.frame221 = tk.Frame(self.frame2.frame22, bg="white", height=176, width=600) 
		self.frame2.frame22.frame222 = tk.Frame(self.frame2.frame22, bg="white", height=254, width=600)
		self.frame2.frame22.frame223 = tk.Frame(self.frame2.frame22, bg="white", height=160, width=600)
		self.frame2.frame22.frame221.grid(row=0, column=0)
		self.frame2.frame22.frame222.grid(row=1, column=0)
		self.frame2.frame22.frame223.grid(row=2, column=0)
		self.frame2.frame22.frame221.grid_propagate(0)
		self.frame2.frame22.frame221.pack_propagate(0)
		self.frame2.frame22.frame222.grid_propagate(0)
		self.frame2.frame22.frame222.pack_propagate(0)
		self.frame2.frame22.frame223.grid_propagate(0)
		self.frame2.frame22.frame223.pack_propagate(0)
		self.frame2.frame22.frame223.label_ins = tk.Label(self.frame2.frame22.frame223, text=self.aff_texte(4), justify='center', font=("Courier", 12, 'bold'), bg='white', fg='#1266ff')
		self.frame2.frame22.frame223.button = tk.Button(self.frame2.frame22.frame223, text=self.aff_texte(5), font='Courier 12', command=self.bouton_deux)
		self.frame2.frame22.frame223.label_ins.grid(row=0, column=0, padx=25, pady=20)
		self.frame2.frame22.frame223.button.grid(row=0, column=1, padx=25, pady=20)
		#Placement de l'image dans le frame221 et du titre
		self.frame2.frame22.frame221.frame_image  = tk.Frame(self.frame2.frame22.frame221, bg="white", height=140, width=150)
		self.frame2.frame22.frame221.frame_vide = tk.Frame(self.frame2.frame22.frame221, bg="white", height=140, width=450)
		self.frame2.frame22.frame221.frame_image.grid_propagate(0)
		self.frame2.frame22.frame221.frame_image.pack_propagate(0)
		self.frame2.frame22.frame221.frame_vide.grid_propagate(0)
		self.frame2.frame22.frame221.frame_vide.pack_propagate(0)
		self.frame2.frame22.frame221.frame_image.grid(row=0, column=0, pady=20)
		self.frame2.frame22.frame221.frame_vide.grid(row=0, column=1, pady=20)
		self.frame2.frame22.frame221.frame_image.canvas_login = tk.Canvas(self.frame2.frame22.frame221.frame_image, bg="white", height=140, width=150, highlightthickness=0)
		self.frame2.frame22.frame221.frame_image.canvas_login.create_image(0,0, anchor=tk.NW, image=self.aff_image_logo())
		self.frame2.frame22.frame221.frame_image.canvas_login.pack()
		self.frame2.frame22.frame221.frame_vide.label_login = tk.Label(self.frame2.frame22.frame221.frame_vide, text=self.aff_texte(0), justify='center', font=('helvetica', 32, 'bold'), bg='white', fg='#0040d9')
		self.frame2.frame22.frame221.frame_vide.label_login.grid(row=0, column=0, pady=40, padx=60)
		#Placement du formulaire dans la frame222
		self.frame2.frame22.frame222.frame2221 = tk.Frame(self.frame2.frame22.frame222, bg="white", height=170, width=600)
		self.frame2.frame22.frame222.frame2222 = tk.Frame(self.frame2.frame22.frame222, bg="white", height=84, width=600)
		self.frame2.frame22.frame222.frame2221.grid(row=0, column=0)
		self.frame2.frame22.frame222.frame2222.grid(row=1, column=0)
		self.frame2.frame22.frame222.frame2221.grid_propagate(0)
		self.frame2.frame22.frame222.frame2221.pack_propagate(0)
		self.frame2.frame22.frame222.frame2222.grid_propagate(0)
		self.frame2.frame22.frame222.frame2222.pack_propagate(0)
		#bouton de validation
		self.bind('<Return>',self.bouton_un)
		self.frame2.frame22.frame222.frame2222.button = tk.Button(self.frame2.frame22.frame222.frame2222, text=self.aff_texte(3), font='Courier 15 bold', command=self.bouton_un) 
		self.frame2.frame22.frame222.frame2222.button.grid(row=0, column=0, padx=350, pady=30)		
		self.frame2.frame22.frame222.frame2221.label_pseudo = tk.Label(self.frame2.frame22.frame222.frame2221, text=self.aff_texte(1), justify='center', font=("Courier", 12, 'bold'), bg='white', fg='#1266ff')
		self.frame2.frame22.frame222.frame2221.pseudo = tk.Entry(self.frame2.frame22.frame222.frame2221, font='Courier 12')
		self.frame2.frame22.frame222.frame2221.label_ps = tk.Label(self.frame2.frame22.frame222.frame2221, text=self.aff_texte(2), justify='center', font=("Courier", 12, 'bold'), bg='white', fg='#1266ff')
		self.frame2.frame22.frame222.frame2221.password = tk.Entry(self.frame2.frame22.frame222.frame2221,show='*', font='Courier 12')
		self.frame2.frame22.frame222.frame2221.label_pseudo.grid(row=0, column=0, pady=15, padx=15) 
		self.frame2.frame22.frame222.frame2221.pseudo.grid(row=0, column=1, pady=15, padx=15)
		self.frame2.frame22.frame222.frame2221.label_ps.grid(row=1, column=0, pady=15, padx=15)
		self.frame2.frame22.frame222.frame2221.password.grid(row=1, column=1, pady=15, padx=15)
		self.addContenuLogin()
		self.addContenuInscrire()
		self.addContenuAddAdmin()
		self.mainloop()

	def addContenuLogin(self):
		pass

	def addContenuInscrire(self):
		pass

	def addContenuAddAdmin(self):
		pass
