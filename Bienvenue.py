#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
import tkinter.font
from database import *
from Interface import Interface
import Login

class Bienvenue(Interface):
	"""Classe hérite de la classe Interface"""

	def __init__(self):
		Interface.__init__(self)

	def aff_texte(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_bienvenue_fr[entier]+" !"
		else:
			return self.text_bienvenue_eng[entier]+ " !"
			

	def traduction_page(self):
		self.destroy()
		self.choix_lang()
		x = Bienvenue()

	def setTitleBienvenue(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			self.setTitle("Bienvenue")
		else:
			self.setTitle("Welcome")

	def addContenuBienvenue(self):
		self.setTitleBienvenue()
		self.frame1.bouton_menu.pack_forget()
		#frame21
		self.frame2.frame21.canvas21 = tk.Canvas(self.frame2.frame21, bg="white", height=590, width=600, highlightthickness=0) 
		self.frame2.frame21.canvas21.grid(row=0, column=0)
		self.frame2.frame21.canvas21.create_image(46,14, anchor=tk.NW, image=self.choix_image_cfm("bienvenue"))
		self.frame2.frame22.frame221 = tk.Frame(self.frame2.frame22, bg="white", height=295, width=600) 
		self.frame2.frame22.frame222 = tk.Frame(self.frame2.frame22, bg="white", height=295, width=600)
		self.frame2.frame22.frame221.grid(row=0, column=0)
		self.frame2.frame22.frame222.grid(row=1, column=0)
		##self.frame2.frame22.create_text(0,0, anchor=tk.NW, activefill="blue", text="BIENVENUE SUR CHIFOUMI", font=self.police, justify='center')
		self.frame2.frame22.frame221.labeltitle = tk.Label(self.frame2.frame22.frame221, text=self.aff_texte(0), justify='center', font=('helvetica', 32, 'bold'), bg='white', fg='#0040d9')
		self.frame2.frame22.frame221.labeltitle.pack(fill='both', pady=50, padx=20)
		self.bind('<Return>', self.start)
		self.frame2.frame22.frame222.button = tk.Button(self.frame2.frame22.frame222, text=self.aff_texte(1), font=('helvetica', 20, 'bold'), bg='grey', fg='white', command=self.start)
		self.frame2.frame22.frame222.button.pack(side=tk.BOTTOM, padx=50, pady=50, fill='both')
		##self.frame2.frame22.button.grid(row=100, column=3, columnspan=1)
		self.mainloop()


	def start(self, envent=""):
		self.destroy()
		log = Login.Login()

	

if __name__ == "__main__":
	bienvenue = Bienvenue()

#a voir pour les touches entrées: entree.bind('<Return>', evaluer)