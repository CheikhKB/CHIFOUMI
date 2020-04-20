#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
from tkinter.filedialog import askopenfilename
from imghdr import what
import os
import tkinter.font

#os.chdir("/home/aSara/Documents/TRAVAIL/ING2/PROJET/Travail/ELABORATION1/125061_FILES/REPTILE/TRAVAIL/IHM_sara")
#utiliser canvas: dessiner des formes

class InterfaceInformation(tk.Tk):
	"""Cette classe est la classe mere des interfaces du projet nous en avons besoin pour generer toutes les autres pages du projet - comme un patron."""
	def __init__(self):
		self.boite_fr = ["Champ.s vide.s", "Mots de passe incorrects", "Ajout réussi","Pseudo déjà utilisé", "Pseudo vide", "Connexion réussie", "Pseudo ou mot de passe incorrect(s)", "Inscription réussie", "Mot de passe vide"]
		self.boite_eng = ["Empty field.s", "Wrong passwords", "Added succesfully", "The username is already use", "The username field is empty", "Connection succeeded", "Incorrect username or password", "Registration succeeded", "The password field is empty"]

		tk.Tk.__init__(self) #appel du constructeur de la classe mere
		
		icon_chifoumi = tk.PhotoImage(file='images/icon_chifoumi.gif')
		#self.iconphoto(True, self.images[0]) #creation du logo/icon des interfaces
		self.police = tk.font.families()[1]
		self.police_italic =tk.font.Font(family='Helvetica', size=12, slant='italic')
		self.width=1200
		self.height=800
		width = self.winfo_screenwidth()
		height = self.winfo_screenheight()
		x = int(width / 2 - 1200 / 2)
		y = int(height / 2 - 800 / 2)
		str1 = "1200x800+"+ str(x) + "+" + str(y)
		self.geometry(str1)		
		self.resizable(width=False, height=False)
		self.creation()
		self.addContenuHistorique()
		self.addContenuListe()
		self.mainloop()

	def setTitle(self, title):
		self.title(title)

	def addContenuSessionListe():
		pass

	def aff_texte_interface_information(self, entier): 
		pass

	def aff_texte(self, entier):
		pass

	def traduction_page(self):
		pass

	def retour(self):
		pass

	def bouton_deux(self):
		pass

	def addContenuHistorique(self):
		pass

	def addContenuListe(self):
		pass

	def choix_lang(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang=="" or lang=="fr":
			fichier = open("langage.txt","w")
			fichier.write("eng")
			fichier.close()
		else:
			fichier = open("langage.txt","w")
			fichier.write("fr")
			fichier.close()

	def aff_boite_dial(self, entier):
		fichier = open("langage.txt", "r")
		lang = fichier.read()
		fichier.close
		if lang == "fr":
			return self.boite_fr[entier]
		else:
			return self.boite_eng[entier]


	def creation(self):
		#frame 1
		self.frame1 = tk.Frame(self, bg="white", height=(self.height)/8, width=self.width)
		self.frame1.grid_propagate(0)
		self.frame1.pack_propagate(0)
		self.frame1.grid(row=0, column=0)
		#dbe5e3
		self.frame1.frame_couleur_un = tk.Frame(self.frame1, bg="#dbe5e3", height=25, width=self.width)
		self.frame1.frame_couleur_deux = tk.Frame(self.frame1, bg="#1266ff", height=55, width=self.width)
		self.frame1.frame_couleur_trois = tk.Frame(self.frame1, bg="#dbe5e3", height=20, width=self.width)
		self.frame1.frame_couleur_un.grid(row=0, column=0)
		self.frame1.frame_couleur_deux.grid(row=1, column=0)
		self.frame1.frame_couleur_trois.grid(row=2, column=0)
		self.frame1.frame_couleur_un.grid_propagate(0)
		self.frame1.frame_couleur_un.pack_propagate(0)
		self.frame1.frame_couleur_deux.grid_propagate(0)
		self.frame1.frame_couleur_deux.pack_propagate(0)
		self.frame1.frame_couleur_trois.grid_propagate(0)
		self.frame1.frame_couleur_trois.pack_propagate(0)

		self.frame1.bouton_trad=tk.Button(self.frame1.frame_couleur_deux, bg="white", fg="black", text=self.aff_texte_interface_information(0), command=self.traduction_page, width=12)
		self.frame1.bouton_exportpdf=tk.Button(self.frame1.frame_couleur_deux, bg="white", fg="black", text=self.aff_texte_interface_information(1), command=self.bouton_deux, width=12)
		self.frame1.bouton_menu=tk.Button(self.frame1.frame_couleur_deux, bg="white", fg="black", text=self.aff_texte_interface_information(2), command=self.retour, width=12)
		self.frame1.bouton_trad.grid(row=0, column=0, pady=15, padx=140)
		self.frame1.bouton_exportpdf.grid(row=0, column=1, pady=15, padx=140)
		self.frame1.bouton_menu.grid(row=0, column=2, pady=15, padx=140)

		
		#frame 2
		self.frame2 = tk.Frame(self, bg="#dbe5e3", height=(14/16)*self.height, width=self.width)
		self.frame2.grid_propagate(0)
		self.frame2.pack_propagate(0)
		self.frame2.grid(row=1, column=0)


		self.frame2.frame_titre = tk.Frame(self.frame2, bg="white", height=90, width=1150)
		self.frame2.frame_titre.grid(row=0, column=0, padx=25)
		self.frame2.frame_titre.grid_propagate(0)
		self.frame2.frame_titre.pack_propagate(0)
		self.frame2.frame_titre.label_titre = tk.Label(self.frame2.frame_titre, text=self.aff_texte(0), fg="#0040d9", font=("Courier", 30, 'bold', 'underline'), bg="white")
		self.frame2.frame_titre.label_titre.pack(pady=25)
		
		self.frame2.frame_general = tk.Frame(self.frame2, bg="white", height=550, width=1150)
		self.frame2.frame_general.grid(row=1, column=0, padx=25)
		self.frame2.frame_general.grid_propagate(0)
		self.frame2.frame_general.pack_propagate(0)
		self.frame2.frame_general.frame_label = tk.Frame(self.frame2.frame_general, bg="white", height=50, width=1150)
		self.frame2.frame_general.frame_data = tk.Frame(self.frame2.frame_general, bg="white", height=500, width=1150)#attention, changé en frame
		self.frame2.frame_general.frame_label.grid_propagate(0)
		self.frame2.frame_general.frame_label.pack_propagate(0)
		self.frame2.frame_general.frame_data.grid_propagate(0)
		self.frame2.frame_general.frame_data.pack_propagate(0)
		self.frame2.frame_general.frame_label.grid(row=0, column=0)
		self.frame2.frame_general.frame_data.grid(row=1, column=0)

		self.frame2.frame_bottom = tk.Frame(self.frame2, bg="red", height=60, width=self.width)
		self.frame2.frame_bottom.grid(row=2, column=0)
		self.frame2.frame_bottom.grid_propagate(0)
		self.frame2.frame_bottom.pack_propagate(0)
		self.frame2.frame_bottom.frame_bottom_gris = tk.Frame(self.frame2.frame_bottom, bg="#dbe5e3", height=20, width=self.width)
		self.frame2.frame_bottom.frame_bottom_bleu = tk.Frame(self.frame2.frame_bottom, bg="#1266ff", height=40, width=self.width)
		self.frame2.frame_bottom.frame_bottom_gris.grid(row=0, column=0)
		self.frame2.frame_bottom.frame_bottom_bleu.grid(row=1, column=0)
		self.frame2.frame_bottom.frame_bottom_gris.grid_propagate(0)
		self.frame2.frame_bottom.frame_bottom_gris.pack_propagate(0)
		self.frame2.frame_bottom.frame_bottom_bleu.grid_propagate(0)
		self.frame2.frame_bottom.frame_bottom_bleu.pack_propagate(0)


