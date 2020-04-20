#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
from tkinter.filedialog import askopenfilename
from imghdr import what
import os
import tkinter.font
#import AccueilJoueur 
#os.chdir("/home/aSara/Documents/TRAVAIL/ING2/PROJET/Travail/ELABORATION1/125061_FILES/REPTILE/TRAVAIL/IHM_sara")
#utiliser canvas: dessiner des formes

class InterfaceJeu(tk.Tk):
	"""Cette classe est la classe mere des interfaces du projet nous en avons besoin pour generer toutes les autres pages du projet - comme un patron."""
	def __init__(self):
		self.boite_fr = ["Champ.s vide.s", "Mots de passe incorrect.s", "Ajout réussi","Pseudo déjà utilisé", "Pseudo vide", "Connexion réussie", "Pseudo ou mot de passe incorrect(s)", "Inscription réussie", "Mot de passe vide"]
		self.boite_eng = ["Empty field.s", "Wrong password.s", "Added succesfully", "The username is already used", "The username field is empty", "Connection succeeded", "Incorrect username or password", "Registration succeeded", "The password field is empty"]

		tk.Tk.__init__(self) #appel du constructeur de la classe mere
		#textes
		
		self.text_interface_jeu_fr = ["English", "Fin Partie", "C'est parti !", "Vous gagnez !", "La machine gagne"]
		self.text_interface_jeu_eng = ["Français", "Exit", "Let's go !", "You won !", "You lost !"]
		self.text_jeu_comment_lose_fr = ["Courage !", "Une prochaine fois peut-être...", "Ne vous découragez pas !"]
		self.text_jeu_comment_lose_eng = ["You can do it !", "Maybe next time...", "Do not give up !"]
		self.text_jeu_comment_win_fr = ["Continuez comme ça !", "Bravo !", "Vous êtes sur la bonne voie !"]
		self.text_jeu_comment_win_eng = ["You are doing great !", "Well done!", "You are on the right track!"]
		self.text_jeu_machine_win_fr = ["Je suis trop fort !", "Impossible de me battre !", "Vous n'avez aucune chance."]
		self.text_jeu_machine_win_eng = ["I'm too strong !!", "Impossible to beat me !", "You have no chance."]
		self.text_jeu_machine_lose_fr = ["Argh!!! J'ai perdu!", "La prochaine fois je vous écrabouille !", "Vous m'avez eu !"]
		self.text_jeu_machine_lose_eng = ["Damn!! I lost!", "Next time I will smash you !", "You got me!"]

		self.text_commande_fr = ["Pierre", "Feuille", "Ciseaux"]
		self.text_commande_eng = ["Rock", "Paper", "Scissors"]
		#ajouter les textes: envoyer a Nedjma demande de traduction
		icon_chifoumi = tk.PhotoImage(file='images/icon_chifoumi.gif')
		self.rien = tk.PhotoImage(file='images/rien.gif')
		self.joueur_p = tk.PhotoImage(file='images/moi_p.gif')
		self.joueur_f = tk.PhotoImage(file='images/moi_f.gif')
		self.joueur_c = tk.PhotoImage(file='images/moi_c.gif')
		self.machine_p = tk.PhotoImage(file='images/ia_p.gif')
		self.machine_f = tk.PhotoImage(file='images/ia_f.gif')
		self.machine_c = tk.PhotoImage(file='images/ia_c.gif')
		self.coups_machine = [self.machine_p, self.machine_f, self.machine_c]
		self.coups_joueur = [self.joueur_p, self.joueur_f, self.joueur_c]
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
		self.addContenuJeu()
		self.mainloop()

	def setTitle(self, title):
		self.title(title)

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

	def traduction_page(self):
		pass
		#self.destroy()
		#self.choix_lang()
		#x = InterfaceJeu()

	def aff_text_lancement(self):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_interface_jeu_fr[2]
		else:
			return self.text_interface_jeu_eng[2]

	def aff_text_interface_jeu(self, entier):
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_interface_jeu_fr[entier]
		else:
			return self.text_interface_jeu_eng[entier]

	
	def aff_boite_dial(self, entier):
		fichier = open("langage.txt", "r")
		lang = fichier.read()
		fichier.close
		if lang == "fr":
			return self.boite_fr[entier]
		else:
			return self.boite_eng[entier]

	def aff_comande(self, entier):
		pass

	def finpartie(self):
		pass

	def aff_text_comment(self):
		pass

	def aff_text_machine(self):
		pass

	def aff_titre_jeu(self):
		pass
	
	def creation(self):
		#frame 1
		self.frame1 = tk.Frame(self, bg="#127fff", height=100, width=self.width)
		self.frame1.grid_propagate(0)
		self.frame1.pack_propagate(0)
		self.frame1.grid(row=0, column=0)

		self.frame1.frame_haut = tk.Frame(self.frame1, bg="#004ce5", height=30, width=self.width)
		self.frame1.frame_haut.pack(fill='both')
		self.frame1.bouton_trad=tk.Button(self.frame1, bg="white", fg="black", text=self.aff_text_interface_jeu(0), command=self.traduction_page, width=12)
		self.frame1.label_score=tk.Label(self.frame1, text=self.aff_titre_jeu(), bg="#127fff", fg="#00007f", font=("Courier", 35, 'bold'))
		self.frame1.bouton_menu=tk.Button(self.frame1, bg="white", fg="black", text=self.aff_text_interface_jeu(1), command=self.finpartie, width=12)
		self.frame1.bouton_menu.pack(side=tk.RIGHT, padx=77, pady=15, fill='both')
		self.frame1.bouton_trad.pack(side=tk.LEFT, padx=75, pady=15, fill='both')

		#self.frame1.bouton_trad.grid(row=0, column=0, pady=30, padx=140)
		#self.frame1.label_score.grid(row=0, column=1, pady=30, padx=140)
		#self.frame1.bouton_menu.grid(row=0, column=2, pady=30, padx=140)
		#self.frame1.bouton_trad.grid_propagate()

		#frame 2
		self.frame2 = tk.Frame(self, bg="white", height=600, width=self.width)
		self.frame2.grid_propagate(0)
		self.frame2.pack_propagate(0)
		self.frame2.grid(row=1, column=0)
		self.frame2.frame21 = tk.Frame(self.frame2, bg="white", height=50, width=1200) 
		self.frame2.frame22 = tk.Frame(self.frame2, bg="white", height=550, width=1200) 
		self.frame2.frame21.grid(row=0, column=0)
		self.frame2.frame22.grid(row=1, column=0)
		self.frame2.frame21.grid_propagate(0)
		self.frame2.frame21.pack_propagate(0)
		self.frame2.frame22.grid_propagate(0)
		self.frame2.frame22.pack_propagate(0)
		self.frame2.frame21.label_texte = tk.Label(self.frame2.frame21, text=self.aff_text_lancement(), bg="white", fg="#0040d9", font=("Courier", 20, 'bold'))
		self.frame2.frame21.label_texte.pack(fill='both', pady=10)

		#FRAME JEU SANS CAMERA




		#frame 3
		self.frame3 = tk.Frame(self, bg="#127fff", height=150, width=self.width)
		self.frame3.grid_propagate(0)
		self.frame3.pack_propagate(0)
		self.frame3.grid(row=2, column=0)

		self.frame3.frame_barre = tk.Frame(self.frame3, bg="#004ce5", height=50, width=self.width)
		self.frame3.frame_barre.pack(pady=25)


		# self.frame3.frame31 = tk.Frame(self.frame3, bg="white", height=100, width=600)
		# self.frame3.frame32 = tk.Frame(self.frame3, bg="white", height=100, width=600)
		# self.frame3.frame31.grid(row=0, column=0)
		# self.frame3.frame32.grid(row=0, column=1)
		# self.frame3.frame31.grid_propagate(0)
		# self.frame3.frame31.pack_propagate(0)
		# self.frame3.frame32.grid_propagate(0)
		# self.frame3.frame32.pack_propagate(0)

		# self.frame3.frame31.label_user = tk.Label(self.frame3.frame31, text=self.aff_joueur(), bg="white", fg="black", font=("helvetica", 20, 'bold', 'underline'))
		# self.frame3.frame32.label_machine = tk.Label(self.frame3.frame32, text=self.aff_machine(), bg="white", fg="black", font=("helvetica", 20, 'bold', 'underline'))
		# self.frame3.frame31.label_user.pack(fill='both', pady=0)
		# self.frame3.frame32.label_machine.pack(fill='both', pady=0)

	def aff_score(self):
		pass

	def addContenuJeu(self):
		pass

	def aff_machine(self):
		pass

	def aff_joueur(self):
		pass

if __name__ == "__main__":
	bienvenue = InterfaceJeu()
