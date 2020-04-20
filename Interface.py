#import des bibliotheques
import tkinter as tk
import string as st
from random import randint, choice
import tkinter.font

class Interface(tk.Tk):
	"""Cette classe est la classe mere des interfaces du projet nous en avons besoin pour generer toutes les autres pages du projet - comme un patron."""

	def __init__(self):
		tk.Tk.__init__(self) 
		icon_chifoumi = tk.PhotoImage(file='images/icon_chifoumi.gif')
		logo_SUPG = tk.PhotoImage(file='images/logo_SUPG.gif')
		logo_USPN = tk.PhotoImage(file='images/logo_USPN.gif')
		image_chifoumi_externe = tk.PhotoImage(file='images/Version2260.gif')
		logo_chifoumi = tk.PhotoImage(file='images/logo_chifoumi.gif')
		add = tk.PhotoImage(file='images/add.gif')
		login = tk.PhotoImage(file='images/login.gif')
		mains = tk.PhotoImage(file='images/mains.gif')
		rien = tk.PhotoImage(file='images/rien.gif')
		joueur_p = tk.PhotoImage(file='images/moi_p.gif')
		joueur_f = tk.PhotoImage(file='images/moi_f.gif')
		joueur_c = tk.PhotoImage(file='images/moi_c.gif')
		machine_p = tk.PhotoImage(file='images/ia_p.gif')
		machine_f = tk.PhotoImage(file='images/ia_f.gif')
		machine_c = tk.PhotoImage(file='images/ia_c.gif')
		logo_pierre = tk.PhotoImage(file='images/Pierre79.gif')
		logo_feuille = tk.PhotoImage(file='images/Feuille67.gif')
		logo_ciseau = tk.PhotoImage(file='images/Ciseau72.gif')
		bienvenue_image = tk.PhotoImage(file='images/Version70.gif')
		image_chifoumi_interne = tk.PhotoImage(file='images/Version1659.gif')
		image_chifoumi_admin = tk.PhotoImage(file='images/Version0890.gif')
		citation_fr = "La caractéristique des phénomènes dûs au hasard, c'est de dépendre de causes trop complexes pour que nous puissions les connaître toutes et les étudier\n\t\t\t\t- Emile Borel"
		notice_titre = "NOTICE"
		notice_titre_explication = "Explication du jeu"
		notice_titre_deroulement = "Déroulement du jeu"
		notice_titre_donnees = "Utilisation de vos données"
		notice_corps_explication = "Pierre-papier-ciseaux ou pierre-feuille-ciseaux est un jeu effectué avec les mains et qui oppose deux joueurs. Les deux joueurs choisissent simultanément un coup parmi 'pierre', 'feuille' et 'ciseaux'."
		notice_corps_explication_p = "- La pierre bat les ciseaux (en les émoussant)"
		notice_corps_explication_c = "- Les ciseaux battent la feuille (en la coupant)"
		notice_corps_explication_f = "- La feuille bat la pierre (en l'enveloppant)"
		notice_corps_deroulement = "Dans ce jeu, vous jouez contre l'ordinateur. Vous pouvez choisir de jouer contre une intelligence artificielle de type réseau de neurones ou contre une intelligence de type chaîne de Markov. Vous pouvez aussi activer votre caméra pour jouer vos coups."
		notice_corps_donnees = "Vos données de jeu sont sauvergardées dans une base de données, que vous pouvez consulter.\nNous utiliserons vos données de jeu pour les analyser et comprendre comment les intelligences créées réagissent et s'adaptent à vos parties."
		notice_fin = "Bonne chance, et amusez vous bien !"
		notice_p = "Pierre"
		notice_f = "Feuille"
		notice_c = "Ciseaux"
		notice_retour = "Retour"
		self.text_acceuiladmin_fr = ["Parties jouées", "Liste Utilisateurs", "Ajout Admin", "Déconnexion"]
		self.text_acceuiljoueur_fr =["Démarrer une partie", "Consulter résultats", "Règles du jeu", "Déconnexion"]
		self.text_bienvenue_fr = ["Bienvenue sur Chifoumi", "Commencer"]
		self.text_choix_fr = ["Choix Adversaire", "Markov", "Réseau de Neurones", "Retour"]
		self.text_interface_fr = ["English", "Menu Principal", citation_fr]
		self.text_notice_fr = [notice_titre, notice_titre_explication, notice_titre_deroulement, notice_titre_donnees, notice_corps_explication, notice_corps_explication_p, notice_corps_explication_f, notice_corps_explication_c, notice_corps_deroulement, notice_corps_donnees, notice_fin, notice_p, notice_f, notice_c, notice_retour]
		citation_eng = "Whatever human knowledge progress is, there will always be room for ignorance, hence for chance and probability.\n\t\t- Emile Borel"
		notice_titre_eng = "MANUAL"
		notice_titre_explication_eng = "Chifoumi's Game Explanation"
		notice_titre_deroulement_eng = "Chifoumi's Game Process"
		notice_titre_donnees_eng = "Use of your personal data"
		notice_corps_explication_eng = "Rock, paper, scissors or paper, scissors, rock, is a two player game. Each player chooses a sign, then both parties show their signs simultaneously. If they made the same choice, the game goes on. If not, the winner is determined according to the rules below :"
		notice_corps_explication_p_eng = "- the rock crushes the scissors"
		notice_corps_explication_c_eng = "- the scissors cut the paper"
		notice_corps_explication_f_eng = "- the paper wraps the rock"
		notice_corps_deroulement_eng = "In this game, you play against the computer. You can choose to play against a neural network or a Markov model. You can also use the camera to send your sign."
		notice_corps_donnees_eng = "Your datasets are saved in our database. You can consult this latter any time. We use your datasets in order to analyze them and understand how our artificial intelligence models react and above all adjust themselves to your choices during a game."
		notice_fin_eng = "Good luck, and we hope you will enjoy the game !"
		notice_p_eng = "Rock"
		notice_f_eng = "Paper"
		notice_c_eng = "Scissors"
		notice_retour_eng = "Go back"
		self.text_choix_c_fr = ["Choix du mode de jeu", "Clavier", "Caméra", "Retour"]
		self.text_choix_c_eng = ["Choose your game mode", "Keyboard", "Webcam", "Go back"]
		self.text_acceuiladmin_eng = ["Game History", "List of Users", "Add Admin", "Log out"]
		self.text_acceuiljoueur_eng =["Start a new game", "View results", "Rules of the game", "Log out"]
		self.text_bienvenue_eng = ["Welcome to Chifoumi", "Start"]
		self.text_choix_eng = ["Choose your opponent", "Markov", "Neural Network", "Go back"]
		self.text_interface_eng = ["Français", "Principal Menu", citation_eng]
		self.text_notice_eng = [notice_titre_eng, notice_titre_explication_eng, notice_titre_deroulement_eng, notice_titre_donnees_eng, notice_corps_explication_eng, notice_corps_explication_p_eng, notice_corps_explication_f_eng, notice_corps_explication_c_eng, notice_corps_deroulement_eng, notice_corps_donnees_eng, notice_fin_eng, notice_p_eng, notice_f_eng, notice_c_eng, notice_retour_eng]
		#interface_saisie
		self.text_inscrire_fr = ["INSCRIPTION", "Entrer pseudo", "Entrer mot de passe", "S'inscrire", "Déjà inscrite.e ?",  "Se connecter", "Confirmer mot de passe"]
		self.text_inscrire_eng = ["SIGN UP", "Enter your name", "Enter your password", "Sign up",  "Already registered ?", "Sign in", "Confirm your password"]
		self.text_login_fr = ["LOGIN", "Entrer pseudo", "Entrer mot de passe", "Se connecter", "Pas encore inscrite.e ?",  "S'inscrire", "        "]
		self.text_login_eng = ["SIGN IN", "Enter your name", "Enter your password", "Sign in",  "Have you not registered yet ?", "Sign up", "        "]
		self.text_addadmin_fr = ["AJOUT ADMIN", "Entrer pseudo", "Entrer mot de passe","Enregistrer", "               ",  "Retour","Confirmer mot de passe"]
		self.text_addadmin_eng = ["ADD ADMIN", "Enter your name", "Enter your password",  "Register", "               ", "Go back", "Confirm your password"]
		#boutons
		self.boite_fr = ["Champ.s vide.s", "Mots de passe incorrects", "Ajout réussi","Pseudo déjà utilisé", "Pseudo vide", "Connexion réussie", "Pseudo ou mot de passe incorrect(s)", "Inscription réussie", "Mot de passe vide"]
		self.boite_eng = ["Empty field.s", "Wrong passwords", "Added succesfully", "The username is already use", "The username field is empty", "Connection succeeded", "Incorrect username or password", "Registration succeeded", "The password field is empty"]
		self.police = tk.font.families()[1]
		self.police_italic =tk.font.Font(family='Helvetica', size=12, slant='italic')
		self.width=1200
		self.height=800
		self.images= [icon_chifoumi, logo_USPN, logo_SUPG, image_chifoumi_externe, logo_chifoumi, add, login, mains, rien, logo_pierre, logo_feuille, logo_ciseau, image_chifoumi_interne, image_chifoumi_admin, bienvenue_image]
		self.set_coups = [joueur_p, joueur_f, joueur_c, machine_p, machine_f, machine_c]
		width = self.winfo_screenwidth()
		height = self.winfo_screenheight()
		x = int(width / 2 - 1200 / 2)
		y = int(height / 2 - 800 / 2)
		str1 = "1200x800+"+ str(x) + "+" + str(y)
		self.geometry(str1)		
		self.resizable(width=False, height=False)
		#self.iconphoto(True, self.images[0]) #creation du logo/icon des interfaces
		self.setTitle("Interface mere") 
		self.setTitleAccueil()
		self.creation()
		self.addContenuInterfaceSaisie()
		self.addContenuAccueil()
		self.addContenuBienvenue()
		self.addContenuChoix()
		self.addContenuNotice()
		self.mainloop()


	def setTitle(self, title):
		self.title(title)

	def setTitleAccueil(self):
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

	def traduction_page(self):
		pass
		#self.choix_lang()
		#x = Interface()
		#x.sessionLogin = self.sessionLogin

	def aff_texte_interface(self, entier): 
		fichier = open("langage.txt","r")
		lang = fichier.read()
		fichier.close()
		if lang == "fr":
			return self.text_interface_fr[entier]
		else:
			return self.text_interface_eng[entier]


	def aff_boite_dial(self, entier):
		fichier = open("langage.txt", "r")
		lang = fichier.read()
		fichier.close
		if lang == "fr":
			return self.boite_fr[entier]
		else:
			return self.boite_eng[entier]


	def choix_image_cfm(self, image):
		if image == "externe":
			return self.images[3]
		else:
			if image == "admin":
				return self.images[13]
			else:
				if image == "interne":
					return self.images[12]
				else:#bienvenue
					return self.images[14]


	def creation(self):

#	dégradé de couleurs
	#   1266ff: 0
	#   0559f2: 5
	#   004ce5: 10
	#	0040d9: 15
		#frame 1
		self.frame1 = tk.Frame(self, bg="white", height=(self.height)/8, width=self.width)
		self.frame1.grid_propagate(0)
		self.frame1.pack_propagate(0)
		self.frame1.grid(row=0, column=0)
		#dbe5e3
		self.frame1.frame_couleur_un = tk.Frame(self.frame1, bg="#004ce5", height=25, width=self.width)
		self.frame1.frame_couleur_deux = tk.Frame(self.frame1, bg="#127fff", height=50, width=self.width)
		self.frame1.frame_couleur_trois = tk.Frame(self.frame1, bg="#004ce5", height=25, width=self.width)
		self.frame1.frame_couleur_un.grid(row=0, column=0)
		self.frame1.frame_couleur_deux.grid(row=1, column=0, pady=0)
		self.frame1.frame_couleur_trois.grid(row=2, column=0)
		self.frame1.frame_couleur_un.grid_propagate(0)
		self.frame1.frame_couleur_un.pack_propagate(0)
		self.frame1.frame_couleur_deux.grid_propagate(0)
		self.frame1.frame_couleur_deux.pack_propagate(0)
		self.frame1.frame_couleur_trois.grid_propagate(0)
		self.frame1.frame_couleur_trois.pack_propagate(0)

		# self.frame1.frame_couleur_un.degradre_un = tk.Frame(self.frame1.frame_couleur_un, bg="#0040d9", height=8, width=self.width)
		# self.frame1.frame_couleur_un.degradre_deux =  tk.Frame(self.frame1.frame_couleur_un, bg="#004ce5", height=8, width=self.width)
		# self.frame1.frame_couleur_un.degradre_trois =  tk.Frame(self.frame1.frame_couleur_un, bg="#0559f2", height=9, width=self.width)
		# self.frame1.frame_couleur_un.degradre_un.grid(row=0, column=0)
		# self.frame1.frame_couleur_un.degradre_deux.grid(row=1, column=0)
		# self.frame1.frame_couleur_un.degradre_trois.grid(row=2, column=0)
		# self.frame1.frame_couleur_un.degradre_un.grid_propagate(0)
		# self.frame1.frame_couleur_un.degradre_un.pack_propagate(0)
		# self.frame1.frame_couleur_un.degradre_deux.grid_propagate(0)
		# self.frame1.frame_couleur_un.degradre_deux.pack_propagate(0)
		# self.frame1.frame_couleur_un.degradre_trois.grid_propagate(0)
		# self.frame1.frame_couleur_un.degradre_trois.pack_propagate(0)

		self.frame1.bouton_trad=tk.Button(self.frame1.frame_couleur_deux, bg="white", fg="black", text=self.aff_texte_interface(0), command=self.traduction_page)
		self.frame1.bouton_menu=tk.Button(self.frame1.frame_couleur_deux, bg="white", fg="black", text=self.aff_texte_interface(1), command=self.menuGeneral)
		self.frame1.bouton_menu.pack(side=tk.RIGHT, padx=80, pady=10, fill='both')
		self.frame1.bouton_trad.pack(side=tk.LEFT, padx=80, pady=10, fill='both')
		#frame 2
		self.frame2 = tk.Frame(self, bg="white", height=(11/16)*self.height, width=self.width)
		self.frame2.grid_propagate(0)
		self.frame2.pack_propagate(0)
		self.frame2.grid(row=1, column=0)
		#frame 2
		self.frame2.frame21 = tk.Frame(self.frame2, bg="white", height=590, width=600) 
		self.frame2.frame22 = tk.Frame(self.frame2, bg="white", height=590, width=600) 
		##placement
		self.frame2.frame21.grid(row=0, column=1)
		self.frame2.frame22.grid(row=0, column=2)
		#frame 3
		self.frame3 = tk.Frame(self, bg="white", height=(3/16)*self.height, width=self.width)
		self.frame3.grid_propagate(0)
		self.frame3.pack_propagate(0)
		self.frame3.grid(row=2, column=0)
		self.frame3.frame_couleur = tk.Frame(self.frame3, bg="white", height=50, width=self.width)
		self.frame3.frame_couleur.frame_c_un = tk.Frame(self.frame3.frame_couleur, bg="white", height=15, width=self.width)
		self.frame3.frame_couleur.frame_c_deux = tk.Frame(self.frame3.frame_couleur, bg="#1266ff", height=25, width=self.width)
		self.frame3.frame_couleur.frame_c_trois = tk.Frame(self.frame3.frame_couleur, bg="white", height=10, width=self.width)
		self.frame3.frame_couleur.frame_c_un.grid(row=0, column=0)
		self.frame3.frame_couleur.frame_c_deux.grid(row=1, column=0)
		self.frame3.frame_couleur.frame_c_trois.grid(row=2, column=0)
		#couleur: 1266ff
		self.frame3.frame_logo = tk.Frame(self.frame3, bg="white", height=100, width=self.width)
		self.frame3.frame_couleur.grid(row=0, column=0)
		self.frame3.frame_logo.grid(row=1, column=0)
		self.frame3.frame_couleur.grid_propagate(0)
		self.frame3.frame_couleur.pack_propagate(0)
		self.frame3.frame_logo.grid_propagate(0)
		self.frame3.frame_logo.pack_propagate(0)
		
		self.frame3.frame_logo.canvas31 = tk.Canvas(self.frame3.frame_logo, bg="white", height=100, width=300, highlightthickness=0)
		self.frame3.frame_logo.canvas32 = tk.Canvas(self.frame3.frame_logo, bg="white", height=100, width=600, highlightthickness=0)
		self.frame3.frame_logo.canvas33 = tk.Canvas(self.frame3.frame_logo, bg="white", height=100, width=300, highlightthickness=0)
		self.frame3.frame_logo.canvas31.create_image(150,50,image=self.images[2])
		self.frame3.frame_logo.canvas32.create_text(60,0, anchor=tk.NW, activefill="#004ce5", text=self.aff_texte_interface(2), width=500, font=self.police_italic, justify='center', fill="#127fff")
		self.frame3.frame_logo.canvas33.create_image(150,50,image=self.images[1])
		#placement des widgets
		self.frame3.frame_logo.canvas31.grid(row=0, column=1, ipady=0) 
		self.frame3.frame_logo.canvas32.grid(row=0, column=2, columnspan=2, pady=0)
		self.frame3.frame_logo.canvas33.grid(row=0, column=4, ipady=0)

	def setTitleAccueil(self):
		pass

	def english(self):
		pass

	def menuGeneral(self):
		pass

	def addContenuInterfaceSaisie(self):
		pass

	def addContenuAccueil(self):
		pass

	def addContenuBienvenue(self):
		pass
	
	def addContenuChoix(self):
		pass


	def addContenuNotice(self):
		pass



