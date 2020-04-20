##coding: utf-8
#
##import sqlite3
#import pymysql
#from datetime import date
#from datetime import datetime
#global conn
#global cursor
#credentials = {'host': 'localhost', 'port':8090, 'user': 'root', 'password': '', 'db': 'RPS'}
#
#conn = pymysql.connect(**credentials)
#
#cursor = conn.cursor()
##
##cursor.execute("""
##CREATE TABLE IF NOT EXISTS utilisateurs(
##     login VARCHAR(10) PRIMARY KEY UNIQUE,
##     mot_de_passe VARCHAR(10),
##     role VARCHAR(10)
##)
##""")
##
##cursor.execute("""
##CREATE TABLE IF NOT EXISTS parties(
##     id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
##     login VARCHAR(10),
##     date VARCHAR(20),
##     score_joueur INTEGER,
##     score_IA INTEGER,
##     taux_reussite FLOAT(4),
##     algo INTEGER,
##     FOREIGN KEY(login) REFERENCES utilisateurs(login)
##)
##""")
##
##cursor.execute("""
##CREATE TABLE IF NOT EXISTS coups(
##     num INTEGER,
##     ref_partie INTEGER,
##     coup_joueur CHAR,
##     coup_IA CHAR,
##     PRIMARY KEY(num, ref_partie),
##     FOREIGN KEY(ref_partie) REFERENCES parties(id)
##)
##""")
#
#conn.commit()
#
#def loginNonUsed(login): #renvoie vrai si le login n'est pas utilisé
#     cursor.execute("""SELECT login FROM utilisateurs WHERE login="%s" """%(login,))
#     user = cursor.fetchone()
#     if user :
#          return False
#     return True
#
#def getUtilisateur(login, mdp): #verification connexion
#     cursor.execute("""SELECT login, mot_de_passe, role FROM utilisateurs WHERE login="%s" and mot_de_passe="%s" """%(login,mdp,))
#     user = cursor.fetchone()
#     return user
#
#def insererUtilisateur(login, pwd, role): #inscrire
#     cursor.execute("""
#     INSERT INTO utilisateurs(login, mot_de_passe, role) VALUES("%s","%s","%s")""" %(login, pwd,role))
#     conn.commit()
#
#def insererPartie(login, a): #au debut
#     now = datetime.now()
#     formatted_date = now.strftime('%Y-%m-%d')
#     cursor.execute("""
#     INSERT INTO parties(login, date, score_joueur, score_IA, taux_reussite, algo) VALUES("%s", "%s", "%s", "%s", "%s", "%s")"""%(login, formatted_date, 0, 0, 0, a))
#     conn.commit()
#     cursor.execute("""SELECT * FROM parties ORDER BY id DESC LIMIT 1""")
#     res = cursor.fetchone()
#     return res[0] #renvoie id partie enregistrée
#
#def updatePartie(rp, sj, sia, tx): #a la fin
#     cursor.execute("""UPDATE parties SET score_joueur = "%s", score_IA="%s", taux_reussite="%s" WHERE id="%s" """%(sj,sia,tx,rp,))
#     conn.commit()
#
#def insererCoup(num, rp, cj, cia): #apres chaque coup
#     cursor.execute("""
#     INSERT INTO coups(num, ref_partie, coup_joueur, coup_IA) VALUES("%s", "%s", "%s", "%s")"""%(num, rp, cj, cia))
#     conn.commit()
#
#def allUsers():
#     cursor.execute("""SELECT login, role FROM utilisateurs""")
#     rows = cursor.fetchall()
#     return rows
#
#def getParties(log):
#     cursor.execute("""SELECT * FROM parties WHERE login like "%s" """%(log))
#     rows = cursor.fetchall()
#     return rows
#
#def allParties():
#     cursor.execute("""SELECT * FROM parties """)
#     rows = cursor.fetchall()
#     return rows 
###############################################
import sqlite3
from datetime import date

global conn
global cursor

conn = sqlite3.connect('chifoumi.db')

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS utilisateurs(
	login VARCHAR(10) PRIMARY KEY UNIQUE,
	mot_de_passe VARCHAR(10),
	role VARCHAR(10)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS parties(
	id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
	login VARCHAR(10),
	date DATE,
	score_joueur INTEGER,
	score_IA INTEGER,
	taux_reussite FLOAT(4),
	algo INTEGER,
	FOREIGN KEY(login) REFERENCES utilisateurs(login)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS coups(
	num INTEGER,
	ref_partie INTEGER,
	coup_joueur CHAR,
	coup_IA CHAR,
	PRIMARY KEY(num, ref_partie),
	FOREIGN KEY(ref_partie) REFERENCES parties(id)
)
""")

conn.commit()

def loginNonUsed(login): #renvoie vrai si le login n'est pas utilisé
	cursor.execute("""SELECT login FROM utilisateurs WHERE login=? """, (login,))
	user = cursor.fetchone()
	if user :
		return False
	return True

def getUtilisateur(login, mdp): #verification connexion
	cursor.execute("""SELECT login, mot_de_passe, role FROM utilisateurs WHERE login=? and mot_de_passe=?""", (login,mdp,))
	user = cursor.fetchone()
	return user

def insererUtilisateur(login, pwd, role): #inscrire
	cursor.execute("""
	INSERT INTO utilisateurs(login, mot_de_passe, role) VALUES(?, ?, ?)""", (login, pwd,role))
	conn.commit()

def insererPartie(login, a): #au debut
	cursor.execute("""
	INSERT INTO parties(login, date, score_joueur, score_IA, taux_reussite, algo) VALUES(?, ?, ?, ?, ?, ?)""", (login, date.today(), 0, 0, 0, a))
	conn.commit()
	cursor.execute("""SELECT * FROM parties ORDER BY id DESC LIMIT 1""")
	res = cursor.fetchone()
	return res[0] #renvoie id partie enregistrée

def updatePartie(rp, sj, sia, tx): #a la fin
	cursor.execute("""UPDATE parties SET score_joueur = ?, score_IA=?, taux_reussite=? WHERE id=?""", (sj,sia,tx,rp,))
	conn.commit()

def insererCoup(num, rp, cj, cia): #apres chaque coup
	cursor.execute("""
	INSERT INTO coups(num, ref_partie, coup_joueur, coup_IA) VALUES(?, ?, ?, ?)""", (num, rp, cj, cia))
	conn.commit()

def allUsers():
	cursor.execute("""SELECT login, role FROM utilisateurs""")
	rows = cursor.fetchall()
	return rows

def getParties(log):
	cursor.execute("""SELECT * FROM parties WHERE login=?""", (log,))
	rows = cursor.fetchall()
	return rows

def allParties():
	cursor.execute("""SELECT * FROM parties """)
	rows = cursor.fetchall()
	return rows
#####################################################
def getCoups():
     cursor.execute("""SELECT ref_partie,num,coup_joueur, coup_IA  FROM coups  ORDER BY ref_partie ASC, num ASC""")
     rows = cursor.fetchall()
     return rows