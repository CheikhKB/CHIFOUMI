from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

traduction = {"r":"p","p":"f","s":"c" }
list_coup  = {'0' : 'p', '1' : 'f', '2' : 'c'}
counter,predict_counter,model_wins,player_wins=0,0,0,0
model_move=None
model_next_move =None
predicted_next_move=None
score=None
game_seq=[]
model=load_model("models/best_9_2.h5")
#model.summary()
K.set_value(model.optimizer.lr, 0.001)

def score(a, b):
	if a == '0' and b == '1' or a == '1' and b == '2' or a == '2' and b == '0':
		return -1 #on perd
	elif a == b:
		return 0 #nul
	else:
		return 1 #on gagne

def make_input_vector(opponent_move=None, model_move=None):
    if opponent_move is None and model_move is None:
        return np.array([0,0,0,0,0,0],dtype=np.float64).reshape(1,1,6)
    elif opponent_move is None or model_move is None:
        raise ValueError
    move_ids = {"r": [1,0,0], "p": [0,1,0], "s": [0,0,1]}
    move= move_ids[model_move] + move_ids[opponent_move]
    return move
 

def rnn_prediction(x,model=model,game_seq_entrainement=3,epochs=15):
    counter_move={"r":"p","p":"s","s":"r"}
    moves={0:"r",1:"p",2:"s"}
    global counter
    global predict_counter
    global model_wins 
    global player_wins 
    global model_move   
    global model_next_move 
    global predicted_next_move 
    global game_seq
    global score
    if counter==0:
        prediction = model.predict(make_input_vector())
        predicted_next_move=moves[np.argmax(prediction)]
        model_next_move=counter_move[predicted_next_move]
    model_move=model_next_move
    predicted_move=predicted_next_move
    one_hot_move=make_input_vector(x, model_move)
    game_seq.append(np.array(one_hot_move,dtype=np.float64).reshape(1,1,6))

    if counter > 0:
        if len(game_seq) >= game_seq_entrainement+1:
            X=game_seq[len(game_seq)-(game_seq_entrainement+1):len(game_seq)-1]
            Y=game_seq[len(game_seq)-1]
            Y=Y[:,:,3:]
            model.fit(X,Y,epochs=epochs,batch_size=1,verbose=0)
            model.reset_states()
            prediction = model.predict(game_seq[len(game_seq)-game_seq_entrainement:])
        
        else:
            prediction = model.predict(game_seq)

        predicted_next_move=moves[np.argmax(prediction)]
        model_next_move=counter_move[predicted_next_move]
    
    
    if predicted_move==x:
        predict_counter+=1

    if model_move == counter_move[x]:
        model_wins += 1
        score=-1
    elif x == counter_move[model_move]:
        player_wins += 1
        score=1
    else:
        score=0
    dt=[x,model_move]
    
    counter+=1
    return dt


def getcounter():
    global counter
    return counter 
def getdata():
    global score
    global model_wins
    global player_wins
    global predict_counter
    global counter
    if counter >0:
        t=(predict_counter/counter)*100
        taux=round(t,2)
    else:
        taux=0
    dt=[score,model_wins,player_wins, taux]
    return dt

def renitialiser():
    global score
    global model_wins
    global player_wins
    global predict_counter
    global counter
    counter,predict_counter,model_wins,player_wins=0,0,0,0