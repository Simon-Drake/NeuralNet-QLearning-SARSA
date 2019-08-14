import numpy as np
import random
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from features import *
from generate_game import *
from Q_values import *

size_board = 4

def H(x):
    return x>0

def BackProp(x_, out1_, Q_, Q_n, W1_, W2_, bias_W1_, bias_W2_, R, gamma, eta, action_chosen):
    x = x_.reshape(1,-1)
    out1 = out1_.reshape(1,-1)
    Q = Q_.reshape(1,-1)
    d_i = ((R + gamma * Q_n) - Q) * H(Q)* action_chosen
    d_j = np.dot(d_i, W2_.T) * H(out1)

    delta_weight_i = eta * np.dot(out1.T, d_i)
    delta_bias_i = eta * d_i[0]
    delta_weight_j = eta * np.dot(x.T, d_j)
    delta_bias_j = eta * d_j[0]

    W2_ += delta_weight_i
    bias_W2_ +=  delta_bias_i
    W1_ += delta_weight_j
    bias_W1_ +=  delta_bias_j

    return W1_, W2_, bias_W1_, bias_W2_

def main():

    """
    Generate a new game
    The function below generates a new chess board with King, Queen and Enemy King pieces randomly assigned so that they
    do not cause any threats to each other.
    s: a size_board x size_board matrix filled with zeros and three numbers:
    1 = location of the King
    2 = location of the Queen
    3 = location fo the Enemy King
    p_k2: 1x2 vector specifying the location of the Enemy King, the first number represents the row and the second
    number the colunm
    p_k1: same as p_k2 but for the King
    p_q1: same as p_k2 but for the Queen
    """
    s, p_k2, p_k1, p_q1 = generate_game(size_board)

    """
    Possible is for the Queen are the eight directions (down, up, right, left, up-right, down-left, up-left, 
    down-right) multiplied by the number of squares that the Queen can cover in one movement which equals the size of 
    the board - 1
    """
    possible_queen_a = (s.shape[0] - 1) * 8
    """
    Possible is for the King are the eight directions (down, up, right, left, up-right, down-left, up-left, 
    down-right)
    """
    possible_king_a = 8

    # Total number of is for Player 1 = is of King + is of Queen
    N_a = possible_king_a + possible_queen_a

    """
    Possible is of the King
    This functions returns the locations in the chessboard that the King can go
    dfK1: a size_board x size_board matrix filled with 0 and 1.
          1 = locations that the king can move to
    a_k1: a 8x1 vector specifying the allowed is for the King (marked with 1): 
          down, up, right, left, down-right, down-left, up-right, up-left
    """
    dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)

    """
    Possible is of the Queen
    Same as the above function but for the Queen. Here we have 8*(size_board-1) possible is as explained above
    """
    dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)

    """
    Possible is of the Enemy King
    Same as the above function but for the Enemy King. Here we have 8 possible is as explained above
    """
    dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

    """
    Compute the features
    x is a Nx1 vector computing a number of input features based on which the network should adapt its weights  
    with board size of 4x4 this N=50
    """
    x = features(p_q1, p_k1, p_k2, dfK2, s, check)

    """
    Initialization
    Define the size of the layers and initialization
    FILL THE CODE
    Define the network, the number of the nodes of the hidden layer should be 200, you should know the rest. The weights 
    should be initialised according to a uniform distribution and rescaled by the total number of connections between 
    the considered two layers. For instance, if you are initializing the weights between the input layer and the hidden 
    layer each weight should be divided by (n_input_layer x n_hidden_layer), where n_input_layer and n_hidden_layer 
    refer to the number of nodes in the input layer and the number of nodes in the hidden layer respectively. The biases
     should be initialized with zeros.
    """
    n_input_layer = 50  # Number of neurons of the input layer. TODO: Change this value
    n_hidden_layer = 200  # Number of neurons of the hidden layer
    n_output_layer = 32  # Number of neurons of the output layer. TODO: Change this value accordingly

    """
    TODO: Define the w weights between the input and the hidden layer and the w weights between the hidden layer and the 
    output layer according to the instructions. Define also the biases.
    """
    
    W1 = np.random.rand(n_input_layer, n_hidden_layer) / float(n_input_layer * n_hidden_layer)
    W2 = np.random.rand(n_hidden_layer, n_output_layer) / float(n_hidden_layer * n_output_layer)
    bias_W1 = np.zeros(n_hidden_layer)
    bias_W2 = np.zeros(n_output_layer)

    # YOUR CODES ENDS HERE

    # Network Parameters
    epsilon_0 = 0.2   #epsilon for the e-greedy policy
    beta = 0.00005      #epsilon discount factor
    gamma = 0.85      #SARSA Learning discount factor
    eta = 0.0035       #learning rate
    N_episodes = 100000  #Number of games, each game ends when we have a checkmate or a draw

    ###  Training Loop  ###

    # Directions: down, up, right, left, down-right, down-left, up-right, up-left
    # Each row specifies a direction, 
    # e.g. for down we need to add +1 to the current row and +0 to current column
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])
    
    # THE FOLLOWING VARIABLES COULD CONTAIN THE REWARDS PER EPISODE AND THE
    # NUMBER OF MOVES PER EPISODE, FILL THEM IN THE CODE ABOVE FOR THE
    # LEARNING. OTHER WAYS TO DO THIS ARE POSSIBLE, THIS IS A SUGGESTION ONLY.    

    R_save = np.zeros([N_episodes, 1])
    N_moves_save = np.zeros([N_episodes, 1])

    # END OF SUGGESTIONS
    
    c = 1 # counter for games
    moves = list()
    rewards = list()
    for n in range(N_episodes):
        next_computed = False

        if c % 1000 == 0:
            print(c)

        epsilon_f = epsilon_0 / (1 + beta * n) #psilon is discounting per iteration to have less probability to explore
        checkmate = 0  # 0 = not a checkmate, 1 = checkmate
        draw = 0  # 0 = not a draw, 1 = draw
        i = 1  # counter for movements

        # Generate a new game
        s, p_k2, p_k1, p_q1 = generate_game(size_board)

        # Possible is of the King
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible is of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible is of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)


        while checkmate == 0 and draw == 0:
            R = 0  # Reward

            # Player 1

            # is & allowed_is
            a = np.concatenate([np.array(a_q1), np.array(a_k1)])
            allowed_a = np.where(a > 0)[0]

            # Computing Features
            x = features(p_q1, p_k1, p_k2, dfK2, s, check)

            # FILL THE CODE 
            # Enter inside the Q_values function and fill it with your code.
            # You need to compute the Q values as output of your neural
            # network. You can change the input of the function by adding other
            # data, but the input of the function is suggested. 
            Q, out1 = Q_values(x, W1, W2, bias_W1, bias_W2)

            """
            YOUR CODE STARTS HERE
            
            FILL THE CODE
            Implement epsilon greedy policy by using the vector a and a_allowed vector: be careful that the i must
            be chosen from the a_allowed vector. The index of this i must be remapped to the index of the vector a,
            containing all the possible is. Create a vector calle da_agent that contains the index of the i 
            chosen. For instance, if a_allowed = [8, 16, 32] and you select the third i, a_agent=32 not 3.
            """

            # Implement Q-learning algorithm by setting this boolean to True. False implements SARSA.
            QLearn = False

            possible_moves = Q[allowed_a]

            eGreedy = int(np.random.rand() < epsilon_f)
            if eGreedy:
                ind = np.random.randint(len(possible_moves))
                a_agent = allowed_a[ind]
            else:
                ind = possible_moves.argmax()
                a_agent = allowed_a[ind]

            action_chosen = [0]*32
            action_chosen[a_agent] = 1


            #THE CODE ENDS HERE. 
            # Player 1 makes the i
            if a_agent < possible_queen_a:
                direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
                steps = a_agent - direction * (size_board - 1) + 1

                s[p_q1[0], p_q1[1]] = 0
                mov = map[direction, :] * steps
                s[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
                p_q1[0] = p_q1[0] + mov[0]
                p_q1[1] = p_q1[1] + mov[1]

            else:
                direction = a_agent - possible_queen_a
                steps = 1

                s[p_k1[0], p_k1[1]] = 0
                mov = map[direction, :] * steps
                s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
                p_k1[0] = p_k1[0] + mov[0]
                p_k1[1] = p_k1[1] + mov[1]

            # Compute the allowed is for the new position

            # Possible is of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible is of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible is of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

            # Player 2

            # Check for draw or checkmate
            if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
                # King 2 has no freedom and it is checked
                # Checkmate and collect reward
                checkmate = 1
                c += 1
                R = 1  # Reward for checkmate

                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the 
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for 
                the i made. You computed previously Q values in the Q_values function. Be careful: this is the last 
                iteration of the episode, the agent gave checkmate.
                """

                if next_computed:
                    qn = Q_next.max()
                    if QLearn:
                        W1, W2, bias_W1, bias_W2 =  BackProp(x, out1, Q, qn, W1, W2, bias_W1, bias_W2, R, gamma, eta, action_chosen)
                    else:
                        eGreedy = int(np.random.rand() < epsilon_f)
                        if eGreedy:
                            qn = random.choice(Q_next)
                            W1, W2, bias_W1, bias_W2 =  BackProp(x, out1, Q, qn, W1, W2, bias_W1, bias_W2, R, gamma, eta, action_chosen)
                        else:
                            qn = Q_next[allowed_a_next].max()
                            W1, W2, bias_W1, bias_W2 =  BackProp(x, out1, Q, qn, W1, W2, bias_W1, bias_W2, R, gamma, eta, action_chosen)
                            

                if checkmate:
                    break

            elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
                # King 2 has no freedom but it is not checked
                draw = 1
                c += 1
                R = 0.1

                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the 
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for 
                the i made. You computed previously Q values in the Q_values function. Be careful: this is the last 
                iteration of the episode, it is a draw.
                """
                if next_computed:
                    qn = Q_next.max()
                    if QLearn:
                        W1, W2, bias_W1, bias_W2 =  BackProp(x, out1, Q, qn, W1, W2, bias_W1, bias_W2, R, gamma, eta, action_chosen)
                    else:
                        eGreedy = int(np.random.rand() < epsilon_f)
                        if eGreedy:
                            qn = random.choice(Q_next)
                            W1, W2, bias_W1, bias_W2 =  BackProp(x, out1, Q, qn, W1, W2, bias_W1, bias_W2, R, gamma, eta, action_chosen)
                        else:
                            qn = Q_next[allowed_a_next].max()
                            W1, W2, bias_W1, bias_W2 =  BackProp(x, out1, Q, qn, W1, W2, bias_W1, bias_W2, R, gamma, eta, action_chosen)

                # YOUR CODE ENDS HERE

                if draw:
                    break

            else:
                # Move enemy King randomly to a safe location
                allowed_enemy_a = np.where(a_k2 > 0)[0]
                a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
                a_enemy = allowed_enemy_a[a_help]

                direction = a_enemy
                steps = 1

                s[p_k2[0], p_k2[1]] = 0
                mov = map[direction, :] * steps
                s[p_k2[0] + mov[0], p_k2[1] + mov[1]] = 3

                p_k2[0] = p_k2[0] + mov[0]
                p_k2[1] = p_k2[1] + mov[1]

            # Update the parameters

            # Possible is of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible is of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible is of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
            # Compute features
            x_next = features(p_q1, p_k1, p_k2, dfK2, s, check)
            # Compute Q-values for the discounted factor
            Q_next, _ = Q_values(x_next, W1, W2, bias_W1, bias_W2)


            next_computed = True

            """
            FILL THE CODE
            Update the parameters of your network by applying backpropagation and Q-learning. You need to use the 
            rectified linear function as activation function (see supplementary materials). Exploit the Q value for 
            the i made. You computed previously Q values in the Q_values function. Be careful: this is not the last 
            iteration of the episode, the match continues.
            """
            a = np.concatenate([np.array(a_q1), np.array(a_k1)])
            allowed_a_next = np.where(a > 0)[0]


            if not check or draw:
                qn = Q_next.max()
                if QLearn:
                    W1, W2, bias_W1, bias_W2 =  BackProp(x, out1, Q, qn, W1, W2, bias_W1, bias_W2, R, gamma, eta, action_chosen)
                else:
                    eGreedy = int(np.random.rand() < epsilon_f)
                    if eGreedy:
                        qn = random.choice(Q_next)
                        W1, W2, bias_W1, bias_W2 =  BackProp(x, out1, Q, qn, W1, W2, bias_W1, bias_W2, R, gamma, eta, action_chosen)
                    else:
                        qn = Q_next[allowed_a_next].max()
                        W1, W2, bias_W1, bias_W2 =  BackProp(x, out1, Q, qn, W1, W2, bias_W1, bias_W2, R, gamma, eta, action_chosen)

            # YOUR CODE ENDS HERE
            i += 1
        moves.append(i)
        rewards.append(R)

    # Comput moving averages over a sliding window
    mv_am = list()
    mv_rewards = list()
    for i, item in enumerate(rewards):
        if i > 250 and i < len(rewards) - 250:
            average_r = 0
            average_mo = 0
            for j in range(-250,250):
                average_mo += moves[i+j]
                average_r += rewards[i+j]
            average_mo /= 500
            average_r /= 500
            mv_am.append(average_mo)
            mv_rewards.append(average_r)
    f, axarr = plt.subplots(1,2, figsize=(20,10))

    axarr[0].plot(range(0,len(mv_am)), mv_am)
    axarr[0].set_title("Moving average: Moves")
    axarr[1].plot(range(0,len(mv_rewards)), mv_rewards)
    axarr[1].set_title("Moving average: Rewards")

    for i in range(0,2):
        plt.setp(axarr[i].get_xticklabels(), fontsize=16)
        plt.setp(axarr[i].get_yticklabels(), fontsize=16)
    plt.tight_layout()
    plt.show()

    # Print results to a file so that we can read and plot together
    result_string_moves = ""
    result_string_rewards = ""
    for i, item in enumerate(mv_am):
        result_string_moves += str(item) + ","
        result_string_rewards += str(mv_rewards[i]) + ","
    result_string_moves += "\n"
    result_string_rewards += "\n"

    with open("results.txt", "w") as f:
        f.write(result_string_moves + result_string_rewards)

if __name__ == '__main__':
    main()
