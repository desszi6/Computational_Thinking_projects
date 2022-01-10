import numpy as np

def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences

def HMM_truestates_map(seq):
    mapping = {"N":0, "C":1, "R":2}
    return [mapping[symbol] for symbol in seq]

def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

#3 state matrices
init_probs_3_state = np.array(
    [0.00, 1.00, 0.00]
)

trans_probs_3_state = np.array([
    [0.90, 0.10, 0.00],
    [0.05, 0.90, 0.05],
    [0.00, 0.10, 0.90],
])

emission_probs_3_state = np.array([
    #   A     C     G     T
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
])

hmm_3_state = hmm(init_probs_3_state,
                  trans_probs_3_state,
                  emission_probs_3_state)

#7 state matrices
init_probs_7_state = np.array(
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]
)

trans_probs_7_state = np.array([
    [0.90, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
    [0.05, 0.00, 0.00, 0.90, 0.05, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],    
])

emission_probs_7_state = np.array([
    #   A     C     G     T
    [0.30, 0.25, 0.25, 0.20],
    [0.20, 0.35, 0.15, 0.30],
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
    [0.30, 0.20, 0.30, 0.20],
    [0.15, 0.30, 0.20, 0.35],
])

hmm_7_state = hmm(init_probs_7_state,
                  trans_probs_7_state,
                  emission_probs_7_state)

import math
def log(x):
    if x == 0:
        return float("-inf")
    else:
        return math.log(x)

def viterbi(obs, hmm):
    X = translate_observations_to_indices(obs)
    N = len(X)
    K = len(hmm.init_probs)
    V = np.zeros((K,N))

    init_probs=np.log(hmm.init_probs)
    trans_probs=np.log(hmm.trans_probs)
    emission_probs=np.log(hmm.emission_probs)

    for i in range(K):
        V[i][0]=init_probs[i]
    for i in range(1, N):
        if i%100000==0:
            print(i) 
        for n in range(K):
            E = emission_probs[n][X[i]]
            T = trans_probs[:,n]+V[:,i-1]
            V[n][i]= E + np.max(T)
    return V

def backtrack(V, hmm):
    assert len(V) == len(hmm.init_probs)
    
    init_probs = np.log(hmm.init_probs)
    trans_probs = np.log(hmm.trans_probs)
    emission_probs = np.log(hmm.emission_probs)
    
    N = len(V[0])
    i = N - 1
    k = np.argmax(V[:,N-1])
    o = []
    
    while i >= 0:
        o.append(str(k))
        k = np.argmax(V[:,i-1] + trans_probs[:,k] )
        i-=1
        if i % 100000 == 0: print(i)
    return ''.join(o[::-1])