import numpy as np
import pickle
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rmsprop(param, dparam, mem,decay_rate=0.99,eps=1e-8,lr=0.01):
            mem *= decay_rate
            mem += (1 - decay_rate) * (dparam * dparam)
            param -= (lr / np.sqrt(mem + eps)) * dparam

class LSTM:
    def __init__(self,input_size,hidden_size,output_size,n_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.W = []
        self.b = []


        for l in range(n_layers):
            in_dim = input_size if l == 0 else hidden_size
            self.W.append(np.random.randn( 4 * hidden_size , hidden_size + in_dim) * 0.01)
            self.b.append(np.zeros((self.hidden_size * 4 ,1)))

            # initializing forget gate bias to 1
            self.b[l][0:hidden_size,0] = 1.0

        self.W_hy = np.random.randn(output_size,hidden_size) * 0.01
        self.b_y = np.zeros((output_size,1))


        # RMSPROP memory :
        self.mW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.mW_hy = np.zeros_like(self.W_hy)
        self.mb_y = np.zeros_like(self.b_y)


    def forward(self,inputs,h_prevs,c_prevs):
        self.inputs=inputs
        self.h_states,self.c_states,self.gates = {},{},{}

        self.h_states[-1],self.c_states[-1] = h_prevs,c_prevs

        outputs = []

        for t , x in enumerate(inputs):
            self.h_states[t],self.c_states[t] = [],[]
            layer_input = x 

            for l in range(self.n_layers):
                z = np.vstack((self.h_states[t-1][l],layer_input))

                gates = np.dot(self.W[l],z) + self.b[l]

                f = sigmoid(gates[0:self.hidden_size])
                i = sigmoid(gates[self.hidden_size:2*self.hidden_size])
                g = np.tanh(gates[2*self.hidden_size:3*self.hidden_size])
                o = sigmoid(gates[3*self.hidden_size:4*self.hidden_size])


                c_t = f * self.c_states[t-1][l] + i * g
                h_t = o * np.tanh(c_t)

                self.h_states[t].append(h_t)
                self.c_states[t].append(c_t)
                self.gates[t,l] = (f,i,g,o,z)

                layer_input = h_t

            outputs.append(np.dot(self.W_hy,self.h_states[t][-1])+self.b_y)

        return outputs,self.h_states[len(inputs)-1],self.c_states[len(inputs)-1]



    def backward(self,d_outputs,lr=0.01):
        dW = [np.zeros_like(w) for w in self.W]
        db = [np.zeros_like(b) for b in self.b]
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        dh_next = [np.zeros((self.hidden_size,1)) for _ in range(self.n_layers)]
        dc_next = [np.zeros((self.hidden_size,1)) for _ in range(self.n_layers)]


        for t in reversed(range(len(self.inputs))):
            dy = d_outputs[t]
            dW_hy += np.dot(dy,self.h_states[t][-1].T)
            db_y += dy
            dh_down = np.dot(self.W_hy.T,dy)

            for l in reversed(range(self.n_layers)):
                dh = dh_down + dh_next[l]

                f, i, g, o, z = self.gates[t, l]

                c_t, c_prev = self.c_states[t][l], self.c_states[t-1][l]

                dtanh_c = np.tanh(c_t)
                do = dh * dtanh_c * (o * (1 - o))
                dc = dh * o * (1 - dtanh_c**2) + dc_next[l]
                
                df = dc * c_prev * (f * (1 - f))
                di = dc * g * (i * (1 - i))
                dg = dc * i * (1 - g**2)
                
                d_gates = np.vstack((df, di, dg, do))
                dW[l] += np.dot(d_gates, z.T)
                db[l] += d_gates


                # Split the gradient: Top goes back in time, Bottom goes down in depth
                dz = np.dot(self.W[l].T, d_gates)
                dh_next[l] = dz[:self.hidden_size]     # Gradient for h_prev
                dc_next[l] = dc * f                    # Gradient for c_prev
                
                # The gradient for the layer below becomes the new dh_down
                dh_down = dz[self.hidden_size:]

        for dparam in dW + db + [dW_hy,db_y]:
            np.clip(dparam,-1,1,dparam)

        for l in range(self.n_layers):
            rmsprop(self.W[l],dW[l],self.mW[l],lr=lr)
            rmsprop(self.b[l],db[l],self.mb[l],lr=lr)

        rmsprop(self.W_hy,dW_hy,self.mW_hy,lr=lr)
        rmsprop(self.b_y,db_y,self.mb_y,lr=lr)






def sample(rnn, seed_id, n_chars, vocab_size,temperature):
    h_states = [np.zeros((rnn.hidden_size, 1)) for _ in range(rnn.n_layers)]
    c_states = [np.zeros((rnn.hidden_size, 1)) for _ in range(rnn.n_layers)]
    
    x = np.zeros((vocab_size, 1))
    x[seed_id] = 1
    ids = []
    
    for t in range(n_chars):
        current_input = x
        new_h_states = []
        new_c_states = []
        
        for l in range(rnn.n_layers):
            z = np.vstack((h_states[l], current_input))
            gates = np.dot(rnn.W[l], z) + rnn.b[l]
            
            f = sigmoid(gates[0:rnn.hidden_size])
            i = sigmoid(gates[rnn.hidden_size:2*rnn.hidden_size])
            g = np.tanh(gates[2*rnn.hidden_size:3*rnn.hidden_size])
            o = sigmoid(gates[3*rnn.hidden_size:4*rnn.hidden_size])
            
            c_t = f * c_states[l] + i * g
            h_t = o * np.tanh(c_t)
            
            new_h_states.append(h_t)
            new_c_states.append(c_t)
            current_input = h_t
            
        h_states = new_h_states
        c_states = new_c_states
        
        y = np.dot(rnn.W_hy, h_states[-1]) + rnn.b_y
        y = y/temperature
        p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
        
        id = np.random.choice(range(vocab_size), p=p.ravel())
        ids.append(id)
        
        x = np.zeros((vocab_size, 1))
        x[id] = 1
        
    return ids

def train(model, text, seq_length=50, iterations=50000, lr=0.01,sampling_temperature=0.7):
    chars = list(set(text))
    vocab_size = len(chars)
    char2id = {ch: i for i, ch in enumerate(chars)}
    id2char = {i: ch for i, ch in enumerate(chars)}
    
    data_ptr = 0
    iteration = 0
    


    h_prevs = [np.zeros((model.hidden_size, 1)) for _ in range(model.n_layers)]
    c_prevs = [np.zeros((model.hidden_size, 1)) for _ in range(model.n_layers)]

    smooth_loss = -np.log(1.0 / vocab_size) * seq_length

    print(f"Starting training for {iterations} iterations...")

    while iteration < iterations:
        if data_ptr + seq_length + 1 > len(text):
            data_ptr = 0
            c_prevs = [np.zeros((model.hidden_size, 1)) for _ in range(model.n_layers)]
            h_prevs = [np.zeros((model.hidden_size, 1)) for _ in range(model.n_layers)]

        inputs_id = [char2id[ch] for ch in text[data_ptr : data_ptr + seq_length]]
        targets_id = [char2id[ch] for ch in text[data_ptr + 1 : data_ptr + seq_length + 1]]

        x_vectors = []
        for ix in inputs_id:
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            x_vectors.append(x)

        y_outputs, h_prevs,c_prevs = model.forward(x_vectors, h_prevs,c_prevs)

        loss = 0
        d_outputs = []
        for t in range(seq_length):
            y = y_outputs[t]
            p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
            
            loss += -np.log(p[targets_id[t], 0] + 1e-15)
            
            dy = np.copy(p)
            dy[targets_id[t]] -= 1
            d_outputs.append(dy)

        model.backward(d_outputs, lr=lr)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        if iteration % 1000 == 0:
            print(f"Iter: {iteration} | Loss: {smooth_loss:.4f}")
            sample_ids = sample(model, inputs_id[0], 100, vocab_size,sampling_temperature)
            print("Sample: " + "".join(id2char[id] for id in sample_ids))


        data_ptr += seq_length
        iteration += 1





with open("D:/projects/nlp/RNN/shakespeare.txt","r") as f:
    text = f.read()
text = "".join([c for c in text if ord(c) < 128])

text = text[:50000]
vocab_size = len(set(text))

model = LSTM(vocab_size,128,vocab_size,2)

train(model,text,iterations=51000,lr=0.001)