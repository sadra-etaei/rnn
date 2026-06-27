import numpy as np
import os 
import pickle


class NLayerRNN:
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.W_xh = []
        self.W_hh = []
        self.b_h = []
        
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            
            self.W_xh.append(np.random.randn(hidden_size, in_dim) * 0.01)
            self.W_hh.append(np.random.randn(hidden_size, hidden_size) * 0.01)
            self.b_h.append(np.zeros((hidden_size, 1)))
            
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        
        self.mW_xh = [np.zeros_like(w) for w in self.W_xh]
        self.mW_hh = [np.zeros_like(w) for w in self.W_hh]
        self.mb_h = [np.zeros_like(b) for b in self.b_h]
        self.mW_hy = np.zeros_like(self.W_hy)
        self.mb_y = np.zeros_like(self.b_y)

    def forward(self, inputs, h_prevs):
        self.hs = {-1: h_prevs} 
        self.inputs = inputs
        outputs = []
        
        for t, x in enumerate(inputs):
            self.hs[t] = []
            layer_input = x 
            
            for l in range(self.num_layers):
                h = np.tanh(np.dot(self.W_hh[l], self.hs[t-1][l]) + 
                            np.dot(self.W_xh[l], layer_input) + 
                            self.b_h[l])
                
                self.hs[t].append(h)
                layer_input = h #
                
            y_t = np.dot(self.W_hy, self.hs[t][-1]) + self.b_y
            outputs.append(y_t)
            
        return outputs, self.hs[len(inputs) - 1]
    
    def rmsprop_update(self,param, dparam, mem,decay=0.99,eps=1e-8,lr=0.01):
        mem *= decay
        mem += (1 - decay) * (dparam * dparam)
        param -= (lr / np.sqrt(mem + eps)) * dparam

    def backward(self, d_outputs, lr=0.01):
        dW_xh = [np.zeros_like(w) for w in self.W_xh]
        dW_hh = [np.zeros_like(w) for w in self.W_hh]
        db_h = [np.zeros_like(b) for b in self.b_h]
        
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)
        
        dh_next = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]
        
        for t in reversed(range(len(self.inputs))):
            dy_t = d_outputs[t]
            
            dW_hy += np.dot(dy_t, self.hs[t][-1].T)
            db_y += dy_t
            
            dh_down = np.dot(self.W_hy.T, dy_t)
            
            for l in reversed(range(self.num_layers)):
                
                dh = dh_down + dh_next[l]
                dtanh = (1 - self.hs[t][l] ** 2) * dh
                
                db_h[l] += dtanh
                
                layer_input = self.inputs[t] if l == 0 else self.hs[t][l-1]
                
                dW_xh[l] += np.dot(dtanh, layer_input.T)
                dW_hh[l] += np.dot(dtanh, self.hs[t-1][l].T)
                
                dh_next[l] = np.dot(self.W_hh[l].T, dtanh)
                np.clip(dh_next[l], -5, 5, out=dh_next[l])
                
                dh_down = np.dot(self.W_xh[l].T, dtanh) 

        for accum_list in [dW_xh, dW_hh, db_h]:
            for dparam in accum_list:
                np.clip(dparam, -5, 5, out=dparam)
        np.clip(dW_hy, -5, 5, out=dW_hy)
        np.clip(db_y, -5, 5, out=db_y)
        
        

        for l in range(self.num_layers):
            self.rmsprop_update(self.W_xh[l], dW_xh[l], self.mW_xh[l],lr=lr)
            self.rmsprop_update(self.W_hh[l], dW_hh[l], self.mW_hh[l],lr=lr)
            self.rmsprop_update(self.b_h[l], db_h[l], self.mb_h[l],lr=lr)
            
        self.rmsprop_update(self.W_hy, dW_hy, self.mW_hy,lr=lr)
        self.rmsprop_update(self.b_y, db_y, self.mb_y,lr=lr)



def sample(rnn, seed_id, n_chars, vocab_size):
    h_states = [np.zeros((rnn.hidden_size, 1)) for _ in range(rnn.num_layers)]
    
    x = np.zeros((vocab_size, 1))
    x[seed_id] = 1
    ids = []
    
    for t in range(n_chars):
        current_input = x
        new_h_states = []
        
        for l in range(rnn.num_layers):
            h = np.tanh(np.dot(rnn.W_hh[l], h_states[l]) + 
                        np.dot(rnn.W_xh[l], current_input) + 
                        rnn.b_h[l])
            
            new_h_states.append(h)
            current_input = h 
        
        h_states = new_h_states
        
        y = np.dot(rnn.W_hy, h_states[-1]) + rnn.b_y
        
        p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
        
        id = np.random.choice(range(vocab_size), p=p.ravel())
        ids.append(id)
        
        x = np.zeros((vocab_size, 1))
        x[id] = 1
        
    return ids


def train(model, text, seq_length=50, iterations=50000, lr=0.01):
    chars = list(set(text))
    vocab_size = len(chars)
    char2id = {ch: i for i, ch in enumerate(chars)}
    id2char = {i: ch for i, ch in enumerate(chars)}
    
    data_ptr = 0
    iteration = 0
    

    h_prevs = [np.zeros((model.hidden_size, 1)) for _ in range(model.num_layers)]
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length

    print(f"Starting training for {iterations} iterations...")

    while iteration < iterations:
        if data_ptr + seq_length + 1 > len(text):
            data_ptr = 0
            h_prevs = [np.zeros((model.hidden_size, 1)) for _ in range(model.num_layers)]

        inputs_id = [char2id[ch] for ch in text[data_ptr : data_ptr + seq_length]]
        targets_id = [char2id[ch] for ch in text[data_ptr + 1 : data_ptr + seq_length + 1]]

        x_vectors = []
        for ix in inputs_id:
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            x_vectors.append(x)

        y_outputs, h_prevs = model.forward(x_vectors, h_prevs)

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
            sample_ids = sample(model, inputs_id[0], 100, vocab_size)
            print("Sample: " + "".join(id2char[id] for id in sample_ids))


        data_ptr += seq_length
        iteration += 1

with open("D:/projects/nlp/RNN/shakespeare.txt","r") as f:
    text = f.read()
text = "".join([c for c in text if ord(c) < 128])
vocab_size = len(set(text))
model  = NLayerRNN(vocab_size,512,vocab_size,3)


train(model,text,100)