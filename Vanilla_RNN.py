import numpy as np 
import urllib.request

# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# urllib.request.urlretrieve(url, "shakespeare.txt")
# print("Shakespeare dataset downloaded successfully!")

class VanillaRNN:
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_xh = np.random.randn(self.hidden_size,self.input_size) * 0.01
        self.W_hh = np.random.randn(self.hidden_size,self.hidden_size) * 0.01
        self.W_hy = np.random.randn(self.output_size,self.hidden_size) * 0.01

        self.b_h = np.zeros((hidden_size,1))

        self.b_y = np.zeros((output_size,1))

        #adagrad

        self.mW_xh = np.zeros_like(self.W_xh)
        self.mW_hh = np.zeros_like(self.W_hh)
        self.mW_hy = np.zeros_like(self.W_hy)
        self.mb_h = np.zeros_like(self.b_h)
        self.mb_y = np.zeros_like(self.b_y)


    def forward(self,inputs,h_prev):
        self.h_states = {-1 : h_prev}
        self.inputs = inputs
        outputs = []
        for t, x in enumerate(inputs):
            h = np.tanh(np.dot(self.W_hh,self.h_states[t-1])+np.dot(self.W_xh,x)) + self.b_h
            self.h_states[t] = h
            y_t  = np.dot(self.W_hy,h) + self.b_y
            outputs.append(y_t)
        return outputs ,self.h_states[len(inputs)-1]
    

    def backward(self,d_outputs,lr=0.01,eps=1e-8):
        dW_xh,dW_hh,dW_hy = np.zeros_like(self.W_xh),np.zeros_like(self.W_hh),np.zeros_like(self.W_hy)
        db_h,db_y = np.zeros_like(self.b_h),np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size,1))


        for t in reversed(range(len((self.inputs)))):

            x_t = self.inputs[t]
            h_t = self.h_states[t]
            h_prev = self.h_states[t-1]
            d_yt= d_outputs[t]

            dW_hy += np.dot(d_yt,h_t.T)
            db_y += d_yt
            dh_next 

            dh_t = np.dot(self.W_hy.T,d_yt) + dh_next
            dtanh = (1 - h_t **2 ) * dh_t
            db_h += dtanh
            dW_xh += np.dot(dtanh,x_t.T)
            dW_hh += np.dot(dtanh,h_prev.T)
            dh_next = np.dot(self.W_hh.T,dtanh)

        for param, dparam, mem in zip(
            [self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y],
            [dW_xh, dW_hh, dW_hy, db_h, db_y],
            [self.mW_xh, self.mW_hh, self.mW_hy, self.mb_h, self.mb_y]
        ):
            mem += dparam * dparam # accumulate past squared gradients
            param -= (lr / np.sqrt(mem + eps)) * dparam


            # self.W_hh -= lr * dW_hh
            # self.W_xh -= lr * dW_xh
            # self.W_hy -= lr * dW_hy
            # self.b_h -= lr * db_h
            # self.b_y -= lr * db_y



def sample(rnn,seed_id,n_chars,vocab_size):
    h = np.zeros((rnn.hidden_size,1))
    x = np.zeros((vocab_size,1))
    x[seed_id] =1

    ids = []

    for t in range(n_chars):
        h = np.tanh(np.dot(rnn.W_xh, x) + np.dot(rnn.W_hh, h) + rnn.b_h)
        y = np.dot(rnn.W_hy,h)+rnn.b_y
        shifted_y = y - np.max(y)

        exp_y = np.exp(shifted_y)
        p = exp_y / np.sum(exp_y)

        id = np.random.choice(range(vocab_size),p=p.ravel())

        ids.append(id)

        x = np.zeros((vocab_size,1))
        x[id] = 1

    return ids


def train(seq_length):
    with open("D:/projects/nlp/RNN/shakespeare.txt","r") as f:
        text = f.read()


    chars = list(set(text))
    vocab_size = len(chars)
    char2id = {char:i for i ,char in enumerate(chars)}
    id2char = {i:char for i ,char in enumerate(chars)}

    rnn = VanillaRNN(vocab_size,512,vocab_size)

    data_ptr = 0
    iteration = 0
    
    print(f"Dataset loaded. Total chars: {len(text)}. Vocab size: {vocab_size}")
    print("Starting training... (Press Ctrl+C to stop at any time)\n")

    h_prev = np.zeros((rnn.hidden_size,1))

    while True :
        if data_ptr + seq_length + 1 > len(text):
            data_ptr = 0
            h_prev = np.zeros((rnn.hidden_size,1))


        inputs = [char2id[char] for char in text[data_ptr:data_ptr + seq_length]]
        targets = [char2id[char] for char in text[data_ptr+1:data_ptr + seq_length + 1]]

        x_vectors = []

        for id in inputs:
            x = np.zeros((vocab_size,1))
            x[id] = 1
            x_vectors.append(x)

        y_outputs,h_prev = rnn.forward(x_vectors,h_prev)

        loss = 0
        d_outputs = []

        for t in range(seq_length):
            y = y_outputs[t]
            shifted_y = y - np.max(y)

            exp_y = np.exp(shifted_y)
            p = exp_y / np.sum(exp_y)

            target_id = targets[t]

            loss += -np.log(p[target_id,0]+1e-15)

            dy = np.copy(p)

            dy[target_id]-=1

            d_outputs.append(dy)

        rnn.backward(d_outputs,0.01)

        
        if iteration % 1000 == 0:
            print(f"\nIteration {iteration} |  Loss: {loss:.4f}")
            
            sample_ids = sample(rnn, inputs[0], 100, vocab_size)
            generated_text = "".join(id2char[id] for id in sample_ids)
            print(f"---- Generated Sample ----\n{generated_text}\n--------------------------")
            
        data_ptr += seq_length
        iteration += 1



train(25)