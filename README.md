# Sequence Models From Scratch: $N$-Layer RNN vs. LSTM in NumPy

An implementation of Multi-Layer Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks built completely from scratch using **only Python and NumPy** for character level language modeling . 


This whole project is motivated by Andrej Karpathys char-rnn blog post and repository with the difference that it is implemented in pure python 
I trained the implemented models on the Shakspeare dataset in order to compare the results achieved by karpathy and myself,and also because it seemed fun 
at first I implemented a 1 layer rnn too see if it was able to learn the data , as expected it was not ,but the same went for the multi-layer RNN, it was  only after implementing the multi layer lstm  
that the model started learning the data and  making sense
the problem with the Vanillar RNN`s was vanishing gradients , due to this phenomenon the loss kept pallatueing and the models failed to learn 
but the lstm model after a couple of hours of training was able to make some sense ,of course unfortunately I was not able to compete with Andrej karpathys generations 



---

## 🚀 The Core 


1. **The Vanilla RNN Boundary:** Implements standard recurrent connections. Highlights the practical challenges of **vanishing and exploding gradients** over extended time dependencies.
2. **The $N$-Layer LSTM Evolution:** Resolves gradient issues by introducing explicit cell states and gating mechanisms. Implemented completely with full matrix-concatenation speedups, deep stacking support ($N$-layers), and numerical stability mechanisms.

---


 ## some of the generated samples : 


## iteration 0 :      
kTEbCbw-K-ksKddbeRK'FRcdxdrxKKdx'xxxdxxxxxxxxxxxxxxTxxxxxxuuanFFnFFFFFFFFFnFFFFFFFFFFFFFFFFFFFFFFFFF


## iteration 50000:

Second Citizen:
My morrow, my honour means is of my boon,
Gentlemen on, then hove the world, my house,
And you m





And how it is a best thou find for the double a more us
her boots the father on 



he officer
Can a comfort the demand in the duke?

MARIANA:
O crave you to you: not many mourning him


o make it show
Here not on faster in this tomowity done,
And like me our sin your honours or makes;




o was a noble
Walk him all of our ebundance, and he his kindly;
The peoples of Coriolanus forest 


love,
When he will we are sooners of who is the world
Than my be down to thee of the women
in 



oe you, vortenery;
I think him god in the garner of mallies come.

VIRGILIA:
The vormers is that how






## the lstm model was trained with 2 layers and 512 hidden dimension for a couple of hours to achieve these results


