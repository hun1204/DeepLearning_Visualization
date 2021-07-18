# DeepLearning_Visualization_Basic
These are basic code what I learned from 'ICT INOVATION Visualization Course'

1. ANN(Artificial Neural Network) 인공신경망
인간의 신경구조를 복잡한 스위치들이 연결된 네트워크로 표현할 수 있다고 1943년도에 제안됨.
Perceptron -> MLP(Multi-layer perceptron)에서 파라미터 개수가 많아 적절한 Weight, Bias 학습이 어려워짐
Backpropagation으로 연산후의 값과 실제 값의 오차를 후방으로 보내 많은 노드를 가져도 학습이 가능하게 됨.
Relu, Backpropagation으로 과적합 문제가 해결되고 GPU를 사용한 행렬연산의 가속화로 딥러닝이 부활함
딥러닝은 feed forward(순전파/앞먹임)를 통해 머신러닝, 딥러닝의 네트워크를 실행하며 학습이 가능해짐.
feed forward 예시 - input->weights->sum net input function->activation(sigmoid)->output
