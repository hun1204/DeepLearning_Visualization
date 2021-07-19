# DeepLearning_Visualization_Basic
These are basic code what I learned from 'ICT INOVATION Visualization Course'

### 1. ANN(Artificial Neural Network) 인공신경망에서 딥러닝까지
* 인간의 신경구조를 복잡한 스위치들이 연결된 네트워크로 표현할 수 있다고 1943년도에 제안됨.
* Perceptron -> MLP(Multi-layer perceptron)에서 파라미터 개수가 많아 적절한 Weight, Bias 학습이 어려워짐
* Backpropagation으로 연산후의 값과 실제 값의 오차를 후방으로 보내 많은 노드를 가져도 학습이 가능하게 됨.
* Backpropagation은 미분이 가능한 함수만 적용이 가능하며 체인룰을 적용해 구한다.
* 딥러닝은 feed forward(순전파/앞먹임)를 통해 머신러닝, 딥러닝의 네트워크를 실행하며 학습이 가능함.
* feed forward 예시 - input->weights,bias->sum net input function->activation(sigmoid)->output
* feed forward의 오차역전파 과정에서 Vanishing Gradient Problem 때문에 한계에 직면하고 딥러닝 대신 한동안 SVM이 쓰임.
* 다시 Relu, dropout으로 과적합, 기울기 소실 문제가 해결되고 GPU를 사용한 행렬연산의 가속화로 딥러닝이 부활함.
* Convolution&Pooling이 나오면서 비약적으로 발전! OBJECT DETECTION, SEGMETATION, Reinforcement Learning, GAN등 다양한 형태의 딥러닝으로 발전함.

![1  feed forward](https://user-images.githubusercontent.com/43362034/126163548-a66b3a69-725f-4571-a3d6-3d27088ad068.JPG)

### 2. Image Processing
* 영상 처리의 분야/geometric transform, enhancement, restoration, compression, object recognition
* object recognition / 영상 내의 존재하는 얼굴의 위치를 찾고 얼굴에 해당하는 사람을 인식하는 기술
* enhancement / 저화질 영상(이미지 또는 비디오)을 고화질로 개선하는 방법(디지털 카메라, HDTV)
* 의료 영상 처리, OCR등 문서처리, Imgae Captioning 이미지의 텍스트를 설명하는 기술 등 다양하다.
* Pixel(Picture elements in digital images)은 영상의 기본 단위
* Bitmap - two dimentional array of pixel values, Image Resolution - The number of pixels in a digital image(640*480, etc)
* 1-Bit Images(0,1), 8-Bit Gray-Level Images(0~255), 24-Bit Color Images(256*256*256)

### 3. Tensorflow
* 구글에서 개발,공개한 딥러닝/머신러닝을 위한 오픈소스 라이브러리/torch, pytorch도 있으며 각자 장점 존재.
* Tensor를 흘려보내며(flow) 데이터를 처리 / 여기서 Tensor는 임의의 차원을 갖는 배열이다.
* Graph-node와edge의 조합 /Node-Operation,Variable,Constant/ Edge-노드간의 연결
* 그래프를 생성해서 실행시키는것이 텐서플로우이다. 현재는 2.x 버젼이 나왔으며 eager excution이 가능한것이 특징이다.
* TensorBoard를 이용해 편리한 시각화가 가능하며 전세계적으로 가장 활발한 커뮤니티를 가지고있다.


### 4. CNN
* Convolution(합성곱), Channel, Filter, Stride, Feature map, Activation Map, Padding, Pooling
* Convolution 일정 영역에 필터를 곱해 값을 모두 더하는 방식이다.
* 같은데이터로 묶어진 채널을 입력값으로 받아 각 채널마다 필터를 적용해 Feature map을 생성해준다.
* CNN에서 출력데이터가 줄어드는것을 막기 위해 Padding을 사용하며 Pooling layer를 통해 Feature map의 크기를 줄이거나 특정 데이터를 강조함.
* CNN은 이미지 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식하기 위해 쓰인다.
* 추출한 이미지의 특징을 모으고 강화하기 위해 Poolinglayer를 사용하며 필터를 공유 파라미터로 사용하기 때문에 ANN에 비해 파라미터가 매우 적음.

![2  CNN](https://user-images.githubusercontent.com/43362034/126166764-8eac5e3f-f0a3-413f-8910-14ee216e8749.JPG)

