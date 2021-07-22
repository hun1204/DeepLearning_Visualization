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

### 5. Pretrained Model
* 작은 이미지 데이터셋에 딥러닝을 적용하는 매우 효과적인 방법으로 pretrained model을 사용한다.
* 사전 훈련된 네트워크는 대량의 데이터셋에서 미리 훈련되어 저장된 네트워크이다.(ImageNet은 1.4백만 개의 레이블, 1000개의 클래스로 학습)
* 이렇게 학습된 특성을 다른 문제에 적용할 수 있는 이런 유연성이 딥러닝의 핵심 장점.
* VGG, ResNet, Inception, Inception-ResNet, Xception 등 다양한 모델들이 존재.
* VGG를 예시로 Input -> Trained convolutional base -> Trained classifier -> Prediction으로 이루어져있으며 
* 합성곱 층으로 학습된 표현은 더 일반적이여서 재사용이 용이하고 분류기는 모델이 훈련된 클래스 집합에 특화되어 있어 재사용이 힘들다.
* 따라서 상황에 맞게 Trained classifier를 사용 안하거나 새로운 Classifier로 교체하여 사용할 수도 있다.

![pretrained model](https://user-images.githubusercontent.com/43362034/126261302-29380ba0-c6f3-45c0-ac34-cbe13e437f82.PNG)


### 6. Data Agmentation
* 수백만개에 이르는 매개변수를 제대로 훈련하기 위해서 많은 데이터가 필요하며 그 품질이 우수해야 한다.
* 이 때 매개변수를 훈련할 충분한 학습 데이터를 확보하지 않으면 모델의 성능을 저해하는 과적합(overfitting)이 발생한다.
* 양질의 데이터(의료 영상 데이터는 의사가 직접 데이터셋을 구축해야 하므로 비용이 많이 듦)를 확보하기란 쉽지 않다.
* 이럴 때 딥러닝 모델을 충분히 훈련하기 위한 데이터를 확보하는 기법이 Data Augmentation이며 
* 구글에서 발표한 데이터에 적합한 어그먼테이션 정책을 자동으로 찾아주는 알고리즘 -> AutoAugmentation (https://arxiv.org/abs/1805.09501)

![augmentation](https://user-images.githubusercontent.com/43362034/126589523-ac600cbe-ada7-4712-b23e-5864afcc9f14.PNG)

### 7. Inception module(GoogleLeNet)
* 전처리, 가중치 초기화 노력을 넘어서 네트워크 구조를 변화시켰다. 큰사이즈의 Conv filter를 적용하기 전에 1x1 conv를 통과시켜 연산 효율을 높이고 이미지내 비선형적인 특징들을 추출해낸다.(Bootleneck structure)
* Pose estimation등의 과제에서 잘 활용되었으나 비대칭 구조가 복잡해 뒤이어 연구를 중단.

![7  inception](https://user-images.githubusercontent.com/43362034/126630134-054e34a8-11ee-4d0d-8774-2260cc5ed6b9.JPG)
