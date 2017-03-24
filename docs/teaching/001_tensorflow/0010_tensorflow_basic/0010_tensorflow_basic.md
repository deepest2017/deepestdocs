# 0010 Tensorflow Basic

---

**Deepest 2017 Teaching Session**

Contact: Kyuhong Shim <skhu20@snu.ac.kr>

---

## 1. What is TF?

Tensorflow 는 Google 에서 개발하고 지금도 전 세계 많은 사람들이 참여하고 있는 오픈소스 소프트웨어 라이브러리입니다. 배우기 쉽다는 점, 강력하고 다양한 기능들이 지금도 끊임없이 추가되고 있다는 점, 다양한 장치(device)와 운영체제(OS)에서 사용할 수 있다는 점, Google 이 받쳐주고 있어 망할 걱정은 안 해도 된다는 점(!) 등, 다양한 장점이 있습니다.

TF는 머신러닝/딥러닝을 위해 만들어졌고 아무래도 관련 분야에서 가장 많이 사용하고 있습니다. 하지만, 속을 들여다보면 대단히 범용적이기 때문에 어떤 수학/과학/공학 응용에도 사용할 수 있습니다. 파이썬으로 작성된 문법은 파이썬의 거대한 생태계와 맞물려 사실상 못 하는 게 없다고 봐도 될 정도입니다.

* TF [Homepage](https://www.tensorflow.org)
* TF [API](https://www.tensorflow.org/api_docs/)
* TF [Repository](https://github.com/tensorflow/tensorflow)
* TF [Model](https://github.com/tensorflow/models)

### TF Install

TF를 설치하는 방법은 간단합니다. 명령어 창을 켜고 TF 패키지를 설치해 주세요. CPU 버전은 윗줄을, GPU 버전은 아랫줄을 사용하면 됩니다. 기본적으로 파이썬이 필요하고, GPU 버전을 위해서는 Nvidia 그래픽 카드와 CUDA 라이브러리 설치가 필요합니다.

```
pip install tensorflow
pip install tensorflow-gpu
```

### Start TF

TF를 시작하는 방법은 간단합니다. 앞으로 이 문서에서는 아래 패키지들은 기본적으로 불러져 있다고 생각하겠습니다. Tensorflow >= 1.0.0 과 Python >= 3.5 에서 작성되었습니다.

```python
import tensorflow as tf
import numpy as np
```
---

## 2. Graph Structure

그래프(Graph)란 점과 선으로 연결된 도형을 의미합니다. 가장 대표적인 그래프 구조는 지하철 노선도입니다.

* 노드(Node) - 그래프의 각 점을 의미합니다.
* 엣지(Edge) - 그래프의 각 선을 의미합니다. 점과 점을 이어 줍니다.
* 서브그래프(Subgraph) - 그래프의 연결된 일부분을 의미합니다.

### Directed Graph

Directed graph 란 방향성이 있는 그래프를 의미합니다. 즉, 엣지의 양쪽 끝이 하나는 시작점, 하나는 끝점으로 다릅니다. 이제, 엣지는 화살표가 되고 노드에는 입력과 출력이 생겼습니다. 방향성이 있는 그래프 중에서도, TF가 사용하는 구조는 한번 방문한 노드는 어지간해서는 다시 방문하지 않는, 나무처럼 앞으로 쭉쭉 뻗어나가는 그래프 구조입니다.

방향성이 있는 그래프 구조를 만들고, 시작점과 끝점을 지정하면 TF는 그래프 구조를 따라가면서 시작점에 넣어준 재료로 끝점의 결과를 얻어냅니다. 꼭 시작점이 그래프 전체의 시작점이 아니어도 되고, 끝점이 그래프 전체의 끝점이 아니어도 됩니다. 정해준 시작점에 재료를 넣는 동작을 **피드(feed)**라고 합니다.

### Symbolic Graph

TF가 만든 그래프는 실제로 존재하는 그래프이지만 재료를 넣어 주기 전에는 어떤 의미도 갖고 있지 않습니다. Symbolic graph 란 실제 값은 가지고 있지 않지만 만약 값이 들어오면 어떻게 하겠다고 설정은 다 되어 있는 추상 그래프를 의미합니다. 입력을 넣으면 출력이 나오는 함수를 생각하면 비슷합니다.

TF는 미리 추상적인 규칙들과 연결을 다 지정해 놓고 실행시키는 Define-and-Run 방식을 채택하고 있습니다. 비슷한 Define-and-Run 방식의 라이브러리로는 **Caffe, Theano** 가 있습니다. TF는 다음과 같이 크게 3단계로 그래프를 다룹니다.

1. 그래프를 선언하기 - 우리가 규칙을 지정해 놓기 (programming!)
2. 그래프를 컴파일(compile)하기 - 미리 어떻게 동작할 지, 어떻게 메모리를 쓸 지 결정해 놓기
3. 그래프를 묶어 주기 - 실제로 원하는 그래프에 데이터를 넣을 수 있도록 입력부와 출력부를 만들어 주기

이렇게 그래프를 고정하는(static) 방식은 늘 어떻게 동작할 지를 알고 있기 때문에 메모리를 아끼고 속도를 빠르게 하는 것이 좀 더 쉽습니다. 한편, 동적으로(dynamic) 같은 그래프에서도 매번 다른 입력에 대해 서로 다른 동작을 하게 하는 것은 쉽지 않습니다. 미리 규칙과 연결을 지정해 놓지 않고 한줄 한줄 따라가며 실행시키는 방식은 Define-by-Run 이라고 하고, 대표적으로 **PyTorch, Chainer** 가 있습니다. 두 종류의 장단점이 있기 때문에, 하나씩은 알고 있는 게 도움이 됩니다.

### Node - Operation

각 노드는 잘게 나눠진 동작(operation)을 담당합니다. 노드가 데이터이고 엣지가 동작이라고 생각하기 쉬운데, 그 반대라는 점을 기억해 주세요. 노드는 0개 이상의 입력을 받아 0개 이상의 출력을 내보냅니다.

```python
u = tf.add(x, y)
z = tf.sum(u)
```
위 코드에서 노드는 2개 입니다. 첫 번째 노드는 _tf.add_ 이고, 2개의 입력을 받아 1개의 출력을 내보냅니다. 두 번째 노드는 _tf.sum_ 이고, 1개의 입력을 받아 1개의 출력을 내보냅니다. 간단한 사칙연산이 아니면 대부분의 동작은 TF에 바로 등록되어 있습니다. 동작의 문법은 Numpy와 아주 비슷하니 Numpy에 익숙하다면 쉽고 빠르게 적응할 수 있습니다! 그럼 Numpy를 쓰면 되지 왜 굳이 TF에서는 동작들을 따로 지정했을까요? 아래 코드처럼 하면 안 되는 걸까요?

```python
u = x + y
z = np.sum(u)
```
결론부터 말하면 간단한 사칙연산(+,-, *, /)를 제외하고는 안 하시는 걸 권장합니다! (그리고 많은 경우 에러가 납니다.) 간단한 사칙연산은 괜찮은 이유도 TF가 내부적으로 알아서 노드로 바꾸기 때문입니다. Numpy 함수는 실제 값을 받아 실제 값을 내보내는데, TF 그래프를 선언하는 과정은 완전히 추상적입니다. 즉, 수행하는 동작은 같아도 TF는 추상화된 데이터(가짜 데이터)를 입력으로 받을 수 있도록 추가적인 노력이 들어간 노드를 사용하는 것입니다. 전체 그래프가 추상 그래프가 되도록 동작은 <tf.xxx> 를 사용해 주세요. 

### Edge - Tensor

각 엣지로는 데이터가 흐릅니다. 위의 코드에서 x, y, u, z 에 해당하는 것들이 엣지라고 할 수 있습니다. 다시 한번, 노드가 데이터이고 엣지가 동작이라고 생각하기 쉬운데, 그 반대입니다. TF에서는 Numpy 와 같이, 데이터의 형태를 텐서 형태로 제한하고 있습니다. 그래서 텐서의 흐름 = Tensorflow 입니다! 텐서는 N차원 배열(N-dimensional array)입니다. 1차원 텐서는 vector, 2차원 텐서는 matrix, 3차원 텐서는 cube 라고 할 수 있습니다. 예를 들어, 4차원 텐서이고 각 차원의 크기가 (4,3,2,1) 이라면 앞에서부터 이렇게 해석할 수 있습니다.
> 모양이 (4,3,2,1)인 텐서에는 4칸이 있고 각 칸은 3칸으로 나눠지고 그 칸은 다시 2칸으로 나눠지고 그 칸 안에는 마지막으로 1칸이 들어 있다. 총 4x3x2x1 = 24개 숫자가 들어 있다.

이미지에서 많이 나오는 (50,32,32,3) 짜리 4차원 텐서를 보겠습니다.
> 모양이 (50,32,32,3)인 이미지 텐서는 50개 데이터를 갖고 있고 각 데이터는 가로 32개, 세로 32개 픽셀로 이루어져 있고 마지막으로 각 픽셀에는 R,G,B를 담은 3개 숫자가 들어 있다.

마지막 칸은 왜 있는 걸까요? (4,3,2) 짜리 3차원 텐서와 다른 점이 있을까요? 들어 있는 숫자의 갯수는 24개로 동일합니다. 하지만, 차원이 다르다는 것은 많은 차이를 만듭니다. 4차원으로 만들어진 텐서는 4차원과는 더할 수 있겠지만 3차원과는 더하지 못하겠죠. 앞으로 TF를 다루면서 3차 이상의 텐서를 많이 다룰 텐데, 차원을 맞춰주는 작업이 필요할 때가 많이 있습니다.

### State - Parameter

TF에서 한 번 데이터를 입력시키고 출력을 얻어낸 다음에는 그래프의 엣지를 지나간 텐서는 남아 있지 않습니다. 계속 추상 그래프의 형태를 유지하는 건데, 이렇게 하면 매번 같은 입력에 대해 같은 결과가 나올 테니 학습을 시킬 수가 없습니다. 그래서, TF에서는 추상 그래프 중에도 실제 값을 같이 갖고 있는 노드들을 만들었습니다. 이 노드들은 입력이 들어오면 내부적으로 갖고 있는 정보(혹은 state)를 사용해 입력을 변화시킵니다. 이 값들은 보통 변수(parameter)들이고, 훈련을 시킨다는 것은 변수를 변화시킨다는 것입니다. 동일한 그래프 구조를 사용하더라도 변수가 달라지면 결과가 달라지겠죠!

---

## 3. Basic Building Blocks

TF의 기본, 그래프를 선언하는 과정은 노드, 엣지, 스테이트를 만들고 이어붙이는 과정입니다. 여기서부터 나오는 코드는 전부 동작하는 코드입니다. 노드, 엣지, 스테이트라는 개념은 유지되지만 각각은 이제 이름이 붙습니다.

### Session

노드, 엣지, 스테이트는 모두 추상 그래프 위에 있습니다. 하지만 우리가 얻고 싶은 값은 실제 값입니다. 둘 사이를 조율해 주는 것이 세션입니다. 추상 그래프의 서브그래프를 가져와 입력부와 출력부를 만들면, 입력부부터 출력부까지 거쳐야 하는 노드, 엣지, 스테이트가 장치(device)의 메모리에 올라갑니다. 세션은 추상 그래프를 사용하는 고객(client)이 되어 입력과 출력을 얻어냅니다. 이 과정이 _sess.run()_ 입니다. _sess.run()_의 인자(argument)로는 출력을 낼 텐서(들)을 넣으면 됩니다. 앞에서 입력부와 출력부를 지정해 준다고 했는데, 입력부를 지정해 주는 방법은 밑에서 Placeholder 를 다루며 보겠습니다.
```python
sess = tf.Session()
```
세션은 보통 자동으로 종료되지만, 더 확실하게 실행이 끝나면 세션도 종료시키기 위해 with문을 씁니다. 아래 코드를 실행시키면 'client' 라는 단어가 보일 겁니다!
```python
with tf.Session() as sess:
    print('Session:', sess)
````
한 프로젝트에서 여러 개의 다른 그래프를 수행시킬 거라고 해도 세션은 하나를 써야 합니다. 간단히 생각해서, with 문 안에서 모든 일을 처리하세요. 아래의 모든 내용은 with 문 안에서 진행합니다!

### Variable

Variable 은 대표적인 스테이트 입니다. 즉, 매번 그래프를 실행시킬 때마다 값이 초기화되지 않고 유지되며, 값을 변화시키면 다음 실행 때는 변화된 값으로 수행됩니다. 따라서 맨 처음, 추상 그래프를 만들 때는 어떤 값을 가지고 있을 지 초기화(initialization)를 해 줘야 합니다. 초기화는 크게 두 가지 방법으로 할 수 있습니다.

1. Variable 을 선언하면서 직접 값을 넣어 준다.
2. Variable 을 선언하면서 어떤 값을 반환하는 initializer operation 을 넣어 준다.

Variable 을 만드는 방법은 _tf.Variable()_ 을 사용하거나 _tf.get_variable()_ 을 사용하는 방법이 있습니다. 두 방법의 차이는 Scope 와 관련이 있습니다.

직접 값을 넣어 줬든지, 어떤 값을 반환할 함수를 넣어 줬든지, 실제로 Variable 에 그 값이 들어가도록 하기 위해서도 세션을 실행시켜야 합니다. TF에서는 하나하나 Variable 을 초기화하는 것도 가능하지만 한 번에 모든 Variable 을 초기화하는 간편한 동작을 제공하고 있습니다.

```python
x = tf.Variable(np.array([3,4,5], dtype='int32'), name='X')
y = tf.Variable(tf.random_uniform([5,5], -1, 1))

init_op = tf.global_variables_initializer()
sess.run(init_op)

print('Variable name:', x.name, y.name)
print('Variable shape:', x.get_shape(), y.get_shape())
print('Variable type:', x.dtype, y,dtype)
print('Variable value:', sess.run(x), sess.run(y))

x.assign(np.array([6,7,8], dtype='int32'))
print('X value:', sess.run(x))  # Value does not change

sess.run(x.assign(np.array([6,7,8], dtype='int32')))
print('X value:', sess.run(x))  # Value changed
```
Name 을 안 넣어 주면 자기가 알아서 이름을 잡습니다. 중요한 Variable의 경우 이름을 넣어 구분하는 것이 나중에 결과를 확인하는 데에도 좋을 수 있습니다. 값을 넣고 빼는 것도 다 세션을 거치지 않으면 추상 그래프에 반영되지 않는다는 것을 주의하세요.

### Constant

Constant 도 스테이트의 한 종류입니다. 다만, Variable 과 다르게 Constant 는 한 번 정해지면 다시 값을 바꿀 수 없습니다. 프로그래밍할 때 첫 글자 c 가 소문자라는 걸 주의하세요!
```python
x = tf.Variable(np.array([3], dtype='int32'))
y = tf.constant(np.array([5], dtype='int32'))
z = x + y

init_op = tf.global_variables_initializer()
sess.run(init_op)

print(sess.run(z))
```

### Placeholder

앞서 본 Variable 과 Constant 가 추상 그래프에 값을 담고 있도록 하는 스테이트라면, Placeholder 는 실제 데이터가 들어올 공간으로 남겨 놓은 입력부 입니다. 그러니까 Placeholder 는 초기화할 필요도 없습니다. 프로그래밍할 때 첫 글자 p 가 소문자라는 걸 주의하세요!
```python
x = tf.Variable(np.array([3], dtype='int32'))
y = tf.placeholder(dtype='int32', shape=(1,))
z = x + y ** 2
u = z + 5

init_op = tf.global_variables_initializer()
sess.run(init_op)

print(sess.run(u, feed_dict={y:np.array([5], dtype='int32')}))  # 33
print(sess.run(u, feed_dict={z:np.array([11], dtype='int32')}))  # 16
```

위 코드를 좀 더 자세히 보겠습니다. 세션을 실행시킬 때 입력부와 출력부를 넣어 준다고 했는데, 첫 번째 run 과 두 번째 run 모두 u 를 출력으로 꺼내고 있습니다. u 를 만드는 데 필요한 건 z 이고, z 를 만드는 데 필요한 건 x 와 y 입니다. x 는 Variable 로 이미 그래프 내부에 값이 들어 있으니 y 만 넣어 주면 u 를 만들어 낼 수 있겠네요!

첫 번째 run 은 그래서 _feed_dict_ 라는 인자를 통해 y 를 입력부로 삼고 5 라는 값을 넣도록 합니다. 이제 세션은 y 값도 알고 x 값도 아니까 그래프를 지나면서 u 값을 계산해 냅니다. 한편, 두 번째 run은 _feed_dict_ 라는 인자에 z 를 입력부로 삼고 11 이라는 값을 넣도록 합니다. z 가 그래프의 시작이 되면 x, y 의 값은 알 필요도 없습니다! (그래서 y 가 Placeholder 임에도 값을 넣으라는 에러가 발생하지 않습니다.) 바로 z 부터 시작해 u 를 계산하고 결과를 냅니다. 

위 예시처럼 반드시 Placeholder 만 _feed_dict_ 에 들어갈 수 있는 건 아닙니다. 그럼에도 Placeholder 를 쓰는 이유 중 하나는, _sess.run()_ 을 했을 때 필요한 입력이 빠지지 않도록 하는 확인의 의미가 있습니다. 위 예시의 첫 번째 run 에서 y 를 feed 해 주지 않으면, 세션이 분석하기로는 분명 y 가 feed 되어야 하는데 안 되었기 때문에 에러가 발생하고 문제를 고칠 수 있습니다. 분명 그래프의 중간 텐서부터 그래프를 수행할 수 있긴 하지만, 대부분의 경우 Placeholder 에 외부 데이터를 넣게 될 것입니다.

---

## 4. Key Features

TF가 강력한 이유는 위의 아름다운 추상 그래프 구조뿐만이 아닙니다. 머신러닝에서 빠질 수 없는 미분 과정이 별도의 식을 쓰지 않아도 자동으로 이루어진다는 점, 시각화(visualization)이 너무나도 강력하다는 점, 스코프를 통해 영향을 미치는 범위를 지정할 수 있다는 점, 어떤 것도 저장하고 불러올 수 있다는 점 등이 TF가 인기 많은 이유라고 할 수 있습니다.

### Automatic Differentiation

자동 미분은 TF만 가지고 있는 기능은 아닙니다. 사실 파이썬 기반 머신러닝 라이브러리들 중 이 기능을 갖고 있지 않은 경우가 더 드물다고 할 수 있습니다. TF에서는 _tf.gradients()_ 를 사용해 미분을 수행해 기울기(gradient)를 구합니다. 더 정확히는 X(들)에 대한 Y(들)의 편미분값을 구해 줍니다. 하지만 TF에서는 생각보다 직접 _tf.gradients()_ 를 다룰 일은 없기 때문에, 이 소단원은 잘 이해가 안 가면 넘어가도 괜찮습니다.

\\( z = f(y), y = g(x) \\) 라는 식을 TF 그래프 구조로 만들었다고 생각해 보겠습니다. \\( \frac{\partial z}{\partial x}\\)라는 식을 구하고 싶더라도, x와 z는 그래프 상에서 직접적으로 연결되어 있지 않습니다.  하지만, \\( \frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \frac{\partial y}{\partial x}\\) 라는 연쇄 법칙(chain rule)을 생각할 수 있습니다. TF는 자동으로 \\( \frac{\partial z}{\partial y}, \frac{\partial y}{\partial x}\\)을 구해 최종적으로 원하는 gradient 를 구해 줍니다.

입력에 대한 출력의 미분을 구하고 싶다면, 입력에서 출력으로 거친 모든 노드를 거꾸로 거슬러 내려오면 됩니다. 이 방식을 역전파 (**back propagation**) 라고 합니다. 당연히 거꾸로 거슬러 내려올 때는 노드가 하는 일이 다르겠죠? 그래서 TF의 거의 모든 노드들에는 입력을 사용해 출력을 계산하는 동작(forward path)과 출력의 기울기를 사용해 입력의 기울기를 계산하는 동작(backward path)이 같이 정의되어 있습니다.
```python
x = tf.Variable(np.array([3], dtype='float32'))
y = tf.Variable(np.array([7], dtype='float32'))
z = 2 * x + tf.square(y)
u = z + 5

init_op = tf.global_variables_initializer()
sess.run(init_op)

grad_x = tf.gradients(u, x)
grad_y = tf.gradients(u, y)

print(sess.run(grad_x))  # 2
print(sess.run(grad_y))  # 14
```

TF가 미분을 하는 방식도 마찬가지로 세션을 통해 그래프를 실행하는 것입니다. 자동 미분이라고 하는 이유는, 그래프를 만들면 자동으로 그래프의 미분값을 구하기 위한 그래프도 만들어지기 때문입니다. (텐서보드에서 숨겨진 기울기 계산 그래프도 확인할 수 있습니다.) 눈에 보이는 문제는 아니지만, backward path 를 따라 거슬러 올라오기 위해서는 forward path 에서 계산한 값이 필요한 경우가 대부분입니다. 그래서 TF는 forward path 에서 사용한 값이 backward path 에서도 사용된다면 그 값을 버리지 않고 (= 메모리에서 지우지 않고) 계속 가지고 있다가 기울기 계산에 사용합니다. 메모리가 큰 GPU가 최고인 이유죠!

위 예시에서는 스칼라 입력과 스칼라 출력을 사용하는 기울기 계산을 봤는데, 실제로 우리가 다루는 것은 N차원 텐서입니다. N차원에서의 미분이 잘 상상이 안 될 수도 있기 때문에 일단 2차원을 생각해 보겠습니다.
```python
x = tf.Variable(np.array([[1,2],[0,1]], dtype='float32'))
y = tf.matmul(x, x)
z = tf.reduce_sum(y)

init_op = tf.global_variables_initializer()
sess.run(init_op)

grad_z = tf.gradients(z, x)
grad_y = tf.gradients(y, x)

print(sess.run(z))
print(sess.run(grad_z))

print(sess.run(y))
print(sess.run(grad_y))
```

위 코드는 x와 x를 행렬곱하고 전체를 합하고 있습니다. x에 대한 z의 기울기를 구하려면 그냥 z를 편미분 하면 됩니다.
$$
y = \left[ \begin{array}{cc} a & b \\ c & d \\ \end{array} \right] \left[ \begin{array}{cc} a & b \\ c & d \\ \end{array} \right] = \left[ \begin{array}{cc} a^2 + bc & ab + bd \\ ac + cd  & bc + d^2 \\ \end{array} \right] \\
z = a^2 + bc + ab + bd + ac + cd + bc + d^2
$$
 
z는 스칼라 값이기 때문에 상대적으로 쉬웠습니다. x에 대한 y의 기울기를 구하려면 어떻게 해야 할까요? 여기서 y는 하나의 값이 아니고, x도 하나의 값이 아닙니다.
x의 원소 각각에 대해 y의 원소 전체에 대한 기울기의 합이 구해집니다.
$$
\frac{\partial y}{\partial x} = \left[ \begin{array}{cc} \frac{\partial y}{\partial a} & \frac{\partial y}{\partial b} \\ \frac{\partial y}{\partial c} & \frac{\partial y}{\partial d} \\ \end{array} \right] \\
\frac{\partial y}{\partial a} = \sum_{i=0}^3 \frac{\partial y_i}{\partial a}
$$

기억해야 할 것은, 여러 개의 입력에 대한 여러 개의 출력의 기울기를 구할 때 각 입력 하나의 기울기로 계산되는 것은 그 입력으로 여러 개의 출력을 각각 미분한 기울기의 합이라는 것입니다. 물론, 잘 이해가 안 가도 괜찮습니다!

### TensorBoard

텐서보드는 TF의 시각화 툴입니다. TF는 _tf.summary_ 라는 모듈을 갖고 있는데, 이 요약 모듈은 TF의 어떤 값이라도 기록할 수 있습니다. 이렇게 기록된 값은 계속 파일에 쓰여지고 이 파일은 텐서보드를 통해 읽어들일 수 있습니다.

텐서보드에서 출력할 수 있는 정보는 이런 것들이 있습니다.

* 그래프 구조 - 노드와 엣지를 직접 그려줍니다.
* 스칼라 값 - 시간축에 따라 어떤 스칼라 값 하나가 변하는 것을 추적합니다.
* 히스토그램 - 어떤 텐서에 대해 평균, 분산, 분포 등을 추적합니다.
* 임베딩 - 수많은 데이터가 공간상에서 어떻게 분포하는지를 시각적으로 보여줍니다.

이런 값들을 얻기 위해, 요약(summary) 모듈은 원하는 텐서에 요약 노드를 붙입니다.  요약 노드는  입력 1개(이상), 출력 0개짜리 노드입니다. 당연히 이 노드들의 값을 보기 위해서도 세션으로 동작시켜야 합니다. 텐서보드를 쓰기 위해서는 다음과 같은 순서를 기억하면 됩니다.

1. 그래프를 선언하면서 보고싶은 값들에 요약 노드를 붙인다. _tf.summary_ 에 추가하면 됩니다.
2. 요약을 저장할 writer 를 만든다. 어디에 요약 로그를 저장할 지 정하는 과정입니다.
3. 로그에 저장하고 싶을 때마다 writer 에 세션을 돌려 나온 요약과 시간 index 를 추가한다.
4. 텐서보드를 실행시켜 로그를 확인한다.

```python
x = tf.Variable(np.array([10], dtype='float32'))
tf.summary.scalar('x', tf.reduce_sum(x))
y = 2 * x
tf.summary.scalar('y', tf.reduce_sum(y))

init_op = tf.global_variables_initializer()
sess.run(init_op)

update_op = x.assign(x-1)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('board/')

for i in range(10):
    print(sess.run(y))
    current_summary = sess.run(merged)
    writer.add_summary(current_summary, i)
    sess.run(update_op)
```

### Optimization

다행히도 TF에는 기울기 하강법(**gradient descent**)를 일일이 직접 구현하지 않아도 되도록 다양한 최적화 도구(optimizer)들을 제공하고 있습니다. SGD, RMSprop, AdaGrad 등입니다. 예시는 곧 만날 Regression 으로 대체하고, 여기서는 어떤 흐름으로 기울기가 내부적으로 계산되는지에 대해 간략하게 다루겠습니다. 기울기 하강법은 어떤 손실값(loss)를 넣어 주었을 때 그 손실에 대해 _minimize(loss)_ 를 불러 변수들을 업데이트합니다. _minimize_ 안에서는 두 개의 함수가 연속적으로 호출되는데, 첫 번째는 _compute_gradients()_ 이고 두 번째는 _apply_gradients()_ 입니다. 

_compute_gradients()_ 는 손실에 대한 변수들의 기울기를 _tf.gradient()_ 로 구합니다. 함수의 인자로 변수들을 넣어 주지 않아도, TF가 알아서 이 손실을 계산하는 데 필요한 모든 변수를 알아서 가져오고 그 변수들에 대해 기울기를 구합니다. 물론, _var_list_ 라는 인자를 통해 특정 변수들에 대해서만 기울기를 계산할 수도 있습니다! 전체 그래프의 일부 변수만을 업데이트하고 싶을 때 유용하게 쓰일 수 있습니다. _apply_gradients()_ 는 위에서 구한 기울기 (혹은 외부에서 구해 온 기울기) 를 변수들에 업데이트합니다.

모든 최적화 도구들을 초기화할 때는 학습 속도(learning rate)를 설정해 줘야 합니다. 좀 더 성능이 좋은 네트워크를 얻고 싶다면 학습 속도를 시간이 가는 것에 따라 줄여 주는 것이 꼭 필요합니다. TF는 학습 속도를 꼭 상수로 줄 필요 없이, 어떤 함수의 결과물로 매번 얻도록 할 수 있습니다. _tf.train.exponential_decay()_ 와 같은 함수로 학습 속도를 정의하고 최적화 도구에 넣는 것을 고려해 보세요.

---

