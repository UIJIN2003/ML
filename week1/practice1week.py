import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

parameters = {"axes.labelsize": 20, "axes.titlesize": 30,'xtick.labelsize': 12, "ytick.labelsize": 12, "legend.fontsize": 12}
plt.rcParams.update(parameters)


'''training, validation, test set data 나누는 함수 '''
def data_division(n_data, Tr_rate, V_rate, Te_rate):

    np.random.shuffle(n_data)  # 데이터 섞기

    tr_index = int(len(n_data) * Tr_rate / 10)  # Tr_set 비율만큼 데이터 index 양 확인
    v_index = int(len(n_data) * V_rate / 10)  # V_set 비율만큼 데이터 index 양 확인
    te_index = int(len(n_data) * Te_rate / 10)  # Te_set 비율만큼 데이터 index 양 확인

    # 비율대로 data 나누기
    tr_set = n_data[0:tr_index]
    v_set = n_data[tr_index: tr_index + v_index]
    te_set = n_data[tr_index + v_index: tr_index + v_index + te_index]

    return tr_set, v_set, te_set


'''받아온 파일의 데이터 x와 y로 자동 분류 해주는 함수'''
def make_input_output(M):

    for i in range(M.shape[1]):  # M의 column 수만큼 반복
        if i == 0:
            x_matrix = M[:, 0]  # M의 첫번째 column 성분 값들 x_matrix에 저장
        elif i < (M.shape[1] - 1):
            # M의 마지막 column 성분 제외한 값들 x_matrix에 저장
            x_matrix = np.column_stack([x_matrix, M[:, i]])
        else:
            y = M[:, i]  # M의 마지막 column 성분 y에 저장
    y = y.reshape(y.shape[0], 1)  # y size 다듬기
    # 한 데이터에 대한 특징들 한 column에 나타나기 위해 transpose
    x_matrix_t = np.transpose(x_matrix)
    y_t = np.transpose(y)  # 위와 동일
    return x_matrix_t, y_t


'''데이터의 class수 세는 함수'''
def y_class(y):

    y_class = np.unique(y)  # class 수 계산
    Q = len(y_class)  # numpy array로 받아지기 때문에 길이를 셈

    return Q


'''데이터의 특징 수 세는 함수'''
def ch_count(y):

    Q = len(y)  # 받아온 데이터의 열 개수를 셈

    return Q


'''One-Hot Encoding 구현 함수'''
def One_Hot_Encoding(y):
    Q = y_class(y)                # 예: Q=3이면
    y_vector = np.zeros((Q, y.shape[1]))
    for i in range(y.shape[1]):
        label = int(y[0, i]) - 1      # y는 정수로 가정
        y_vector[label, i] = 1

    return y_vector


'''row기반 dummy추가해주는 함수'''
def add_dummy(x):

    x_dummy = np.ones(x.shape[1])  # 입력데이터 x의 길이만큼 dummy 생성
    x = np.row_stack([x, x_dummy])  # row방향으로 쌓음

    return x


'''sigmoid 구현 함수'''
def sigmoid_function(z):

    return (1/(1 + np.exp(-z)))


'''대표값 찾아서 1로 만들어주는 함수'''
def classification_data_max(y):

    p = np.zeros_like(y)  # 받아온 y데이터의 row와 column 크기만큼 요소가 0인 matrix 생성

    y_max = np.argmax(y, axis=0)  # y의 최댓값 index 저장

    # y 데이터 길이만큼 p (i번째 최댓값 index, i)에 1 저장
    for i in range(y.shape[1]):
        p[y_max[i], i] = 1

    return p


'''0.5기준으로 0, 1분류 함수'''
def classification_division_half(y_hat_data):

    trans_y = np.zeros([y_hat_data.shape[0], y_hat_data.shape[1]])

    for i in range(y_hat_data.shape[1]):
        for j in range(y_hat_data.shape[0]):
            if (y_hat_data[j, i] >= 0.5):
                trans_y[j, i] = 1
            else:
                trans_y[j, i] = 0

    return trans_y


'''데이터 정확도 함수'''
def data_accuracy(y_h, y):  # 한 데이터에 대한 성분 row로 나열한 데이터기준

    count = 0  # count 기능 이용할 변수 0으로 초기화

    for i in range(y_h.shape[1]):  # 예측 데이터 column 성분만큼 반복
        if (y_h[:, i] == y[:, i]).all():  # 받아온 데이터와 예측 데이터의 같은 column의 row성분 값이 모두 같은지 확인
            count += 1  # 위 조건에 해당할 때 count

    accuracy = count / y_h.shape[1]  # count 된 수를 예측데이터 column 개수만큼 나눠줌

    return accuracy


'''batch size 1 forward_propagation 구현 함수'''


# Hidden Layer의 node 수 지정
def forward_propagation_1(x_input_added_dummy, v_matrix, w_matrix, L):

    alpha = np.dot(v_matrix, x_input_added_dummy)  # v와 xinput을 곱해 alpha를 구함
    # batch size 1일 때 sigmoid에 넣으면 형태 깨져서 reshape이용
    b_matrix = sigmoid_function(alpha).reshape(-1, 1)

    b_matrix = add_dummy(b_matrix)  # b에 dummy 추가

    beta = np.dot(w_matrix, b_matrix)  # w와 b 곱해서 beta 구함
    y_hat = sigmoid_function(beta)  # beta를 sigmoid function에 넣어 y_hat 구함

    return y_hat, b_matrix


'''Legacy 이용한 Two_Layer_Neural_Network'''
def Two_Layer_Neural_Network_Legacy(X, Y, L, epoch, LR):

    # 예: Q=3이면
    Q = y_class(Y)
    y_vector = One_Hot_Encoding(Y)
    x_input = add_dummy(X)  # 입력에 dummy data 추가

    M = ch_count(X)  # input 속성 수 체크
    Q = y_class(Y)  # ouput class 수 체크

    # weight 초기화
    v = np.random.rand(L, M + 1) * 2 - 1
    w = np.random.rand(Q, L + 1) * 2 - 1

    v_list = []
    w_list = []

    MSE_list = []
    ACCURACY_list = []

    for j in range(epoch):

        y_hat_all = []  #y_hat_all list 초기화

        for i in range(Y.shape[1]):

            # parameter 저장
            v_list.append(v)
            w_list.append(w)

            # forward propagation 진행
            y_hat, b_matrix = forward_propagation_1(x_input[:, i], v, w, L)
            y_hat_all.append(y_hat)

            # back propagation
            # sigmoid function 미분
            grad_sf = 2 * (y_hat - y_vector[:, i].reshape(-1, 1)) * (y_hat * (1 - y_hat))
            grad_W = np.dot(grad_sf, b_matrix.T)  # weight w에 대한 미분

            grad_A = np.dot(w.T, grad_sf) * b_matrix * (1 - b_matrix)  # Alpha에 대한 미분

            grad_A = np.delete(grad_A, -1)  # dummy term 지우기
            grad_A = grad_A.reshape(-1, 1)  # (x, )를 (x, 1)꼴로 만들어 주기

            # weight v에 대한 미분
            grad_V = np.dot(grad_A, x_input[:, i].reshape(1, -1))

            # weight update
            v = v - LR * grad_V
            w = w - LR * grad_W

        # y_hat 값 펼치기
        y_hat_all_epoch = np.hstack(y_hat_all)

        error = y_hat_all - y_vector  # error 계산
        MSE = np.mean(error ** 2)  # MSE 계산
        MSE_list.append(MSE)  # MSE list에 저장

        y_hat_classific = classification_data_max(y_hat_all_epoch)  # 데이터 당 최댓값을 1로 만들어주는 분류 함
        accuracy = data_accuracy(y_hat_classific, y_vector)
        ACCURACY_list.append(accuracy)

    return v_list, w_list, ACCURACY_list, MSE_list


'''Hidden Layer 2개 FP'''
def forward_propagation_H2(X_added_dummy, V, W, U):

    alpha = np.dot(V, X_added_dummy)
    b_matrix = sigmoid_function(alpha).reshape(-1, 1)
    b_matrix_added_dummy = add_dummy(b_matrix)

    beta = np.dot(W, b_matrix_added_dummy)

    c_matrix = sigmoid_function(beta)
    c_matrix_added_dummy = add_dummy(c_matrix)

    gamma = np.dot(U, c_matrix_added_dummy)
    y_hat = sigmoid_function(gamma)
    return y_hat, b_matrix_added_dummy, c_matrix_added_dummy


'''Hidden Layer 2개 NN'''
def Two_Layer_Neural_Network_H2(X, Y, L, epoch, LR):
    
    # 예: Q=3이면
    Q = y_class(Y)
    y_vector = One_Hot_Encoding(Y)
    X_added_dummy = add_dummy(X)  # 입력에 dummy data 추가
    
    M = ch_count(X)  # input 속성 수 체크
    Q = y_class(Y)  # ouput class 수 체크
    
    # parameter 생성
    V = np.random.rand(L, M + 1) - 0.5
    W = np.random.rand(L, L + 1) - 0.5
    U = np.random.rand(Q, L + 1) - 0.5
    
    V_list = []
    W_list = []
    U_list = []
    
    MSE_list = []
    ACCURACY_list = []
    
    for j in range(epoch):
    
        y_hat_all = []
    
        for i in range(Y.shape[1]):
    
            # parameter 저장
            V_list.append(V)
            W_list.append(W)
            U_list.append(U)
    
            # forward propagation 진행
            y_hat, B_a, C_a = forward_propagation_H2(X_added_dummy[:, i], V, W, U)
            y_hat_all.append(y_hat)
    
            # back propagation
            grad_sf = 2 * (y_hat - y_vector[:, i].reshape(-1, 1)) * y_hat * (1 - y_hat)
    
            grad_U = np.dot(grad_sf, C_a.T)
            grad_C = np.dot(U.T, grad_sf) * C_a * (1 - C_a)
    
            grad_C = np.delete(grad_C, -1)
            grad_C = grad_C.reshape(-1, 1)
    
            grad_W = np.dot(grad_C, B_a.T)
    
            grad_A = np.dot(W.T, grad_C) * B_a * (1 - B_a)
    
            grad_A = np.delete(grad_A, -1)
            grad_A = grad_A.reshape(-1, 1)
    
            # weight v에 대한 미분
            grad_V = np.dot(grad_A, X_added_dummy[:, i].reshape(1, -1))
    
            # weight update
            V = V - LR * grad_V
            W = W - LR * grad_W
            U = U - LR * grad_U
    
        y_hat_all_epoch = np.hstack(y_hat_all)
    
        error = y_hat_all_epoch - y_vector  # error 계산
        MSE = np.mean(error ** 2)  # MSE 계산
        MSE_list.append(MSE)  # MSE list에 저장
    
        y_hat_classific = classification_division_half(y_hat_all_epoch)  # 데이터 당 최댓값을 1로 만들어주는 분류 함
        accuracy = data_accuracy(y_hat_classific, y_vector)
        ACCURACY_list.append(accuracy)

    return V_list, W_list, U_list, ACCURACY_list, MSE_list


'''confusion matrix 구현 함수'''
def confusion_matrix(y_hat, y_data):

    y_pred_index = np.argmax(y_hat, axis=0)  # y_hat 데이터당 최댓값 index 가져옴
    y_true_index = np.argmax(y_data, axis=0)  # y_data 데이터당 최댓값 index 가져옴

    true_num = 0  # 정확히 예측한 횟수 초기화

    classes_num = ch_count(y_data)  # y_data class 수 체크

    # 정확도 나타내기 위해 class수 + 1개만큼 정방 행렬 만듦
    confusion_matrix = np.zeros((classes_num + 1, classes_num + 1))

    # y 길이만큼반복
    for i in range(len(y_pred_index)):
        # 실제값, 예측값 index에 해당하는 자리에 1 더함
        confusion_matrix[y_true_index[i], y_pred_index[i]] += 1

    # class 수만큼 반복
    for i in range(classes_num):

        # row방향으로 더한 값이 0보다 클 때 전체 데이터로 정확히 예측한 값 나눠줌
        if sum(confusion_matrix[i, : classes_num]) > 0:
            confusion_matrix[i, classes_num] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, : classes_num])

        # column 방향으로 더한 값이 0보다 클 때 전체 데이터로 정확히 예측한 값 나눠줌
        if sum(confusion_matrix[: classes_num, i]) > 0:
            confusion_matrix[classes_num, i] = confusion_matrix[i, i] / np.sum(confusion_matrix[: classes_num, i])

        true_num += confusion_matrix[i, i]  # 정확히 예측한 값 세기
    confusion_matrix[classes_num, classes_num] = true_num / len(y_pred_index)  # 전체 데이터에 대한 정확도 마지막 index에 저장

    return confusion_matrix  # confusion_matrix 반환




# Hyper parameters 설정
L = 5
epoch = 1500
LR = 0.001

# 실습 데이터 불러오기
data = pd.read_csv('E:\\DML\\week1\\NN_data.csv')  # data 불러오기
data = data.to_numpy(dtype='float32')   # numpy array로 저장

# training, validation, test set 비율 설정
tr_r = 10
val_r = 0
te_r = 0

tr_data, val_data, te_data = data_division(data, tr_r, val_r, te_r)

# data 다시 섞기
np.random.shuffle(tr_data)

# input output 나누기
X, Y = make_input_output(tr_data)

# Hidden layer 1 NN
v, w, accuracy, mse = Two_Layer_Neural_Network_Legacy(X, Y, L, epoch, LR)

# Hidden Layer 2 NN
v_list, w_list, u_list, ACCURACY, MSE = Two_Layer_Neural_Network_H2(X, Y, L, epoch, LR)

#그래프로 나타내기
plt.figure(figsize=(12,5))


''' hidden layer 1 '''
# MSE
plt.subplot(1,2,1)
plt.plot(mse, label="MSE", color='red')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training MSE")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Accuracy
plt.subplot(1,2,2)
plt.plot(accuracy, label="Accuracy", color='blue')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

''' hidden layer 2 '''
plt.figure(figsize=(12,5))
# MSE
plt.subplot(2,2,1)
plt.plot(MSE, label="MSE", color='red')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training MSE")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Accuracy
plt.subplot(2,2,2)
plt.plot(ACCURACY, label="Accuracy", color='blue')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

      
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 1. Y 값에 따른 색상을 미리 정의 (딕셔너리 또는 리스트)
color_map = {1: 'b', 2: 'g', 3: 'r', 4: 'c', 5: 'm'}

# 2. Y 배열의 각 값에 해당하는 색상으로 구성된 리스트를 한 번에 생성
#    .get(key, default)를 사용하면 else 조건까지 깔끔하게 처리 가능
colors = [color_map.get(label, 'y') for label in Y[0, :]]

# 3. scatter 함수를 단 한 번만 호출하여 모든 점을 한 번에 그리기
ax.scatter(X[0, :], X[1, :], X[2, :], c=colors)
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_zlabel("y0")
plt.show()
        

