from io import BytesIO
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from matplotlib.font_manager import FontProperties
from sklearn import preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def predict(x_img, x):
    # 下載模型到本地
    #model_url = 'https://drive.google.com/file/d/1Ch8kUFXHesjfTbrDRdgz_k6RSr83z4mq/view?usp=sharing'
    #model_path = 'models/CNN_plastic.h5'
    #urllib.request.urlretrieve(model_url, model_path)

    # 載入模型
    #model = load_model(model_path)
    model = load_model('app/models_data/CNN_plastic.h5')

    prediction = model.predict([x_img, x])
    return prediction

# 在外部定義追踪器和顏色列表
plot_count = 0
colors = ['#FF8C00', '#8B0000', '#4682B4', '#2E8B57', '#DAA520', '#8A2BE2']  # 你可以加入更多顏色

def plot_cruve(stress, strain):
    global plot_count
    color = colors[plot_count % len(colors)]  # 循環使用顏色列表
    label = f'Prediction {plot_count + 1}'
    
    plt.plot(strain[0], stress[0], label=label, alpha=0.7, color=color)
    plt.title("Stress Strain Curve", fontsize=20, x=0.5, y=1.03)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Strain", fontsize=18, labelpad=5)
    plt.ylabel("Stress", fontsize=18, labelpad=5)
    
    # 設定圖標文字大小
    fontP = FontProperties()
    fontP.set_size(10)
    plt.legend(loc="best", fontsize=12, prop=fontP)
    
    plt.tight_layout()
    
    plot_count += 1


materials = ['Angle of weaving', 'Width of Yarn', 'Height of Yarn', 'Space', 
             'Epoxy_E', 'Epoxy_v',
             'Epoxy_Yield strength 1', 'Epoxy_Plastic strain 1', 
             'Epoxy_Yield strength 2', 'Epoxy_Plastic strain 2', 
             'Fibre_Density', 'Fibre_Linear density', 
             'Fibre_E1', 'Fibre_E2', 'Fibre_E3', 
             'Fibre_G12', 'Fibre_G23', 'Fibre_G13', 
             'Fibre_v1', 'Fibre_v2', 'Fibre_v3']

default = [90, 0.9, 0.3, 1.8, 
           20000, 0.4, 
           3, 0, 600, 0.3, 
           2550, 0.00056, 
           72000, 72000, 72000,
           30000, 30000, 30000,
           0.2, 0.2, 0.2]


# 讀取data，取得fit函數
train_df = pd.read_csv('app/models_data/plastic_data.csv')

x_train_data = train_df.values[:,25:46]
x_train_data = np.array(x_train_data)
X_minMax = preprocessing.MinMaxScaler().fit(x_train_data)

y_train_data = train_df.values[:,48:]
y_train_data = np.array(y_train_data)
Y_minMax = preprocessing.MinMaxScaler().fit(y_train_data)


def predict_materials(img_data, material_var):
    data = np.zeros((1, len(materials)))

    for i in range(len(materials)):
        if i == 0:
            data[0][i] = float(material_var[i])-90
        else: 
            data[0][i] = float(material_var[i])
    
    data_fit = X_minMax.transform(data)
    
    img_data_pre = img_data.reshape(1, 5, 5, 1)
    
    prediction = predict(img_data_pre, data_fit)
    prediction = Y_minMax.inverse_transform(prediction)
    
    stress_pred = np.zeros((prediction.shape[0], 20))
    strain_pred = np.zeros((prediction.shape[0], 20))
    i = 0
    for value in prediction:
        for j in range(prediction.shape[1]):
            if j % 2 == 0:
                stress_pred[i][j//2] = value[j]
            elif j % 2 == 1:
                strain_pred[i][j//2] = value[j]
        i += 1
    
    img_result = plot_cruve(stress_pred, strain_pred)
    # return img_result
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer.getvalue()

material_var = default
img_data = np.ones((1, 25))