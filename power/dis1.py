import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle

from utils import mkdirs


def dis_run(distributed_root, predict_root, save_root, save_weight_root):
    # 定义起始和结束日期  
    start_date = '2023-10-01'  
    end_date = '2023-10-08' 

    # 定义时间间隔  
    interval = pd.Timedelta('15min')  
    
    # 使用date_range()函数生成时间序列  
    time_series = pd.date_range(start=start_date, end=end_date, freq=interval) 

    mask = ((time_series.time >= pd.to_datetime('00:00').time()) & (time_series.time <= pd.to_datetime('04:45').time())) |  \
           ((time_series.time >= pd.to_datetime('20:00').time()) & (time_series.time <= pd.to_datetime('23:45').time())) 

    model_xgb = xgb.XGBRegressor()
    for i in range(1, 13, 1):
        dis_file = "FBS_" + str(i) + "_history.csv"
        pre_file = "FBS_" + str(i) + "_20231001-20231007_weather.csv"
        out_file = "FBS_" + str(i) + "_power_output.csv"
        out_weight = "FBS_" + str(i) + "_power_output"

        dis_path = Path(distributed_root) / dis_file
        pre_path = Path(predict_root) / pre_file
        out_path = Path(save_root) / out_file

        data = pd.read_csv(str(dis_path))
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])

        # 将在 00:00 - 04:45 和 20:00 - 23:45 的数据置为 0, 因为其中有些包含Nan
        mask = ((data['TIMESTAMP'].dt.time >= pd.to_datetime('00:00').time()) & (data['TIMESTAMP'].dt.time <= pd.to_datetime('04:45').time())) |  \
            ((data['TIMESTAMP'].dt.time >= pd.to_datetime('20:00').time()) & (data['TIMESTAMP'].dt.time <= pd.to_datetime('23:45').time()))
        data.loc[mask, 'POWER'] = 0

        # 去除其中的 Nan
        data = data.dropna()

        # 计算相关度
        correlation_matrix = data.corr()
        target_correlation = correlation_matrix['POWER'].abs().sort_values(ascending=False)

        # 取相关性大于0.5的特征(排除自身)
        # filter_features = target_correlation[target_correlation > 0.5].index.tolist()[1:]
        # print(filter_features)
        filter_features = ['WEATHER2_IR', 'WEATHER1_IR', 'WEATHER2_TMP', 'WEATHER1_TMP']

        # 划分训练集和验证集
        features = data[filter_features]
        power = data["POWER"]
        X_train, y_train = features, power
        # X_train, X_test, y_train, y_test = train_test_split(features, power, test_size=0.2, shuffle=False)

        # 开始训练模型
        model_xgb.fit(X_train, y_train)
        # val(model_xgb, X_test, y_test)

        # 保存模型到文件
        filename = Path(save_weight_root) / out_weight
        pickle.dump(model_xgb, open(filename, 'wb'))

        predict_pred = predict(model_xgb, str(pre_path), filter_features)
        
        df = pd.DataFrame({
            "TIMESTAMP": time_series,
            "POWER": predict_pred
        })
        df.loc[mask, 'POWER'] = 0

        # df.to_excel(out_path, index=False)
        print(f"{out_file} save success")
        
        # break


def val(model: xgb.XGBRegressor, X_test, y_test):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, len(y_pred), 1), y_test, color='blue', label='Original Data')
    plt.plot(np.arange(0, len(y_pred), 1), y_pred, color='red', linewidth=1, label='Predicted Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Comparison of Original Data and Predicted Data')
    plt.legend()
    plt.show()


def predict(model: xgb.XGBRegressor, file, filter_features):
    data_predict = pd.read_csv(file)
    data_predict = data_predict[filter_features]
    predict_pred = model.predict(data_predict)

    return predict_pred.tolist()


def all_predict(predict_root, save_root, weight_root):
    # 定义起始和结束日期  
    start_date = '2023-10-01'  
    end_date = '2023-10-08' 

    # 定义时间间隔  
    interval = pd.Timedelta('15min')  
    
    # 使用date_range()函数生成时间序列  
    time_series = pd.date_range(start=start_date, end=end_date, freq=interval) 

    mask = ((time_series.time >= pd.to_datetime('00:00').time()) & (time_series.time <= pd.to_datetime('04:45').time())) |  \
           ((time_series.time >= pd.to_datetime('20:00').time()) & (time_series.time <= pd.to_datetime('23:45').time())) 
    filter_features = ['WEATHER2_IR', 'WEATHER1_IR', 'WEATHER2_TMP', 'WEATHER1_TMP']
    for i in range(1, 13, 1):
        pre_file = "FBS_" + str(i) + "_20231001-20231007_weather.csv"
        out_file = "FBS_" + str(i) + "_power_output.csv"
        out_weight = "FBS_" + str(i) + "_power_output"

        pre_path = Path(predict_root) / pre_file
        out_path = Path(save_root) / out_file
        weight_path = Path(weight_root) / out_weight

        # 加载模型
        loaded_model = pickle.load(open(weight_path, 'rb'))

        data_predict = pd.read_csv(pre_path)
        data_predict = data_predict[filter_features]
        predict_pred = loaded_model.predict(data_predict)
        
        df = pd.DataFrame({
            "TIMESTAMP": time_series,
            "POWER": predict_pred
        })
        df.loc[mask, 'POWER'] = 0

        df.to_excel(out_path, index=False)
        print(f"{out_file} save success")


if __name__ == "__main__":
    distributed_root_ = "D:\Desktop\智慧能源专项赛-赛题二数据\分布式历史数据"
    predict_root_ = "D:\Desktop\智慧能源专项赛-赛题二数据\天气预报数据\分布式功率预测天气预报数据-23年10月1日-10月7日"
    save_root_ = "./outs/分布式12个区域未来7天功率预测"
    # save_root_ = "./outs"
    save_weight_root_ = "./new_weights/dis1"

    mkdirs(save_root_)
    dis_run(distributed_root_, predict_root_, save_root_, save_weight_root_)

    # all_predict(predict_root_, save_root_, save_weight_root_)




