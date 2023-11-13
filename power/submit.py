import pandas as pd
from pathlib import Path
import pickle


def mkdirs(path):
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True)


def power(predict_root, save_root, weight_root):
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
        pre_file = f"FBS_{i}_20231001-20231007_weather.csv"
        out_file = f"FBS_{i}_power_output.csv"
        out_weight = f"FBS_{i}_power_output"

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
        # 绘制原始数据和预测结果的对比图
        # plt.figure(figsize=(10, 6))
        # plt.plot(np.arange(0, len(predict_pred), 1), predict_pred, color='blue', label='Original Data')
        # plt.xlabel('X')
        # plt.ylabel('y')
        # plt.title('Comparison of Original Data and Predicted Data')
        # plt.legend()
        # plt.show()

        df.to_csv(out_path, index=False)
        print(f"power {out_file} save success")


def energy(pre_path, save_root, weight_root):
    # 定义起始和结束日期  
    start_date = '2022-11-01'  
    end_date = '2023-10-31' 

    # 定义时间间隔  
    interval = pd.Timedelta('1day')  
    
    # 使用date_range()函数生成时间序列  
    time_series = pd.date_range(start=start_date, end=end_date, freq=interval)[::-1] 

    month_series = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    predict_data = pd.read_csv(pre_path).iloc[:, 1:]
    predict_data = predict_data.groupby("AREA_NO")

    for i in range(1, 13, 1):
        out_file = f"FBS_{i}_energy_output.csv"
        out_weight = f"FBS_{i}_energy_output"
        predict_sub_dis1 = predict_data.get_group(i)[["WEATHER1_IR", "WEATHER2_IR"]]
        out_path = Path(save_root) / out_file

        # 加载模型
        with open(Path(weight_root)/out_weight, 'rb') as file:
            loaded_model = pickle.load(file)
        
        # 预测
        predict_pred = loaded_model.predict(predict_sub_dis1)
        
        df = pd.DataFrame({
            "ENERGY_DATE": time_series,
            "ENERGY": predict_pred
        })

        dic = {
            "MONTH": ["22-Nov", "22-Dec", "23-Jan", "23-Feb", "23-Mar", "23-Apr",
                      "23-May", "23-Jun", "23-Jul", "23-Aug", "23-Sep", "23-Oct"],
            "ENERGY": []
        }

        # 依次找出22年11月份到23年10月份的数据和
        for m in month_series:
            month_mask = (time_series.month == m)

            # 根据月份掩码筛选数据
            filtered_data = df[month_mask]

            # 计算符合月份的能量总和
            energy_sum = filtered_data['ENERGY'].sum()
            energy_sum = int(energy_sum)
            dic["ENERGY"].append(energy_sum)

        pd.DataFrame(dic).to_csv(out_path, index=False)
        print(f"energy {out_file} save success")


def load_solar(code, forecast_root):
    try:
        df = pd.read_csv(Path(forecast_root) / f'JZS_SOLARFARM{code}_20231001-20231007_weather.csv')
        # print(df.columns)
        df = df[['TIMESTAMP','WEATHER1_IR', 'WEATHER2_IR']]
        df.columns = [col + f'_{code}' if col != 'TIMESTAMP'  else col for col in df.columns]
        return df
    except:
        print(Path(forecast_root) / f'JZS_SOLARFARM{code}_20231001-20231007_weather.csv')
        return pd.DataFrame()


def load_wind(code, forecast_root):
    try:
        df = pd.read_csv(Path(forecast_root) / f'JZS_WINDFARM{code}_20231001-20231007_weather.csv')
        df = df[['TIMESTAMP','WEATHER1_WS',  'WEATHER2_WS']]
        df.columns = [col + f'_{code}' if col != 'TIMESTAMP' else col for col in df.columns]
        return df
    except:
        print(Path(forecast_root) / f'JZS_WINDFARM{code}_20231001-20231007_weather.csv')
        return pd.DataFrame()


def load_data_weather_all(forecast_root, group):
    id_list=[]
    if group ==1:
        id_list = [str(i).rjust(2, "0") for i in range(1, 6)]
    if group ==2:
        id_list = [str(i).rjust(2, "0") for i in range(6, 11)]

    df_list_solar = [load_solar(code, forecast_root) for code in id_list]
    df_list_wind = [load_wind(code, forecast_root) for code in id_list]
    df_all = pd.concat(df_list_wind+df_list_solar,axis=1).fillna(value=0)

    group_num=group
    target_feature_col_list = []
    if group_num==1:
        target_feature_col_list = [ 'WEATHER1_WS_01',
           'WEATHER2_WS_01', 'WEATHER1_WS_02', 'WEATHER2_WS_02', 'WEATHER1_WS_03',
           'WEATHER2_WS_03', 'WEATHER1_WS_04', 'WEATHER2_WS_04', 'WEATHER1_WS_05',
           'WEATHER2_WS_05', 'WEATHER1_IR_01', 'WEATHER2_IR_01', 'WEATHER1_IR_02',
           'WEATHER2_IR_02', 'WEATHER1_IR_03', 'WEATHER2_IR_03', 'WEATHER1_IR_04',
           'WEATHER2_IR_04', 'WEATHER1_IR_05', 'WEATHER2_IR_05']
    elif group_num==2:
        target_feature_col_list = [ 'WEATHER1_WS_06',
           'WEATHER2_WS_06', 'WEATHER1_WS_07', 'WEATHER2_WS_07', 'WEATHER1_WS_08',
           'WEATHER2_WS_08', 'WEATHER1_WS_09', 'WEATHER2_WS_09', 'WEATHER1_WS_10',
           'WEATHER2_WS_10', 'WEATHER1_IR_06', 'WEATHER2_IR_06', 'WEATHER1_IR_07',
           'WEATHER2_IR_07', 'WEATHER1_IR_08', 'WEATHER2_IR_08', 'WEATHER1_IR_09',
           'WEATHER2_IR_09', 'WEATHER1_IR_10', 'WEATHER2_IR_10']
    return df_all[target_feature_col_list]


def center(forecast_root, save_root, weight_root):
    # 定义起始和结束日期
    start_date = '2023-10-01'
    end_date = '2023-10-08'

    # 定义时间间隔
    interval = pd.Timedelta('15min')

    # 使用date_range()函数生成时间序列
    time_series = pd.date_range(start=start_date, end=end_date, freq=interval)

    for g in [1,2]:
        test_df = load_data_weather_all(forecast_root, g)
        model_path = Path(weight_root) / f'JZS_model_group_{g}.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        predict_pred = model.predict(test_df)
        df = pd.DataFrame({
            "TIMESTAMP": time_series,
            "POWER": predict_pred
        })
        out_path = Path(save_root) / f'JZS-AREA{g}_power_output.csv'
        df.to_csv(out_path, index=False)
        print(f"center {f'JZS-AREA{g}_power_output.csv'} save success")


def run():
    predict_root_ = r"./天气预报数据/分布式功率预测天气预报数据-23年10月1日-10月7日"
    save_root_d1 = r"./outs/分布式12个区域未来7天功率预测"
    save_weight_root_d1 = r"./new_weights/dis1"
    mkdirs(save_root_d1)
    power(predict_root_, save_root_d1, save_weight_root_d1)

    pre_path_ = r"./天气预报数据/分布式发电量预测天气预报数据-22年11月-23年10月/FBS-SSRD-20221101-20231031.csv"
    save_root_d2 = r"./outs/分布式12个区域未来12个月发电量预测"
    save_weight_root_d2 = r"./new_weights/dis2"
    mkdirs(save_root_d2)
    energy(pre_path_, save_root_d2, save_weight_root_d2)

    forecast_root = r'./天气预报数据/集中式天气预报数据-23年10月1日到10月7日'
    save_root_c = r"./outs/集中式2个场群未来7天功率预测"
    save_weight_root_c = r"./new_weights/cen"
    mkdirs(save_root_c)
    center(forecast_root, save_root_c, save_weight_root_c)


if __name__ == "__main__":
    run()
