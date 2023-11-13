"""
@FileName：submit_generate_prediction_result.py\n
@Description：\n
@Author：duanlianda\n
@Time：2023-11-10 2:45 p.m.\n
"""
import pandas as pd
import pickle
import os


forecast_root = r'data\智慧能源专项赛-赛题二数据\天气预报数据\天气预报数据\集中式天气预报数据-23年10月1日到10月7日'

#JZS_SOLARFARM01_20231001-20231007_weather.csv
def generate_code(num):
    return '0'*(2-len(str(num)))+str(num)

def load_data_weather_all(group=1):
    def load_solar(code):
        try:
            df = pd.read_csv(os.path.join(forecast_root,f'JZS_SOLARFARM{code}_20231001-20231007_weather.csv'))
            # print(df.columns)
            df = df[['TIMESTAMP','WEATHER1_IR', 'WEATHER2_IR']]
            df.columns = [col + f'_{code}' if col != 'TIMESTAMP'  else col for col in df.columns]
            return df
        except:
            print(os.path.join(forecast_root,f'JZS_SOLARFARM{code}_20231001-20231007_weather.csv'))
            return pd.DataFrame()
    def load_wind(code):
        try:
            df = pd.read_csv(os.path.join(forecast_root,f'JZS_WINDFARM{code}_20231001-20231007_weather.csv'))
            df = df[['TIMESTAMP','WEATHER1_WS',  'WEATHER2_WS']]
            df.columns = [col + f'_{code}' if col != 'TIMESTAMP' else col for col in df.columns]
            return df
        except:
            print(os.path.join(forecast_root,f'JZS_WINDFARM{code}_20231001-20231007_weather.csv'))
            return pd.DataFrame()
    id_list=[]
    if group ==1:
        id_list = [str(i).rjust(2, "0") for i in range(1, 6)]
    if group ==2:
        id_list = [str(i).rjust(2, "0") for i in range(6, 11)]

    df_list_solar = [load_solar(code=code) for code in id_list]
    df_list_wind = [load_wind(code=code) for code in id_list]
    df_all = pd.concat(df_list_wind+df_list_solar,axis=1).fillna(value=0)
    # print(df_all.columns)
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


def run():
    # 定义起始和结束日期  
    start_date = '2023-10-01'  
    end_date = '2023-10-08' 

    # 定义时间间隔  
    interval = pd.Timedelta('15min')  
    
    # 使用date_range()函数生成时间序列  
    time_series = pd.date_range(start=start_date, end=end_date, freq=interval) 

    for g in [1,2]:
        test_df = load_data_weather_all(group=g)
        model_path = rf'JZS_model_group_{g}.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        predict_pred = model.predict(test_df)
        df = pd.DataFrame({
            "TIMESTAMP": time_series,
            "ENERGY": predict_pred
        })
        df.to_csv(rf'output/JZS-AREA{g}_power_output.csv')
    print("FINISHED!")


if __name__ == '__main__':
    run()
