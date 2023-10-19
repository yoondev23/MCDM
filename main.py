
#%%
import os
import json
import warnings
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from typing import Dict

from tabulate import tabulate
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('png', 'pdf')
from shapely.geometry import LineString

from model.pymcdm import methods as mcdm_methods
from model.pymcdm import weights as mcdm_weights
from model.pymcdm import normalizations as norm
from model.pymcdm import correlations as corr
from model.pymcdm.helpers import rankdata, rrankdata


warnings.simplefilter('ignore')
warnings.filterwarnings(action='ignore')

#%%

def load_config(config_filename: str, CONFIG_PATH=os.path.join('config'), **kwargs) -> Dict:
    print(f'config_file directoty: {os.path.join(CONFIG_PATH, config_filename)}')
    with open(os.path.join(CONFIG_PATH, config_filename), encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def data_load(cfg):
    DATASET_PATH =  os.path.join('dataset')
    EU_dataset_nm = cfg['EU_dataset_nm']
    df_barr = pd.read_csv(os.path.join(DATASET_PATH, EU_dataset_nm))

    # 위치 좌표를 담을 리스트 초기화
    barrier_point = []
    for i in range(len(df_barr)):
        barrier_point += [[df_barr['위도'][i]] + [df_barr['경도'][i]]]

    # 데이터프레임의 모든 요인을 저장할 집합 초기화
    factor = set()

    # '요인' 열을 반복하며 요인을 추출하고 집합에 추가
    for i in df_barr['요인']:
        if i == i:
            factor_list = i.split(',')
            for j in factor_list:
                factor.add(j)

    # 모든 요인을 열로 추가하고, 해당 요인이 있는 경우 해당 열 값을 1로 설정
    for i in factor:
        df_barr[i] = 0
        
    #요인이 'nan'인 경우 삭제
    df_barr = df_barr.dropna(subset=['요인'])

    # 각 요인이 있는 경우 해당 열 값을 1로 설정
    for i in factor:
        df_barr.loc[df_barr['요인'].str.contains(i), i] = 1

     # '주의대상' 열에 '휠체어'가 포함된 행만 선택하여 반환
    df_barr_target = df_barr[df_barr['주의대상'].str.contains('휠체어')]
    return df_barr_target


def route2factor(cfg, df_barr_target, route_):#route를 geopandas로 변경 후, 해당 route에 있는 factor개수 counting
    df_barr = df_barr_target
    weight = cfg['Weight']
    coor_1m = (1/88.74/1000) #약 1m
    
    df = pd.DataFrame(route_)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[0], df[1]),  crs="EPSG:4326")
    gdf['route'] = 1
    gdf_1 = gdf.groupby(['route'])['geometry'].apply(lambda x:LineString(x.tolist()))
    gdf_1 = gpd.GeoDataFrame(gdf_1, geometry='geometry')
    # gdf_1.buffer(0.0001).plot()

    factor = set()    
    for i in df_barr['요인']:
        if i == i:
            factor_list = i.split(',')
            for j in factor_list:
                factor.add(j)
    for i in factor:
        df_barr[i] = 0
    df_barr = df_barr.dropna(subset=['요인'])#요인이 'nan'인 경우 삭제

    for i in factor:
        df_barr.loc[df_barr['요인'].str.contains(i), i] = 1
    gdf_barr = gpd.GeoDataFrame(
        df_barr, geometry=gpd.points_from_xy(df_barr['위도'], df_barr['경도']),  crs="EPSG:4326")
    gdf_buffer = gpd.GeoDataFrame(gdf_1.buffer(coor_1m*10), geometry=0, crs="EPSG:4326") # 반경 약 10m의 장애 요인들 고려
    polygons_contains = gpd.sjoin(gdf_buffer, gdf_barr , op='contains')           
    df_factor = pd.DataFrame(polygons_contains[['안내시설',	'경사',	'통행폭',	'높이',	'돌출물',	'단차',	'마감']].sum()).T
    for key in list(weight.keys()):
        df_factor[key] = df_factor[key].apply(lambda x: weight[key]*x)
    
    print(df_factor)
    return df_factor

def route_load(cfg, route_path):
    with open(os.path.join(route_path), 'r') as f:
        json_data = json.load(f)
    routes = json_data['routes']
    for idx, _ in enumerate(routes):
        globals()[f'route_{idx}'] = []
        for i in range(len(json_data["routes"][idx]["routeItems"])):
            globals()[f'route_{idx}'] += json_data["routes"][idx]["routeItems"][i]["path"]
        print(f'route preprocessing for route_{idx}')
        if idx == 0:
            check_df_0 = route2factor(cfg, df_barr_target, globals()[f'route_{idx}'])
        else:
            check_df_1 = route2factor(cfg, df_barr_target, globals()[f'route_{idx}'])
            check_df_0 = pd.concat([check_df_0, check_df_1], axis=0)
    df_route2factor = check_df_0.reset_index(drop=True)
    return df_route2factor

def mcdm_run(data, save_name):
    data = data+0.001
    matrix = data[data.columns].to_numpy()
    
    weights = mcdm_weights.equal_weights(matrix)
    # print(f'weights: {weights}')
    # print(mcdm_weights.entropy_weights(matrix))
    # print(mcdm_weights.standard_deviation_weights(matrix))
    types = np.array([1, -1, -1, -1, -1, -1, -1])
    topsis = mcdm_methods.TOPSIS()
    # print(topsis(matrix, weights, types))
    # print(topsis(matrix, weights, types).shape)
    topsis_methods = {
        'minmax': mcdm_methods.TOPSIS(norm.minmax_normalization),
        # 'max': mcdm_methods.TOPSIS(norm.max_normalization),
        # 'sum': mcdm_methods.TOPSIS(norm.sum_normalization),
        # 'vector': mcdm_methods.TOPSIS(norm.vector_normalization),
        # 'log': mcdm_methods.TOPSIS(norm.logaritmic_normalization),
    }
    results = {}
    for name, function in topsis_methods.items():
        results[name] = function(matrix, weights, types)
    display_result = tabulate([[name, *rankdata(pref, reverse=True)] for name, pref in results.items()],
                headers=['Method'] + [f'route{i+1}' for i in range(10)])
    print(display_result)
    text_file=open(save_name,"w")
    text_file.write(display_result)
    text_file.close()
    return results['minmax']

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default='ver_1.yaml')
parser.add_argument('--route_path', default='Path_API_Response_Example.json')
parser.add_argument('--SAVE_PATH', default='SAVE_PATH')

args = parser.parse_args()
config_filename = args.config_filename
route_path = args.route_path
SAVE_PATH = args.SAVE_PATH

cfg = load_config(config_filename)
df_barr_target = data_load(cfg)
df_route2factor = route_load(cfg, route_path)
result = mcdm_run(df_route2factor, save_name=SAVE_PATH)
print(1)
print(result)



# %%
