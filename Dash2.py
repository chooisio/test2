from dash import Dash, html, dcc, Input, Output
import csv
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import json
from shapely.geometry import Polygon, Point


# function Definition
def time_def(hr):
    result = -1
    result_str = ['Morning', 'Afternoon', 'Evening']
    if hr < 12: result = 0
    elif hr < 18: result = 1
    else: result = 2
    return str(result) + ' ' + result_str[result]

def week_def(date):
    weekday = ['1 Mon', '2 Tue', '3 Wed', '4 Thur', '5 Fri', '6 Sat', '7 Sun']
    return weekday[date%7]

app = Dash(__name__)

colors = {
    'background': '#434648',
    'fig_background': '#5F6366',
    'plot_background': '#5F6366',
    'text': '#FFFFFF',
    'red': '#C1395E',
    'orange': '#E07B42',
    'yellow': '#F0CA50',
    'green': '#AEC17B',
    'blue': '#89A7C2',
    'BIG': '#AFBC8f',
    'CAR': '#E7D292',
    'MOTOR': '#E5A952',
}

# 資料處理以便後續畫圖的處理
'''Windows 11'''
'''
# path = '../data_source/IntegrateData.csv'
path = "..\data_source\IntegrateData.csv"
df = pd.read_csv(path, encoding='utf-8')
with open('../data_source/taipei_districts.json', 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)
'''

'''mac'''
path = 'IntegrateData.csv'
df = pd.read_csv(path, encoding='utf-8')
with open('taipei_districts.json', 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)


#處理pivot_table，內容為路段資訊及每日總量
df['InfoData'] = pd.to_datetime(df['InfoData'], format='%Y-%m-%d %H:%M:%S')
df['Date'] = df['InfoData'].dt.date
df['RoadTotal'] = df['BIGVOLUME'] + df['CARVOLUME'] + df['MOTORVOLUME']
grouped = df.groupby(['RoadName', 'Date', 'PositionLon', 'PositionLat'])['RoadTotal'].sum().reset_index()
pivot_table = grouped.pivot_table(values='RoadTotal', index=['RoadName', 'PositionLon', 'PositionLat'], columns='Date', fill_value=0)

# 計算各個 RoadName 的 Total
Total = pivot_table.sum(axis=1)

# 計算各個 RoadName 的 Label 和 Color
interval_list = [15000, 25000, 35000, 45000]
range_str = ["≤ {}".format(interval_list[0])]
range_str += ["{} to {}".format(interval_list[i]+1,interval_list[i+1]) for i in range(3)]
range_str += ["≥ {}".format(interval_list[3]+1)]
bins = [0] + interval_list + [np.inf]
# bins= [0, 15000, 25000, 35000, 45000, inf]
Color_list = pd.cut(Total, bins=bins, labels=["BLUE","GREEN", "YELLOW", "ORANGE", "RED"], include_lowest=True).tolist()

ColorLabel_list = pd.DataFrame({'RoadName': pivot_table.index.get_level_values('RoadName').tolist(),
                                'PositionLon':pivot_table.index.get_level_values('PositionLon').tolist(),
                                'PositionLat':pivot_table.index.get_level_values('PositionLat').tolist(),
                                'Total_Vol':Total.tolist(),
                                'Color':Color_list,
                                'Range':pd.cut(Total, bins=bins, labels=range_str).tolist()})

DataOfBM_df = pd.merge(pivot_table, ColorLabel_list, on=['PositionLon', 'PositionLat', 'RoadName'])
DataOfBM_df = DataOfBM_df.sort_values('Total_Vol')
#DataOfBM_pd ----> 'lon', 'lat', 'RoadName', 'Total_Vol', 'Color', 'Range'

#路段的資料
RoadInfo = pd.DataFrame({'RoadName': pivot_table.index.get_level_values('RoadName').tolist(),
                        'PositionLon':pivot_table.index.get_level_values('PositionLon').tolist(),
                        'PositionLat':pivot_table.index.get_level_values('PositionLat').tolist()})

# function： 取得RoadName的list，造成Bubble Map會取範圍外得同名道路，所以後面會再取邊界
def get_roadname(lon, lat):
    lon = float(lon)
    lat = float(lat)
    df = RoadInfo.query('PositionLon == @lon and PositionLat == @lat')
    if len(df) > 0:
        return df['RoadName'].iloc[0]
    else:
        return None

#---------------------------------------------Mapbox callback function---------------------------------------------
@app.callback( Output('Bubble Map', 'figure'),
               Input('Line Chart', 'selectedData'),
               Input('Pie Chart', 'clickData'),
               Input('Bar Chart', 'clickData'),
               Input('Bubble Map', 'selectedData'))
def draw_BubbleMap(selectedData, clickData, clickData2, selectedData2):
    DataOfBM = DataOfBM_df.copy()
    df_subset = df.copy()
    if selectedData is None and clickData is None and clickData2 is None and selectedData2 is None: #Default
        df_subset = df.copy()
        df_subset['RoadTotal'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData is not None and selectedData2 is not None and selectedData['points'] != []: # Bubble Map and Pie Chart and Line Chart
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        lon_range = [selectedData2['range']['mapbox'][0][0], selectedData2['range']['mapbox'][1][0]] # 取邊界
        lat_range = [selectedData2['range']['mapbox'][1][1], selectedData2['range']['mapbox'][0][1]]
        df_subset = df_subset[(df_subset['PositionLon'] >= lon_range[0]) & (df_subset['PositionLon'] <= lon_range[1]) & (df_subset['PositionLat'] >= lat_range[0]) & (df_subset['PositionLat'] <= lat_range[1])]
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME']
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and selectedData2 is not None and selectedData['points'] != []: # Line Chart and Bubble Map
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        lon_range = [selectedData2['range']['mapbox'][0][0], selectedData2['range']['mapbox'][1][0]] # 取邊界
        lat_range = [selectedData2['range']['mapbox'][1][1], selectedData2['range']['mapbox'][0][1]]
        df_subset = df_subset[(df_subset['PositionLon'] >= lon_range[0]) & (df_subset['PositionLon'] <= lon_range[1]) & (df_subset['PositionLat'] >= lat_range[0]) & (df_subset['PositionLat'] <= lat_range[1])]
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset['RoadTotal'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData is not None and selectedData['points'] != []: # Line Chart and Pie Chart
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset = df.copy()
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif clickData is not None and selectedData2 is not None: # Pie Chart and Bubble Map
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        lon_range = [selectedData2['range']['mapbox'][0][0], selectedData2['range']['mapbox'][1][0]] # 取邊界
        lat_range = [selectedData2['range']['mapbox'][1][1], selectedData2['range']['mapbox'][0][1]]
        df_subset = df_subset[(df_subset['PositionLon'] >= lon_range[0]) & (df_subset['PositionLon'] <= lon_range[1]) & (df_subset['PositionLat'] >= lat_range[0]) & (df_subset['PositionLat'] <= lat_range[1])]
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and selectedData['points'] != []: # Line Chart
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset = df.copy()
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset['RoadTotal'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    elif clickData is not None: # Pie Chart
        df_subset = df.copy()
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif clickData2 is not None: # Bar Chart
        label = clickData2['points'][0]['label']
        DataOfBM = DataOfBM_df[DataOfBM_df['Color'] == label]
    elif selectedData2 is not None: # Bubble Map
        # print(selectedData2)
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        lon_range = [selectedData2['range']['mapbox'][0][0], selectedData2['range']['mapbox'][1][0]] # 取邊界
        lat_range = [selectedData2['range']['mapbox'][1][1], selectedData2['range']['mapbox'][0][1]]
        df_subset = df_subset[(df_subset['PositionLon'] >= lon_range[0]) & (df_subset['PositionLon'] <= lon_range[1]) & (df_subset['PositionLat'] >= lat_range[0]) & (df_subset['PositionLat'] <= lat_range[1])]
    else:
        df_subset = df.copy()

    # if clickData is not None or selectedData is not None: # Pie Chart or Line Chart
    if clickData is not None or selectedData is not None or selectedData2 is not None: # Pie Chart or Line Chart and Bubble Map
        grouped = df_subset.groupby(['RoadName', 'Date', 'PositionLon', 'PositionLat'])['RoadTotal'].sum().reset_index()
        pivot_table = grouped.pivot_table(values='RoadTotal', index=['RoadName', 'PositionLon', 'PositionLat'], columns='Date', fill_value=0)

        Total = pivot_table.sum(axis=1)
        if clickData2 is not None: # Bar Chart
            label = clickData2['points'][0]['label']
            if label == 'BLUE':
                Total = Total[Total.values <= interval_list[0]] 
            elif label == 'GREEN':
                Total = Total[(Total.values > interval_list[0]) & (Total.values <= interval_list[1])]
            elif label == 'YELLOW':
                Total = Total[(Total.values > interval_list[1]) & (Total.values <= interval_list[2])]
            elif label == 'ORANGE':
                Total = Total[(Total.values > interval_list[2]) & (Total.values <= interval_list[3])]
            elif label == 'RED':
                Total = Total[Total.values > interval_list[3]]

        Label_list = pd.cut(Total, bins=bins, labels=range_str).tolist()
        Color_list = pd.cut(Total, bins=bins, labels=["BLUE","GREEN", "YELLOW", "ORANGE", "RED"]).tolist()

        DataOfBM = pd.DataFrame({
                                    'PositionLon': Total.index.get_level_values('PositionLon'), # pivot table改成Total
                                    'PositionLat': Total.index.get_level_values('PositionLat'),
                                    'RoadName': Total.index.get_level_values('RoadName'),
                                    'Total_Vol': Total, 
                                    'Color': Color_list, 
                                    'Range': Label_list
                                })

    map_fig = px.scatter_mapbox(
                                DataOfBM,
                                labels=dict(color='Total_Vol'),
                                lat=DataOfBM['PositionLat'],
                                lon=DataOfBM['PositionLon'],
                                color=DataOfBM['Range'],
                                hover_name=DataOfBM['RoadName'],
                                size=DataOfBM['Total_Vol'],
                                size_max=20,
                                zoom=10,
                                color_discrete_map={
                                    range_str[0]: colors['blue'],
                                    range_str[1]: colors['green'],
                                    range_str[2]: colors['yellow'],
                                    range_str[3]: colors['orange'],
                                    range_str[4]: colors['red']
                                }
    )
    map_fig.update_layout(
        mapbox_style='carto-positron',
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['fig_background'],
        font_color=colors['text'],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=10)
        ),
        geo=dict(
            scope='asia',
            center=dict(lat=25, lon=121.53),
            projection_scale=100,
            showland=True
        ),
        margin=dict(l=30, r=30, t=60, b=30),
        title='Taiwan Map',
        hovermode='closest'
    )
    return map_fig

#---------------------------------------------Choropleth Map callback function---------------------------------------------
@app.callback( Output('Choropleth Map', 'figure'),
               Input('Line Chart', 'selectedData'),
               Input('Pie Chart', 'clickData'),
               Input('Bubble Map', 'selectedData'),
               Input('Bar Chart', 'clickData') )
def draw_ChoroplethMap(selectedData, clickData, selectedData2, clickData2):
    # if selectedData is None and clickData is None: #Default
    if selectedData is None and clickData is None and selectedData2 is None and clickData2 is None: # Default
        df_subset = df.copy()
        # df_subset = df.head(30000)
    elif selectedData is not None and clickData is not None and selectedData2 is not None and clickData2 is not None and selectedData['points'] != []: # All
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName2)].copy()
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset = df_subset.head(30000)
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif clickData is not None and selectedData2 is not None and clickData2 is not None: # Pie Chart and Bubble Map and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName2)].copy()
        df_subset = df_subset.head(30000)
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and selectedData2 is not None and clickData2 is not None and selectedData['points'] != []: # Line Chart and Bubble Map and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName2)].copy()
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset = df_subset.head(30000)
    elif selectedData is not None and clickData is not None and clickData2 is not None and selectedData['points'] != []: # Line Chart and Pie Chart and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset = df_subset.head(30000)
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData is not None and selectedData2 is not None and selectedData['points'] != []: # Line Chart and Pie Chart and Bubble Map
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset = df_subset.head(30000)
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif clickData is not None and selectedData2 is not None: # Pie Chart and Bubble Map
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        df_subset = df_subset.head(30000)
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif clickData is not None and clickData2 is not None: # Pie Chart and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset.head(30000)
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif clickData is not None and selectedData is not None and selectedData['points'] != []: # Pie Chart and Line Chart
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset = df.copy()
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset = df_subset.head(30000)
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData2 is not None and clickData2 is not None: # Bubble Map and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        # df_subset = df_subset.head(30000)
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName2)].copy()
        df_subset = df_subset.head(30000)
    elif selectedData2 is not None and selectedData is not None and selectedData['points'] != []: # Bubble Map and Line Chart
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset = df_subset.head(30000)
    elif clickData2 is not None and selectedData is not None and selectedData['points'] != []: # Bar Chart and Line Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset = df_subset.head(30000)
    elif selectedData is not None and selectedData['points'] != []: # Line Chart
        first = selectedData['range']['x'][0]
        last = selectedData['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset = df.copy()
        # df_subset = df_subset.head(50000)
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset = df_subset.head(30000)
    elif clickData is not None: # Pie Chart
        df_subset = df.copy()
        df_subset = df_subset.head(30000)
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData2 is not None: # Bubble Map
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData2['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        df_subset = df_subset.head(30000)
    elif clickData2 is not None: # Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset.head(30000)
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
    else:
        df_subset = df.copy()
        df_subset = df_subset.head(30000)
      
    geo_dict = {feature['properties']['TOWN']: Polygon(feature['geometry']['coordinates'][0]) 
                for feature in geojson_data['features']}

    df_subset['Town'] = [town_id for town_id, polygon in geo_dict.items() for lon, lat in zip(df_subset['PositionLon'], df_subset['PositionLat'])
                     if polygon.contains(Point(lon, lat))]
    
    grouped = df_subset.groupby('Town').agg({'RoadTotal': 'sum'}).reset_index()

    choropleth_map = px.choropleth(
                                            grouped,
                                            geojson=geojson_data,
                                            locations='Town',
                                            color='RoadTotal',
                                            color_continuous_scale="Pinkyl",
                                            featureidkey="properties.TOWN",
                                            projection="mercator",
                                            labels={'RoadTotal':'RoadTotal'}
    )
    choropleth_map.update_layout(
        # mapbox_style='carto-positron',
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['fig_background'],
        font_color=colors['text'],
        margin=dict(l=30, r=30, t=60, b=30),
        title='Taiwan Map',
        hovermode='closest'
    )
    choropleth_map.update_geos(fitbounds="locations", visible=True)
   
    return choropleth_map

#---------------------------------------------Bar Chart callback function---------------------------------------------
@app.callback( Output('Bar Chart', 'figure'),
               Input('Bubble Map', 'selectedData'),
               Input('Line Chart', 'selectedData'),
               Input('Pie Chart', 'clickData'),
               Input('Bar Chart', 'clickData')) # test
def draw_BarChart(selectedData, selectedData2, clickData, clickData2):
    color_counts = {"BLUE": 0, "GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0}
    label = ''
    # if selectedData is None and selectedData2 is None and clickData is None: #Default
    df_subset = df.copy()
    if selectedData is None and selectedData2 is None and clickData is None and clickData2 is None: # Default
        for color in Color_list:
            color_counts[color] += 1
    elif selectedData is not None and selectedData2 is not None and clickData is not None and clickData2 is not None and selectedData2['points'] != []: # All
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME']
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and selectedData2 is not None and clickData2 is not None and selectedData2['points'] != []: # Bubble Map and Line Chart and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif selectedData is not None and clickData is not None and clickData2 is not None: # Bubble Map and Pie Chart and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME']
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and selectedData2 is not None and clickData is not None and selectedData2['points'] != []: # Bubble Map and Line Chart and Pie Chart, Line Chart放最後
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME']
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and selectedData2 is not None and selectedData2['points'] != []: # Bubble Map and Line Chart
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif clickData is not None and selectedData2 is not None and selectedData2['points'] != []: # Pie Chart and Line Chart
            df_subset = df.copy()
            first = selectedData2['range']['x'][0]
            last = selectedData2['range']['x'][1]
            first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
            last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
            df_subset['Date'] = pd.to_datetime(df_subset['Date'])
            df_subset = df[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
            if clickData['points'][0]['label'] == 'BIG':
                df_subset['RoadTotal'] = df_subset['BIGVOLUME']
            elif clickData['points'][0]['label'] == 'CAR':
                df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
            elif clickData['points'][0]['label'] == 'MOTOR':
                df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData is not None: # Bubble Map and Pie Chart
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME']
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData2 is not None: # Bubble Map and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName)].copy()
    elif clickData is not None and clickData2 is not None: # Pie Chart and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME']
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
    elif selectedData2 is not None and clickData2 is not None and selectedData2['points'] != []: # Line Chart and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif selectedData is not None: # Bubble Map
        marker_sizes = [p["marker.size"] for p in selectedData["points"]]
        for size in marker_sizes:
            if size > interval_list[3]:
                color_counts["RED"] += 1
            elif size > interval_list[2]:
                color_counts["ORANGE"] += 1
            elif size > interval_list[1]:
                color_counts["YELLOW"] += 1
            elif size > interval_list[0]:
                color_counts["GREEN"] += 1
            else:
                color_counts["BLUE"] += 1
    elif clickData is not None: # Pie Chart   
            df_subset = df.copy()
            if clickData['points'][0]['label'] == 'BIG':
                df_subset['RoadTotal'] = df_subset['BIGVOLUME']
            elif clickData['points'][0]['label'] == 'CAR':
                df_subset['RoadTotal'] = df_subset['CARVOLUME'] 
            elif clickData['points'][0]['label'] == 'MOTOR':
                df_subset['RoadTotal'] = df_subset['MOTORVOLUME']
        # else: #Line Chart
    elif selectedData2 is not None and selectedData2['points'] != []: # Line Chart
            first = selectedData2['range']['x'][0]
            last = selectedData2['range']['x'][1]
            first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
            last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
            df_subset = df.copy()
            df_subset['Date'] = pd.to_datetime(df_subset['Date'])
            df_subset = df[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif clickData2 is not None: # Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
    else:
        df_subset = df.copy()

    if clickData is not None or selectedData2 is not None or clickData2 is not None: # Pie Chart or Line Chart or Bar Chart
        grouped = df_subset.groupby(['RoadName', 'Date', 'PositionLon', 'PositionLat'])['RoadTotal'].sum().reset_index()
        pivot_table = grouped.pivot_table(values='RoadTotal', index=['RoadName', 'PositionLon', 'PositionLat'], columns='Date', fill_value=0)

        Total = pivot_table.sum(axis=1)

        Color_list2 = pd.cut(Total, bins=bins, labels=["BLUE","GREEN", "YELLOW", "ORANGE", "RED"]).tolist()
        if clickData2 is not None:
            for color in Color_list2:
                if color == label:
                    color_counts[color] += 1
        else:
            for color in Color_list2:
                if color == "BLUE" or color == "GREEN" or color == "YELLOW" or color == "ORANGE" or color == "RED":
                    color_counts[color] += 1
        
    data = {'COLOR': list(color_counts.keys()), 'QUANTITY': list(color_counts.values())}
    total_flow = pd.DataFrame(data)
    barchart = px.bar(
        total_flow,
        labels=dict(x="Traffic Volume", y="COLOR LABEL"),
        y=total_flow["COLOR"],
        x=total_flow["QUANTITY"],
        color=total_flow["COLOR"],
        color_discrete_map={
            "BLUE": colors['blue'],
            "GREEN": colors['green'],
            "YELLOW": colors['yellow'],
            "ORANGE": colors['orange'],
            "RED": colors['red']
        }
    )
    barchart.update_layout(
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['fig_background'],
        font_color=colors['text'],
        title= {
            'text': 'Quantity Of Each Color',
            'x': 0.5,
            'y': 0.95
        },
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='closest',
        showlegend=False
    )
    return barchart

#---------------------------------------------Pie Chart callback function---------------------------------------------
@app.callback( Output('Pie Chart', 'figure'),
               Input('Bubble Map', 'selectedData'),
               Input('Line Chart', 'selectedData'),
               Input('Bar Chart', 'clickData'),
               Input('Pie Chart', 'clickData') )
def draw_PieChart(selectedData, selectedData2, clickData, clickData2):
    EACHVOLUME = {'BIG': 0, "CAR": 0, "MOTOR": 0}
    TargetRoadName = []
    if selectedData is None and selectedData2 is None and clickData is None and clickData2 is None: # Default
        df_subset = df.copy()
    elif selectedData is not None and selectedData2 is not None and clickData is not None and selectedData2['points'] != []: # Bubble Map and Line Chart and Bar Chart, Line Chart放最後
        label = clickData['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif selectedData is not None and selectedData2 is not None and selectedData2['points'] != []: # Bubble Map and Line Chart
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif clickData is not None and selectedData2 is not None and selectedData2['points'] != []: # Bar Chart and Line Chart
        label = clickData['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif selectedData is not None and clickData is not None: # Bubble Map and Bar Chart
        label = clickData['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
    elif selectedData is not None: # Bubble Map
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
    elif selectedData2 is not None and selectedData2['points'] != []: # Line Chart
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset = df.copy()
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif clickData is not None: # Bar Chart
        label = clickData['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
    else: 
        df_subset = df.copy()

    EACHVOLUME['BIG'] = df_subset["BIGVOLUME"].sum()
    EACHVOLUME['CAR'] = df_subset["CARVOLUME"].sum()
    EACHVOLUME['MOTOR'] = df_subset["MOTORVOLUME"].sum()
    if clickData2 is not None: # Pie Chart
        if clickData2['points'][0]['label'] == 'BIG':
            EACHVOLUME['CAR'] = 0
            EACHVOLUME['MOTOR'] = 0
        elif clickData2['points'][0]['label'] == 'CAR':
            EACHVOLUME['BIG'] = 0
            EACHVOLUME['MOTOR'] = 0
        elif clickData2['points'][0]['label'] == 'MOTOR':
            EACHVOLUME['CAR'] = 0
            EACHVOLUME['BIG'] = 0
    
    DataOfPie = pd.DataFrame({'CLASS':list(EACHVOLUME.keys()), 'FLOW':list(EACHVOLUME.values())})
    piechart = px.pie(
                        DataOfPie, 
                        names= DataOfPie["CLASS"], 
                        values= DataOfPie["FLOW"],
                        color= DataOfPie["CLASS"],
                        color_discrete_map={
                            "BIG": colors['BIG'],
                            "CAR": colors['CAR'],
                            "MOTOR": colors['MOTOR']
                        }
                    )
    piechart.update_layout(
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['fig_background'],
        font_color=colors['text'],
        title= {
            'text': 'Proportion Of Cars',
            'x': 0.5,
            'y': 0.95
        },
        hovermode= 'closest'
    )
    return piechart

#---------------------------------------------Heat Map callback function---------------------------------------------
@app.callback( Output('Heap Map', 'figure'),
            Input('Bubble Map', 'selectedData'),
            Input('Line Chart', 'selectedData'),
            Input('Pie Chart', 'clickData'),
            Input('Bar Chart', 'clickData'))
def draw_HeapMap(selectedData, selectedData2, clickData, clickData2):
    DataOfHM = pd.DataFrame(index=[], columns=[])

    if selectedData is None and selectedData2 is None and clickData is None and clickData2 is None: #Default
        df_subset = df.copy()
        df_subset['Total Volume'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData is not None and clickData2 is not None and selectedData2 is not None and selectedData2['points'] != []: # All
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['Total Volume'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['Total Volume'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['Total Volume'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData is not None and selectedData2 is not None and selectedData2['points'] != []: # Bubble Map and Pie Chart and Line Chart
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['Total Volume'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['Total Volume'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['Total Volume'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData is not None and clickData2 is not None: # Bubble Map and Pie Chart and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['Total Volume'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['Total Volume'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['Total Volume'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData2 is not None and selectedData2 is not None and selectedData2['points'] != []: # Bubble Map and Bar Chart and Line Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset['Total Volume'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    elif clickData is not None and clickData2 is not None and selectedData2 is not None and selectedData2['points'] != []: # Pie Chart and Bar Chart and Line Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['Total Volume'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['Total Volume'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['Total Volume'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData is not None: # Bubble Map and Pie Chart
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['Total Volume'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['Total Volume'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['Total Volume'] = df_subset['MOTORVOLUME']
    elif selectedData is not None and clickData2 is not None: # Bubble Map and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
        df_subset['Total Volume'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    elif selectedData is not None and selectedData2 is not None and selectedData2['points'] != []: # Bubble Map and Line Chart
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset['Total Volume'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    elif clickData is not None and clickData2 is not None: # Pie Chart and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['Total Volume'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['Total Volume'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['Total Volume'] = df_subset['MOTORVOLUME']
    elif clickData is not None and selectedData2 is not None and selectedData2['points'] != []: # Pie Chart and Line Chart
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset = df.copy()
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['Total Volume'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['Total Volume'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['Total Volume'] = df_subset['MOTORVOLUME']
    elif clickData2 is not None and selectedData2 is not None and selectedData2['points'] != []: # Bar Chart and Line Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset['Total Volume'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    elif selectedData is not None: # Bubble Map
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        df_subset['Total Volume'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    elif selectedData2 is not None and selectedData2['points'] != []: # Line Chart
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset = df.copy()
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        df_subset['Total Volume'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    elif clickData is not None: # Pie Chart
        df_subset = df.copy()
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['Total Volume'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['Total Volume'] = df_subset['CARVOLUME'] 
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['Total Volume'] = df_subset['MOTORVOLUME']
    elif clickData2 is not None: # Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        df_subset['Total Volume'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    else : #先不考慮交集
        df_subset = df.copy()
        df_subset['Total Volume'] = df_subset['BIGVOLUME'] + df_subset['CARVOLUME'] + df_subset['MOTORVOLUME']
    
    df_subset['InfoTime'] = pd.to_datetime(df_subset['InfoTime'], format='%Y-%m-%d %H:%M:%S')
    df_subset['Date'] = df_subset['InfoTime'].dt.date
    df_subset['Time'] = df_subset['InfoTime'].dt.hour.apply(time_def)
    df_subset['Weekday'] = df_subset['InfoTime'].dt.dayofweek.apply(week_def)


    DataOfHM = pd.pivot_table(df_subset, values='Total Volume', index=['Weekday'], columns=['Time'], aggfunc=np.sum, fill_value=0)
    heatmap = px.imshow(
                        DataOfHM,
                        labels=dict(y="Day of Week", x="Time of Day", color="Total Volume"),
                        x=list(DataOfHM.columns),
                        y=list(DataOfHM.index),
                        color_continuous_scale='Pinkyl'
    )
    
    heatmap.update_layout(
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['fig_background'],
        font_color=colors['text'],
        title= {
            'text': 'Total Volume Of Each Week',
            'x': 0.5,
            'y': 0.95
        },
        hovermode= 'closest'
    )
    return heatmap

#---------------------------------------------Line Chart callback function---------------------------------------------
@app.callback( Output('Line Chart', 'figure'),
            Input('Bubble Map', 'selectedData'),
            Input('Pie Chart', 'clickData'),
            Input('Bar Chart', 'clickData'),
            Input('Line Chart', 'selectedData'))
def draw_LineChart(selectedData, clickData, clickData2, selectedData2):
    TargetRoadName = []
    if selectedData is None and clickData is None and clickData2 is None and selectedData2 is None: # Default
        df_subset = df.copy()
    elif selectedData is not None and clickData2 is not None and selectedData2 is not None and selectedData2['points'] != []: # Bubble Map and Bar Chart and Line Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif selectedData is not None and clickData2 is not None: # Bubble Map and Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        TargetRoadName2 = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName2)].copy()
    elif selectedData is not None and selectedData2 is not None and selectedData2['points']: # Bubble Map and Line Chart
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif clickData2 is not None and selectedData2 is not None and selectedData2['points']: # Bar Chart and Line Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    elif selectedData is not None: # Bubble Map
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
    elif clickData2 is not None: # Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
    elif selectedData2 is not None and selectedData2['points']: # Line Chart選擇非空，因為Line Chart會更新兩次
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset = df.copy()
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
        print("OK")
    else:
        df_subset = df.copy()

    df_subset['BIGSPEED'] = df_subset['BIGSPEED'] * df_subset['BIGVOLUME']
    df_subset['CARSPEED'] = df_subset['CARSPEED'] * df_subset['CARVOLUME']
    df_subset['MOTORSPEED'] = df_subset['MOTORSPEED'] * df_subset['MOTORVOLUME']
    pivoted_df = pd.pivot_table(df_subset, values=["BIGSPEED", "CARSPEED", "MOTORSPEED", "BIGVOLUME", "CARVOLUME", "MOTORVOLUME"],
                                 index=["Date"], aggfunc=np.sum, fill_value=0)

    pivoted_df["BIG"] = pivoted_df["BIGSPEED"] / pivoted_df["BIGVOLUME"]
    pivoted_df["CAR"] = pivoted_df["CARSPEED"] / pivoted_df["CARVOLUME"]
    pivoted_df["MOTOR"] = pivoted_df["MOTORSPEED"] / pivoted_df["MOTORVOLUME"]

    if clickData is not None: # Pie Chart
        if clickData['points'][0]['label'] == 'BIG':
            data = 'BIG'
        elif clickData['points'][0]['label'] == 'CAR':
            data = 'CAR'
        elif clickData['points'][0]['label'] == 'MOTOR':
            data = 'MOTOR'
        linechart = px.line(pivoted_df, x=pivoted_df.index, y=[data],
                            labels=dict(x="Date", y="Average Speed"),
                            markers=True,
                            color_discrete_map={
                                data: colors[data]
                            },
                        )
    else:
        linechart = px.line(pivoted_df, x=pivoted_df.index, y=["BIG", "CAR", "MOTOR"],
                        labels=dict(x="Date", y="Average Speed"),
                        markers=True,
                        color_discrete_map={
                            "BIG": colors['BIG'],
                            "CAR": colors['CAR'],
                            "MOTOR": colors['MOTOR']
                        },
                       )

    linechart.update_layout(
        plot_bgcolor= colors['plot_background'],
        paper_bgcolor= colors['fig_background'],
        font_color=colors['text'],
        title= {
            'text': 'Average Speed Of Cars',
            'x': 0.5,
            'y': 0.95
        },
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode= 'closest'
    )
    return linechart

#---------------------------------------------Text callback function---------------------------------------------
def get_text_fig(data, title):
    # fig = go.Figure()
    # 除了go.Figure()畫文字框，也可以用px做
    fig = px.scatter()

    fig.add_annotation(
        text=data,
        showarrow=False,
        font=dict(size=27)
    )
    fig.update_layout(
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['fig_background'],
        font_color=colors['text'],
        title={
            'text': title,
        },
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

@app.callback(Output('Text 1', 'figure'),
            Output('Text 2', 'figure'),
            Input('Bubble Map', 'selectedData'),
            Input('Pie Chart', 'clickData'),
            Input('Bar Chart', 'clickData'),
            Input('Line Chart', 'selectedData'))
def draw_Text(selectedData, clickData, clickData2, selectedData2):
    df_subset = df.copy()
    if clickData2 is not None: # Bar Chart
        label = clickData2['points'][0]['label']
        df_subset = DataOfBM_df.copy()
        df_subset = df_subset[df_subset['Color'] == label]
        TargetRoadName = [get_roadname(point['PositionLon'], point['PositionLat']) for _, point in df_subset.iterrows()]
        df_subset = df[df['RoadName'].isin(TargetRoadName)].copy()
    if selectedData is not None: # Bubble Map
        TargetRoadName = [get_roadname(point['lon'], point['lat']) for point in selectedData['points']]
        df_subset = df_subset[df_subset['RoadName'].isin(TargetRoadName)].copy()
    if selectedData2 is not None and selectedData2['points'] != []: # Line Chart
        first = selectedData2['range']['x'][0]
        last = selectedData2['range']['x'][1]
        first = datetime.strptime(first, '%Y-%m-%d %H:%M:%S.%f')
        last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S.%f')
        df_subset['Date'] = pd.to_datetime(df_subset['Date'])
        df_subset = df_subset[(df_subset['Date'] >= first) & (df_subset['Date'] <= last)]
    if clickData is not None: # Pie Chart
        if clickData['points'][0]['label'] == 'BIG':
            df_subset['RoadTotal'] = df_subset['BIGVOLUME']
        elif clickData['points'][0]['label'] == 'CAR':
            df_subset['RoadTotal'] = df_subset['CARVOLUME']
        elif clickData['points'][0]['label'] == 'MOTOR':
            df_subset['RoadTotal'] = df_subset['MOTORVOLUME']

    # get the total volume of the selected road
    Total_Volume = df_subset['RoadTotal'].sum()
    Total_Volume = int(Total_Volume)
    if Total_Volume >= 1000000:
        Total_Volume = Total_Volume / 1000000
        Total_Volume = str(int(Total_Volume)) + 'M'
    elif Total_Volume >= 1000:
        Total_Volume = Total_Volume / 1000
        Total_Volume = str(int(Total_Volume)) + 'K'
    else:
        Total_Volume = str(Total_Volume)

    # get the avg speed of the selected road
    Average_Speed = df_subset['AVGSPEED'].mean()
    Average_Speed = int(Average_Speed)
    if Average_Speed >= 1000:
        Average_Speed = Average_Speed / 1000
        Average_Speed = str(int(Average_Speed)) + 'K'
    else:
        Average_Speed = str(Average_Speed)

    return [
        get_text_fig(Total_Volume, 'Total Volume'),
        get_text_fig(Average_Speed, 'Average Speed')
    ]

#---------------------------------------------Dash Board 版面---------------------------------------------
app.layout = html.Div(
    style={'height': '100%', 'width': '100%', 'backgroundColor': colors['background'], 'fontFamily': 'Times New Roman, sans-serif', 'padding': '2%'},
    children=[        
        html.H1("DashBoard", style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '2%', 'fontSize': '40px'}),
        html.Div(
            style={'height': '100vh', 'width': '100%', 'display': 'flex', 'flexWrap': 'wrap'},
            children=[
                html.Div( #左邊
                    style={'height': '100vh', 'width': '34%', 'float': 'left', 'marginRight': '1%'},
                    children=[
                        html.Div(
                            style={'width': '100%', 'height': '50%', 'marginBottom': '1%'},
                            children=[dcc.Graph(id='Bubble Map', style={'height': '50vh'})]
                        ),
                        html.Div(
                            style={'width': '100%', 'height': '50%', 'marginBottom': '1%'},
                            children=[dcc.Graph(id='Choropleth Map', style={'height': '50vh'})]
                        )
                    ]
                ),
                html.Div( #右邊
                    style={'height': '100vh', 'width': '65%', 'float': 'right', 'display': 'flex', 'flexWrap': 'wrap'},
                    children=[
                        html.Div(
                            style={'width': '17%', 'height': '49%', 'marginRight': '1%'},
                            children=[
                                dcc.Graph(id='Text 1', style={'height': '50%'}),
                                dcc.Graph(id='Text 2', style={'height': '50%'})
                            ]
                        ),
                        html.Div(
                            style={'width': '81%', 'height': '49%', 'marginBottom': '1%'},
                            children=[dcc.Graph(id='Line Chart', style={'height': '100%'})]
                        ),
                        html.Div(
                            style={'width': '29%', 'height': '50%', 'marginRight': '1%'},
                            children=[dcc.Graph(id='Bar Chart', style={'height': '100%'})]
                        ),
                        html.Div(
                            style={'width': '34%', 'height': '50%', 'marginRight': '1%'},
                            children=[dcc.Graph(id='Heap Map', style={'height': '100%'})]
                        ),
                        html.Div(
                            style={'width': '35%', 'height': '50%'},
                            children=[dcc.Graph(id='Pie Chart', style={'height': '100%'})]
                        )
                    ]
                ),
            ]
        )
    ]
)


#---------------------------------------------html update---------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)