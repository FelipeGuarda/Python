# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib as mpl

###########################
# Increase the animation size limit
###########################
# For example, allow up to 200 MB for the animation embed:
mpl.rcParams['animation.embed_limit'] = 200_000_000  

###########################
# 1) READ AND MERGE THE DATA
###########################
header_row = 1
skip_rows = [2, 3]

data1 = pd.read_csv(
    "G:\\Shared drives\\FMA-G\\C-CONSERVACIÓN\\C1-BOSQUE PEHUEN\\C1.5_DATOS\\1. Estacion Meteorológica\\Datos estación meteorológica\\CR800Series_Table 01122020_11042022.dat",
    header=header_row,
    skiprows=skip_rows,
    low_memory=False
)
data2 = pd.read_csv(
    "G:\\Shared drives\\FMA-G\\C-CONSERVACIÓN\\C1-BOSQUE PEHUEN\\C1.5_DATOS\\1. Estacion Meteorológica\\Datos estación meteorológica\\CR800Series_Table 03082021_12122022.dat",
    header=header_row,
    skiprows=skip_rows,
    low_memory=False
)
data3 = pd.read_csv(
    "G:\\Shared drives\\FMA-G\\C-CONSERVACIÓN\\C1-BOSQUE PEHUEN\\C1.5_DATOS\\1. Estacion Meteorológica\\Datos estación meteorológica\\CR800Series_Table 08112022_18032024.dat",
    header=header_row,
    skiprows=skip_rows,
    low_memory=False
)
data4 = pd.read_csv(
    "G:\\Shared drives\\FMA-G\\C-CONSERVACIÓN\\C1-BOSQUE PEHUEN\\C1.5_DATOS\\1. Estacion Meteorológica\\Datos estación meteorológica\\CR800Series_Table 14012023_24052024.dat",
    header=header_row,
    skiprows=skip_rows,
    low_memory=False
)
data5 = pd.read_csv(
    "G:\\Shared drives\\FMA-G\\C-CONSERVACIÓN\\C1-BOSQUE PEHUEN\\C1.5_DATOS\\1. Estacion Meteorológica\\Datos estación meteorológica\\CR800Series_Table 21092018_30012020.dat",
    header=header_row,
    skiprows=skip_rows,
    low_memory=False
)
data6 = pd.read_csv(
    "G:\\Shared drives\\FMA-G\\C-CONSERVACIÓN\\C1-BOSQUE PEHUEN\\C1.5_DATOS\\1. Estacion Meteorológica\\Datos estación meteorológica\\CR800Series_Table1_13092019_21012021.dat",
    header=header_row,
    skiprows=skip_rows,
    low_memory=False
)

df_list = [data1, data2, data3, data4, data5, data6]
merged_df = pd.concat(df_list, ignore_index=True)

# Drop duplicates based on 'RECORD' and 'TIMESTAMP'
merged_df.drop_duplicates(subset=['RECORD', 'TIMESTAMP'], inplace=True)

###########################
# 2) CREATE A DAY-LEVEL DATE COLUMN
###########################
merged_df['TIMESTAMP'] = pd.to_datetime(merged_df['TIMESTAMP'])
merged_df['Year'] = merged_df['TIMESTAMP'].dt.year
merged_df['Month'] = merged_df['TIMESTAMP'].dt.month
merged_df['Day'] = merged_df['TIMESTAMP'].dt.day
merged_df['Date'] = pd.to_datetime(merged_df[['Year','Month','Day']])

###########################
# 3) AGGREGATE DAILY RAINFALL
###########################
daily_rainfall = (
    merged_df.groupby('Date')['Rain_mm_Tot']
    .sum()
    .reset_index()
    .rename(columns={'Rain_mm_Tot': 'Total_Rain_mm'})
)

# Mark rainy days if total > 2 mm
daily_rainfall['Is_Rainy'] = daily_rainfall['Total_Rain_mm'].apply(lambda x: 1 if x > 2 else 0)

# Sort by date to ensure chronological order
daily_rainfall.sort_values('Date', inplace=True)

# Instead of a simple cumsum, let's reset the counter to 0 when it rains.
days_no_rain = 0
days_without_rain_list = []
for _, row in daily_rainfall.iterrows():
    if row['Is_Rainy'] == 1:
        days_no_rain = 0  # reset when it rains
    else:
        days_no_rain += 1
    days_without_rain_list.append(days_no_rain)

daily_rainfall['Days_Without_Rain'] = days_without_rain_list

# Merge back into merged_df
merged_df = pd.merge(
    merged_df,
    daily_rainfall[['Date','Is_Rainy','Days_Without_Rain']],
    on='Date',
    how='left'
)

###########################
# 4) SCORING SYSTEM
###########################
def calculate_temperature_score(temp):
    if temp < 0:
        return 2.7
    elif 0 <= temp < 6:
        return 5.4
    elif 6 <= temp < 11:
        return 8.1
    elif 11 <= temp < 16:
        return 10.8
    elif 16 <= temp < 21:
        return 13.5
    elif 21 <= temp < 26:
        return 16.2
    elif 26 <= temp < 31:
        return 18.9
    elif 31 <= temp < 36:
        return 21.6
    else:
        return 25

def calculate_humidity_score(humidity):
    if 0 <= humidity < 11:
        return 25
    elif 11 <= humidity < 21:
        return 22.5
    elif 21 <= humidity < 31:
        return 20
    elif 31 <= humidity < 41:
        return 17.5
    elif 41 <= humidity < 51:
        return 15
    elif 51 <= humidity < 61:
        return 12.5
    elif 61 <= humidity < 71:
        return 10
    elif 71 <= humidity < 81:
        return 7.5
    elif 81 <= humidity < 91:
        return 5
    else:
        return 2.5

def calculate_wind_speed_score(wind_speed):
    if wind_speed < 0.277:
        return 3.125
    elif 0.277 <= wind_speed < 1.66:
        return 6.25
    elif 1.66 <= wind_speed < 3.055:
        return 9.375
    elif 3.055 <= wind_speed < 4.44:
        return 12.5
    elif 4.44 <= wind_speed < 5.83:
        return 15.625
    elif 5.83 <= wind_speed < 7.22:
        return 18.75
    elif 7.22 <= wind_speed < 8.61:
        return 21.875
    else:
        return 25

def calculate_days_without_rain_score(days):
    if days < 1:
        return 2.5
    elif 1 <= days < 6:
        return 5
    elif 6 <= days < 11:
        return 7.5
    elif 11 <= days < 16:
        return 10
    elif 16 <= days < 21:
        return 12.5
    elif 21 <= days < 26:
        return 15
    elif 26 <= days < 31:
        return 17.5
    elif 31 <= days < 36:
        return 20
    elif 36 <= days < 41:
        return 22.5
    else:
        return 25

def calculate_scores(row):
    t_score = calculate_temperature_score(row['AirTC_Avg'])
    h_score = calculate_humidity_score(row['RH_Avg'])
    w_score = calculate_wind_speed_score(row['WS_ms_Avg'])
    r_score = calculate_days_without_rain_score(row['Days_Without_Rain'])
    total_score = t_score + h_score + w_score + r_score
    return pd.Series([t_score, h_score, w_score, r_score, total_score])

merged_df[['temperature_score','humidity_score','wind_speed_score','days_without_rain_score','Total_Score']] = (
    merged_df.apply(calculate_scores, axis=1)
)

###########################
# 5) HIGHEST RISK PER DAY
###########################
highest_risk_per_day = merged_df.loc[
    merged_df.groupby('Date')['Total_Score'].idxmax()
].copy()

###########################
# 6) STRAIGHT-LINE RADIAL PLOT
###########################
fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': 'polar'})

# We have 4 variables => 4 angles
angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist()
angles += angles[:1]  # close loop
line, = ax.plot([], [], linewidth=2)

unique_dates = highest_risk_per_day['Date'].unique()

def get_radar_points(values):
    """Return (angles, closed_values) for a straight-line radar shape."""
    closed_values = values + [values[0]]
    return angles, closed_values

def update(date):
    # Filter the row with the highest risk for 'date'
    row = highest_risk_per_day[highest_risk_per_day['Date'] == date]
    
    # Gather the 4 scores
    vals = [
        row['temperature_score'].values[0],
        row['humidity_score'].values[0],
        row['wind_speed_score'].values[0],
        row['days_without_rain_score'].values[0]
    ]
    x_points, y_points = get_radar_points(vals)

    # Update the line
    line.set_data(x_points, y_points)

    # Color by total risk
    total_score = row['Total_Score'].values[0]
    if total_score >= 91:
        line_color = '#c71a15'
    elif total_score >= 81:
        line_color = '#d35907'
    elif total_score >= 71:
        line_color = '#d38107'
    elif total_score >= 61:
        line_color = '#d3aa07'
    elif total_score >= 51:
        line_color = '#d0d307'
    elif total_score >= 41:
        line_color = '#248107'
    elif total_score >= 31:
        line_color = '#07c5d3'
    elif total_score >= 21:
        line_color = '#0795d3'
    elif total_score >= 11:
        line_color = '#0763d3'
    else:
        line_color = '#a067bd'
    line.set_color(line_color)

    # White background 
    ax.set_facecolor('white')

    # Update the title to show the day
    ax.set_title(f"Date: {date.strftime('%Y-%m-%d')}", fontsize=14)

    return line,

# Set radial axis limit to 25, since each variable can have a max of 25
ax.set_ylim(0, 25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['Temperature','Humidity','Wind','Days w/o Rain'])

# Create the animation
ani = FuncAnimation(
    fig,
    update,
    frames=unique_dates,
    interval=300,  # adjust speed as desired
    blit=False     # ensure the date updates every frame
)

# Display the animation in the notebook
HTML(ani.to_jshtml())
plt.show()
