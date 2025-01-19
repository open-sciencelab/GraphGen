import pandas as pd
import plotly.express as px
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_metric_trend(df, x, y):

    fig = px.line(df, x=x, y=y, color='max length', markers=True, color_discrete_sequence=['#925EB0', '#7E99F4', '#CC7C71', '#7AB656']) # A5AEB7

    fig.update_xaxes(tickvals=df[x], ticktext=[f'{int(val * 100)}%' for val in df[x].unique()])

    avg = df.groupby(x)[y].mean().reset_index()
    avg['max length'] = 'Average'

    x_smooth = np.linspace(avg[x].min(), avg[x].max(), 500)
    spline = make_interp_spline(avg[x], avg[y], k=3)  # k=3 for cubic spline
    y_smooth = spline(x_smooth)

    # Add the smooth average line to the plot
    fig.add_scatter(x=x_smooth, y=y_smooth, mode='lines', name='Smooth Average',
                    line={
                        'color': 'red',
                        'width': 4,
                        'dash': 'solid'
                    })

    fig.update_layout(xaxis_title='Percentage',
                      yaxis_title='Agr Average',
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      xaxis={
                            'showgrid': True,
                            'gridcolor': 'lightgray',
                            'linewidth': 1,
                            'linecolor': 'lightgray',
                            'showline': True,
                            'mirror': True,
                      },
                      yaxis={
                            'showgrid': True,
                            'gridcolor': 'lightgray',
                            'linewidth': 1,
                            'linecolor': 'lightgray',
                            'showline': True,
                            'mirror': True,
                      })

    fig.show()

if __name__ == "__main__":
    data = {
        "max length 512": {
            "percentage": [0, 0.05, 0.15, 0.3, 0.4, 0.6, 0.8, 1],
            "agr average": [0, 0.79, 2.078, 1.755, 1.247, 3.447, 2.967, 2.175]
        },
        "max length 1024": {
            "percentage": [0, 0.05, 0.15, 0.3, 0.4, 0.6, 0.8, 1],
            "agr average": [0, 1.266, 1.399, 1.723, 1.247, 2.581, 2.581, 1.291] # TODO: 0.4, 0.6
        },
        "max length 1536": {
            "percentage": [0, 0.05, 0.15, 0.3, 0.4, 0.6, 0.8, 1],
            "agr average": [0, 0.983, 1.622, 2.563, 1.453, 1.326, 1.225, 1.885]
        },
        "max length 2048": {
            "percentage": [0, 0.05, 0.15, 0.3, 0.4, 0.6, 0.8, 1],
            "agr average": [0, 0.365, 0.967, 2.231, 1.256, 1.616, 2.052, 1.529]
        }
    }

    df_list = []
    for length, values in data.items():
        df_temp = pd.DataFrame(values)
        df_temp['max length'] = length
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)

    plot_metric_trend(df, 'percentage', 'agr average')

