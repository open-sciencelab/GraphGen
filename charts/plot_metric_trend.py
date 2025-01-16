import pandas as pd
import plotly.express as px

def plot_metric_trend(df, x, y):
    fig = px.line(df, x=x, y=y, color='epoch', markers=True)

    fig.update_xaxes(tickvals=df[x], ticktext=[f'{int(val * 100)}%' for val in df[x].unique()])

    fig.update_layout(xaxis_title='Percentage',
                      yaxis_title='Agr Average Improvement',
                      plot_bgcolor='rgba(0,0,0,0)',  # 设置绘图区域背景为透明
                      paper_bgcolor='rgba(0,0,0,0)',  # 设置整个图表背景为透明
                      xaxis={
                            'showgrid': True,
                            'gridcolor': 'lightgray',
                            'linewidth': 1,
                            'linecolor': 'lightgray',
                            'showline': True,  # 显示x轴线
                            'mirror': True,  # 在对面也显示轴线
                      },
                      yaxis={
                            'showgrid': True,
                            'gridcolor': 'lightgray',
                            'linewidth': 1,
                            'linecolor': 'lightgray',
                            'showline': True,  # 显示y轴线
                            'mirror': True,  # 在对面也显示轴线
                      })

    fig.show()
    # TODO: 添加baseline的线

if __name__ == "__main__":
    epoch_1 = {
        "percentage": [0.05, 0.15, 0.3, 0.4, 0.6, 1],
        "agr average": [2.16, 0.48, 1.62, 1.77, 2.99, 2.19]
    }
    epoch_2 = {
        "percentage": [0.05, 0.15, 0.3, 0.4, 0.6, 1],
        "agr average": [1.78, 0.52, 2.23, 2.21, 1.62, 1.53]
    }
    epoch_3 = {
        "percentage": [0.05, 0.15, 0.3, 0.4, 0.6, 1],
        "agr average": [1.39, 0.21, 1.60, 1.76, 1.31, 1.80]
    }

    df_epoch_1 = pd.DataFrame(epoch_1)
    df_epoch_1['epoch'] = 'Epoch 1'

    df_epoch_2 = pd.DataFrame(epoch_2)
    df_epoch_2['epoch'] = 'Epoch 2'

    df_epoch_3 = pd.DataFrame(epoch_3)
    df_epoch_3['epoch'] = 'Epoch 3'

    df_combined = pd.concat([df_epoch_1, df_epoch_2, df_epoch_3])

    plot_metric_trend(df_combined, x='percentage', y='agr average')
