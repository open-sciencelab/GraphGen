from collections import defaultdict
import plotly.graph_objects as go

def plot_loss_distribution(stats: list[dict]):
    """
    Plot the distribution of edges' loss.

    :return fig
    """

    if not stats:
        return go.Figure()

    max_loss = max(item['average_loss'] for item in stats)
    bin_numbers = 50
    bin_size = max_loss / bin_numbers

    length_distribution = defaultdict(int)

    for item in stats:
        bin_start = (item['average_loss'] // bin_size) * bin_size
        bin_key = f"{bin_start}-{bin_start + bin_size}"
        length_distribution[bin_key] += 1

    sorted_bins = sorted(length_distribution.keys(),
                         key=lambda x: float(x.split('-')[0]))

    fig = go.Figure(data=[
        go.Bar(
            x=sorted_bins,
            y=[length_distribution[bin_] for bin_ in sorted_bins],
            text=[length_distribution[bin_] for bin_ in sorted_bins],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Distribution of Loss',
        xaxis_title='Loss Range',
        yaxis_title='Count',
        bargap=0.2,
        showlegend=False
    )

    if len(sorted_bins) > 10:
        fig.update_layout(
            xaxis={
                'tickangle': 45,
                'tickmode': 'array',
                'ticktext': sorted_bins[::2],
                'tickvals': list(range(len(sorted_bins)))[::2]
            }
        )
    return fig