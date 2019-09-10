from pyecharts.charts import Page, Scatter
from pyecharts import options as opts


__all__ = ['weights_visualize']

def scatter_base(value, label, title, subtitle):
    c = (Scatter()
         .add_xaxis(list(range(len(value.numpy().reshape(-1)))))
         .add_yaxis(label, value.numpy().reshape(-1).tolist(), symbol_size=2,)
         .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
         .set_global_opts(title_opts=opts.TitleOpts(title=title, subtitle=subtitle)))
    return c

def weights_visualize(model, layer_name):
    charts = []
    for name in layer_name:
        for weights in model.weights:
            if name in weights.name:
                label = weights.name.split('/')[-1]
                title = name
                subtitle = 'shape='+str(model.weights[0].shape)
                charts.append(scatter_base(weights, label, title, subtitle))
    page = Page().add(*charts)
    return page.render()
