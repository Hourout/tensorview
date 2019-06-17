import pandas as pd
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts

__all__ = ['statistics']

def statistics(model, jupyter=True, path='Kolmogorov-Smirnov Curve.html', title="Model Summary", subtitle=""):
    t = pd.DataFrame([[i.name, i.__class__.__name__, i.trainable, i.dtype, i.input_shape, i.output_shape, i.count_params()] for i in model.layers],
                     columns=['layer_custom_name', 'layer_object_name', 'trainable', 'dtype', 'input_shape', 'output_shape', 'params'])
#     t['output_memory(MB)'] = (t.output_shape.map(lambda x:sum([reduce(lambda y,z:y*z, i[1:]) for i in x]) if isinstance(x, list) else reduce(lambda y,z:y*z, x[1:]))
#                        *t.dtype.map(lambda x:int(re.sub("\D", "", x))))/32#/1024/1024)
    t.loc['total'] = ['', '', '', '', '', '', t.params.sum()]
    t['input_shape'] = t.input_shape.map(lambda x:str(x).replace("),(", "),\n(") if isinstance(x, list) else x)
    t = t.reset_index().rename(columns={'index':''})
    for i in t.columns:
        t[i] = t[i].astype(str)
    table = Table()
    headers = t.columns.tolist()
    rows = t.values.tolist()
    table.add(headers, rows).set_global_opts(title_opts=ComponentTitleOpts(title=title, subtitle=subtitle))
    return table.render_notebook() if jupyter else table.render(path)
