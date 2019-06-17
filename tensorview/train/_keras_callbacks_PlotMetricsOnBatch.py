from collections import defaultdict

from tensorflow.io.gfile import exists
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pyecharts.charts import Line
from pyecharts.charts import Page
from pyecharts.charts import Timeline
from pyecharts import options as opts
from pandas import Series

__all__ = ['PlotMetricsOnBatch']

class PlotMetricsOnBatch(Callback):
    """
    Arguments:
        metrics_name : list, Customized evaluation indicator name list,
                      sequentially created according to loss function and measurement function;
        columns : int, default 2，The number of sub graphs that the width of metrics
                 visualiztion image to accommodate at most;
        iter_num : int, default None, Pre-specify the maximum value of x-axis in each
                  sub-picture to indicate the maximum number of batch training;
        wait_num : int, default 1, Indicates how many batches are drawn each time a graph is drawn;
        figsize : tuple，default None，Represents the customize image size;
        cell_size : tuple, default (6, 4)，Indicates the customize image size, which is used when figsize=None;
        valid_fmt : str, default "val_{}", The string preceding the underscore is used to
                   implement the validation set indicator naming;
        eval_batch_num : int, default None, Indicates how many batches are evaluated for each validation set;
    """
    def __init__(self, metrics_name, valid_x=None, valid_y=None, valid_steps=None,
                 valid_batch_size=None, valid_sample_weight=None,
                 columns=2, iter_num=None, wait_num=1, figsize=None,
                 cell_size=(6, 4), valid_fmt="val_{}", eval_batch_num=None,
                 visual_name='model_visual', visual_name_gif='model_visual_gif',
                 visual_path=None, visual_image=True, visual_gif=False):
        super(PlotMetricsOnBatch, self).__init__()
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.valid_steps = valid_steps
        self.valid_batch_size = valid_batch_size
        self.valid_sample_weight = valid_sample_weight
        self.metrics_name = metrics_name
        self.metrics = metrics_name
        self.columns = columns
        self.iter_num = iter_num
        self.wait_num = wait_num
        self.figsize = figsize
        self.cell_size = cell_size
        self.valid_fmt = valid_fmt
        self.batch_logs = defaultdict(list)
        self.batch_num = 0
        self.eval_batch_num = eval_batch_num
        self.visual_name = visual_name
        self.visual_name_gif = visual_name_gif
        self.visual_path = visual_path
        self.visual_image = visual_image
        self.visual_gif = visual_gif

    def on_batch_end(self, batch, logs=None):
        self.batch_num += 1
        for i in ['batch', 'size']:
            logs.pop(i)
        for old_name, new_name in zip(self.model.metrics_names, self.metrics_name):
            logs[new_name] = logs.pop(old_name)
        if self.eval_batch_num is not None:
            self.new_val_name = [self.valid_fmt.split('_')[0]+'_'+i for i in self.metrics_name]
            self.metrics = self.new_val_name+self.metrics_name
            if self.batch_num%self.eval_batch_num==0:
                loss_list = self.model.evaluate(x=self.valid_x, y=self.valid_y, batch_size=self.valid_batch_size, verbose=0, sample_weight=self.valid_sample_weight, steps=self.valid_steps)
                for loss_value, new_name in zip(loss_list, self.new_val_name):
                    logs[new_name] = loss_value
        if self.figsize is None:
            self.figsize = (self.columns*self.cell_size[0], ((len(self.metrics)+1)//self.columns+1)*self.cell_size[1])
        for metric in logs:
            self.batch_logs[metric] += [logs[metric]]
        self.draw(metrics=self.metrics, logs=self.batch_logs, batch=self.batch_num, columns=self.columns,
                  iter_num=self.iter_num, wait_num=self.wait_num, eval_batch_num=self.eval_batch_num,
                  figsize=self.figsize, cell_size=self.cell_size, valid_fmt=self.valid_fmt)

    def on_train_end(self, logs=None):
        if self.visual_image:
            self.visual(name=self.visual_name, path=self.visual_path, gif=False)
        if self.visual_gif:
            self.visual(name=self.visual_name_gif, path=self.visual_path, gif=True)
        self.draw(metrics=self.metrics, logs=self.batch_logs, batch=self.wait_num, columns=self.columns,
                  iter_num=self.iter_num, wait_num=self.wait_num, eval_batch_num=self.eval_batch_num,
                  figsize=self.figsize, cell_size=self.cell_size, valid_fmt=self.valid_fmt)
    
    def draw(self, metrics, logs, batch, columns, iter_num, wait_num, eval_batch_num, figsize, cell_size, valid_fmt):
        logit = batch%wait_num==0 if eval_batch_num is None else (batch%wait_num==0)|(batch%eval_batch_num==0)
        if logit:
            clear_output(wait=True)
            plt.figure(figsize=figsize)
            for metric_id, metric in enumerate(metrics):
                plt.subplot((len(metrics)+1)//columns+1, columns, metric_id+1)
                if iter_num is not None:
                    plt.xlim(1, iter_num)
                if valid_fmt.split('_')[0] not in metric:
                    plt.plot(range(1, len(logs[metric])+1), logs[metric], label="train")
                else:
                    plt.plot(range(eval_batch_num, len(logs[metric])*eval_batch_num+1, eval_batch_num),
                             logs[metric], label=valid_fmt.split('_')[0])
                plt.title(metric)
                plt.xlabel('batch')
                plt.legend(loc='center right')
            plt.tight_layout()
            plt.show()
    
    def visual(self, name='model_visual', path=None, gif=False):
        if path is not None:
            assert exists(path), "`path` not exist."
            file = path+'/'+'{}.html'.format(name)
        else:
            file = '{}.html'.format(name)
        page = Page(interval=1, layout=Page.SimplePageLayout)
        plot_list = []
        width_len = '750px'
        height_len = '450px'
        for metric_id, metric in enumerate(self.metrics):
            if not gif:
                if metric.split('_')[0]==self.valid_fmt.split('_')[0]:
                    continue
                line = Line(opts.InitOpts(width=width_len, height=height_len))
                line = line.add_xaxis(list(range(1, self.batch_num+1)))
                line = line.add_yaxis('train', Series(self.batch_logs[metric]).round(4).tolist(), is_smooth=True)
                if self.valid_fmt.format(metric) in self.batch_logs:
                    test_value = []
                    for i in Series(self.batch_logs[self.valid_fmt.format(metric)]).round(4).tolist():
                        test_value += [None]*(self.eval_batch_num-1)+[i]
                    line = line.add_yaxis(self.valid_fmt.split('_')[0], test_value, is_smooth=True, is_connect_nones=True)
                line = line.set_series_opts(label_opts=opts.LabelOpts(is_show=False),
                                            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='max_value'),
                                                                                    opts.MarkPointItem(type_='min', name='min_value')]))
                line = line.set_global_opts(title_opts=opts.TitleOpts(title=metric),
                                            xaxis_opts=opts.AxisOpts(name='batch', name_location='center', is_scale=True),
                                            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                                            toolbox_opts=opts.ToolboxOpts())
                plot_list.append(line)
            else:
                if metric.split('_')[0]==self.valid_fmt.split('_')[0]:
                    continue
                timeline = Timeline(opts.InitOpts(width=width_len, height=height_len)).add_schema(play_interval=100, is_auto_play=True)
                for i in range(1, self.batch_num+1):
                    line = Line(opts.InitOpts(width=width_len, height=height_len))
                    line = line.add_xaxis(list(range(1, i+1)))
                    line = line.add_yaxis('train', Series(self.batch_logs[metric])[:i].round(4).tolist(), is_smooth=True)
                    if self.valid_fmt.format(metric) in self.batch_logs:
                        test_value = []
                        for j in Series(self.batch_logs[self.valid_fmt.format(metric)]).round(4).tolist():
                            test_value += [None]*(self.eval_batch_num-1)+[j]
                        line = line.add_yaxis(self.valid_fmt.split('_')[0], test_value, is_smooth=True, is_connect_nones=True)
                    line = line.set_series_opts(label_opts=opts.LabelOpts(is_show=False),
                                                markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='max_value'),
                                                                                        opts.MarkPointItem(type_='min', name='min_value')]))
                    line = line.set_global_opts(title_opts=opts.TitleOpts(title=metric),
                                                xaxis_opts=opts.AxisOpts(name='batch', name_location='center', is_scale=True))
                    timeline.add(line, str(i))
                plot_list.append(timeline)
        page.add(*plot_list).render(file)
        return file
