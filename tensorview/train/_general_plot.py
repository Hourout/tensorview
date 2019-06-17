from collections import defaultdict

from tensorflow.io.gfile import exists
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pyecharts.charts import Line
from pyecharts.charts import Page
from pyecharts.charts import Timeline
from pyecharts import options as opts
from pandas import Series

__all__ = ['PlotMetrics']

class params(object):
    columns = None
    iter_num = None
    mode = None
    wait_num = None
    figsize = None
    cell_size = None
    valid_fmt = None
    logs = None
    xlabel = None
    polt_num = None
    frames = None
    metrics = None

class PlotMetrics():
    """
    Arguments:
        columns : int, default 2, The number of sub graphs that the width of metrics
                 visualiztion image to accommodate at most;
        iter_num : int, default None, Pre-specify the maximum value of x-axis in each
                  sub-picture to indicate the maximum number of batch or epoch training;
        mode : int, default 1, 1 means the x-axis name is 'batch', 0 means the x-axis name is 'epoch';
        wait_num : int, default 1, Indicates how many batches or epochs are drawn
                  each time a graph is drawn;
        figsize : tuple, default None，Represents the customize image size;
        cell_size : tuple, default (6, 4), Indicates the customize image size,
                   which is used when figsize=None;
        valid_fmt : str, default "val_{}",The string preceding the underscore is used to
                   instruction the training and validation is displayed together in the
                   same sub graph. The training indicator is not required to have a prefix.
                   The validation indicator prefix is 'val' in the "val_{}";
    """
    def __init__(self, columns=2, iter_num=None, mode=1, wait_num=1, figsize=None,
                 cell_size=(6, 4), valid_fmt="val_{}"):
        self.params = params
        self.params.columns = columns
        self.params.iter_num = iter_num
        self.params.mode = mode
        self.params.wait_num = wait_num
        self.params.figsize = figsize
        self.params.cell_size = cell_size
        self.params.valid_fmt = valid_fmt
        self.params.logs = defaultdict(list)
        self.params.xlabel = {0:'epoch', 1:'batch'}
        self.params.polt_num = 0
        self.params.frames = []

    def update(self, log):
        """
        Arguments:
        log : dict, name and value of loss or metrics;
        """
        self.params.metrics = list(filter(lambda x: self.params.valid_fmt.split('_')[0] not in x.lower(), log))
        if self.params.figsize is None:
            self.params.figsize = (self.params.columns*self.params.cell_size[0],
                                   ((len(self.params.metrics)+1)//self.params.columns+1)*self.params.cell_size[1])
        for metric in log:
            self.params.logs[metric] += [log[metric]]
        self.params.polt_num += 1

    def draw(self):
        """
        Arguments:
        save_image_path : str, if save_image=True, train end save last image to path;
        save_gif : bool, default False, if save_gif=True, train end save all image to gif;
        save_gif_path : str, if save_gif=True, train end save gif to path;
        """
        if self.params.polt_num%self.params.wait_num==0:
            clear_output(wait=True)
            figure = plt.figure(figsize=self.params.figsize)
            for metric_id, metric in enumerate(self.params.metrics):
                plt.subplot((len(self.params.metrics)+1)//self.params.columns+1, self.params.columns, metric_id+1)
                if self.params.iter_num is not None:
                    plt.xlim(1, self.params.iter_num)
                plt.plot(range(1, len(self.params.logs[metric])+1), self.params.logs[metric], label="train")
                if self.params.valid_fmt.format(metric) in self.params.logs:
                    plt.plot(range(1, len(self.params.logs[metric])+1),
                             self.params.logs[self.params.valid_fmt.format(metric)],
                             label=self.params.valid_fmt.split('_')[0])
                plt.title(metric)
                plt.xlabel(self.params.xlabel[self.params.mode])
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
        for metric_id, metric in enumerate(self.params.metrics):
            if not gif:
                line = Line(opts.InitOpts(width=width_len, height=height_len))
                line = line.add_xaxis(list(range(1, self.params.polt_num+1)))
                line = line.add_yaxis('train', Series(self.params.logs[metric]).round(4).tolist(), is_smooth=True)
                if self.params.valid_fmt.format(metric) in self.params.logs:
                    line = line.add_yaxis(self.params.valid_fmt.split('_')[0],
                                          Series(self.params.logs[self.params.valid_fmt.format(metric)]).round(4).tolist(), is_smooth=True)
                line = line.set_series_opts(label_opts=opts.LabelOpts(is_show=False),
                                            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='最大值'),
                                                                                    opts.MarkPointItem(type_='min', name='最小值')]))
                line = line.set_global_opts(title_opts=opts.TitleOpts(title=metric),
                                            xaxis_opts=opts.AxisOpts(name=self.params.xlabel[self.params.mode],
                                                                     name_location='center', is_scale=True),
                                            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                                            toolbox_opts=opts.ToolboxOpts())
                plot_list.append(line)
            else:
                timeline = Timeline(opts.InitOpts(width=width_len, height=height_len)).add_schema(play_interval=100, is_auto_play=True)
                for i in range(1, self.params.polt_num+1):
                    line = Line(opts.InitOpts(width=width_len, height=height_len))
                    line = line.add_xaxis(list(range(1, i+1)))
                    line = line.add_yaxis('train', Series(self.params.logs[metric])[:i].round(4).tolist(), is_smooth=True)
                    if self.params.valid_fmt.format(metric) in self.params.logs:
                        line = line.add_yaxis(self.params.valid_fmt.split('_')[0],
                                              Series(self.params.logs[self.params.valid_fmt.format(metric)])[:i].round(4).tolist(), is_smooth=True)
                    line = line.set_series_opts(label_opts=opts.LabelOpts(is_show=False),
                                            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max', name='最大值'),
                                                                                    opts.MarkPointItem(type_='min', name='最小值')]))
                    line = line.set_global_opts(title_opts=opts.TitleOpts(title=metric),
                                            xaxis_opts=opts.AxisOpts(name=self.params.xlabel[self.params.mode],
                                                                     name_location='center', is_scale=True))
                    timeline.add(line, str(i))
                plot_list.append(timeline)
        page.add(*plot_list).render(file)
        return file
