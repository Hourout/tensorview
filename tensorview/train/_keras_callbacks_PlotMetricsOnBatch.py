from collections import defaultdict

import imageio
from tensorflow.io import gfile
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from IPython.display import clear_output


def draw(metrics, logs, batch, columns, iter_num, wait_num, eval_batch_num, figsize, cell_size, valid_fmt,
         save_image=False, save_image_path=None, save_gif=False, save_gif_path=None):
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
                plt.plot(range(eval_batch_num, len(logs[metric])*eval_batch_num+1, eval_batch_num), logs[metric], label=valid_fmt.split('_')[0])
            plt.title(metric)
            plt.xlabel('batch')
            plt.legend(loc='center right')
        plt.tight_layout()
        if save_image:
            if save_image_path is not None:
                plt.savefig(save_image_path, bbox_inches='tight')
        if save_gif_path is not None:
            if not gfile.exists('./gif_temp_dirs'): gfile.makedirs('./gif_temp_dirs')
            plt.savefig('./gif_temp_dirs/'+str(batch)+'.png', bbox_inches='tight')
            if save_gif:
                frames = []
                image_path_list = sorted(gfile.glob('./gif_temp_dirs/*.png'), key=lambda i:int(i[16:-4]))
                for image_path in image_path_list:
                    frames.append(imageio.imread(image_path))
                imageio.mimsave(save_gif_path, frames, 'GIF', duration = 1)
                gfile.rmtree('./gif_temp_dirs')
        plt.show()

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
    def __init__(self, metrics_name, columns=2, iter_num=None, wait_num=1, figsize=None,
                 cell_size=(6, 4), valid_fmt="val_{}", eval_batch_num=None, save_image=False,
                 save_image_path=None, save_gif=False, save_gif_path=None):
#         tf.logging.set_verbosity(tf.logging.ERROR)
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
        self.save_image = save_image
        self.save_image_path = save_image_path
        self.save_gif = save_gif
        self.save_gif_path = save_gif_path

    def on_batch_end(self, batch, logs=None):
        self.batch_num += 1
        for i in ['batch', 'size']:
            logs.pop(i)
        for old_name, new_name in zip(self.model.metrics_names, self.metrics_name):
            logs[new_name] = logs.pop(old_name)
        if len(self.validation_data)>0:
            if self.eval_batch_num is not None:
                self.new_val_name = [self.valid_fmt.split('_')[0]+'_'+i for i in self.metrics_name]
                self.metrics = self.new_val_name+self.metrics_name
                if self.batch_num%self.eval_batch_num==0:
                    loss_list = self.model.test_on_batch(self.validation_data[0],
                                                         [self.validation_data[i] for i in range(1, int(len(self.validation_data)/2))])
                    for loss_value, new_name in zip(loss_list, self.new_val_name):
                        logs[new_name] = loss_value
        if self.figsize is None:
            self.figsize = (self.columns*self.cell_size[0], ((len(self.metrics)+1)//self.columns+1)*self.cell_size[1])
        for metric in logs:
            self.batch_logs[metric] += [logs[metric]]
        draw(metrics=self.metrics, logs=self.batch_logs, batch=self.batch_num, columns=self.columns,
             iter_num=self.iter_num, wait_num=self.wait_num, eval_batch_num=self.eval_batch_num,
             figsize=self.figsize, cell_size=self.cell_size, valid_fmt=self.valid_fmt,
             save_image=False, save_image_path=None, save_gif=False, save_gif_path=self.save_gif_path)

    def on_train_end(self, logs=None):
        draw(metrics=self.metrics, logs=self.batch_logs, batch=self.wait_num, columns=self.columns,
             iter_num=self.iter_num, wait_num=self.wait_num, eval_batch_num=self.eval_batch_num,
             figsize=self.figsize, cell_size=self.cell_size, valid_fmt=self.valid_fmt,
             save_image=self.save_image, save_image_path=self.save_image_path,
             save_gif=self.save_gif, save_gif_path=self.save_gif_path)
