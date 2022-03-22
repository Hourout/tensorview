![](https://github.com/Hourout/tensorview/blob/master/image/tensorview.png)

Dynamic visualization training service in Jupyter Notebook for Keras, tf.keras and others.


![PyPI version](https://img.shields.io/pypi/pyversions/tensorview.svg)
![Github license](https://img.shields.io/github/license/Hourout/tensorview.svg)
[![PyPI](https://img.shields.io/pypi/v/tensorview.svg)](https://pypi.python.org/pypi/tensorview)
![PyPI format](https://img.shields.io/pypi/format/tensorview.svg)
![downloads](https://img.shields.io/pypi/dm/tensorview.svg)

Tensorview is efficient visualization training packages of Keras, enabling the live visualization of loss function and metrics during training process.

Tensorview is process in the Jupyter Notebook, so we recommend Jupyter Notebook as Tensorview's developing environment.

![](https://github.com/Hourout/beefly/blob/master/image/plot_metrics000.gif)

| [API Document](https://github.com/Hourout/tensorview/blob/master/document/api.md) | [中文介绍](https://github.com/Hourout/tensorview/blob/master/document/Chinese.md) |

## Installation

To install [this verson from PyPI](https://pypi.org/project/beefly/), type:

```
pip install tensorview
```

To get the newest one from this repo (note that we are in the alpha stage, so there may be frequent updates), type:

```
pip install git+https://github.com/Hourout/tensorview.git
```

## Feature
- Visualization of general machine learning training
  - Support customizing the name of loss function and metric 
  - Support customizing the name of validation set
  - Support displaying the result of training set and test set in the same plot
  - Support setting maximum training number of iterations
  - Support both batch way and epoch way to train
  - Support visual plotting
  - Support visual plotting in both batch way and epoch way periodically
  
- keras backend
  - Support customizing the name of loss function and metric in epoch way
  - Support customizing the name of validation set in epoch way
  - Support displaying the result of training set and test set in the same plot in epoch way
  - Support setting maximum training number of iterations in epoch way
  - Support visual plotting in epoch way
  - Support visual plotting in both batch way and epoch way periodically in epoch way
  - Support customizing the name of loss function and metric in batch way
  - Support customizing the name of validation set in batch way
  - Support displaying the result of training set and test set in the same plot in batch way
  - Support setting maximum training number of iterations in batch way
  - Support visual plotting in batch way
  - Support visual plotting in both batch way and epoch way periodically in batch way

- layer visualize
  - feature layer visualize
  - weights visualize
  - heatmaps visualize
 
- model statistics

- image visualize

## Example

[more example](https://github.com/Hourout/tensorview/blob/master/example/readme.md)

using tensorview

```python
import tensorview as tv
```

- [tv.train.PlotMetrics()](https://github.com/Hourout/tensorview/blob/master/example/tv.train.plot_metrics.ipynb)
- [tv.train.PlotMetricsOnBatch()](https://github.com/Hourout/tensorview/blob/master/example/PlotMetricsOnBatch.ipynb)
- [tv.train.PlotMetricsOnEpoch()](https://github.com/Hourout/tensorview/blob/master/example/PlotMetricsOnEpoch.ipynb)
- [tv.model.[..]](https://github.com/Hourout/tensorview/blob/master/example/model.ipynb)

## Contact
Please contact me if you have any related questions or improvements.

[WeChat](https://github.com/Hourout/tensorview/blob/master/image/hourout_wechat.jpg)
