import numpy as np
from pyecharts.charts import Page, Scatter, Tab
from pyecharts.components import Image
from pyecharts import options as opts
import tensorflow as tf
from linora.image import ImageAug
from cv2 import applyColorMap, COLORMAP_JET, addWeighted


__all__ = ['visualize_weights', 'visualize_layer', 'visualize_heatmaps']

def scatter_base(value, label, title, subtitle):
    c = (Scatter()
         .add_xaxis(list(range(len(value.numpy().reshape(-1)))))
         .add_yaxis(label, value.numpy().reshape(-1).tolist(), symbol_size=2,)
         .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
         .set_global_opts(title_opts=opts.TitleOpts(title=title, subtitle=subtitle)))
    return c

def visualize_weights(model, layer_name, path='visualize_weights.html'):
    charts = []
    for name in layer_name:
        for weights in model.weights:
            if name in weights.name:
                label = weights.name.split('/')[-1]
                title = name
                subtitle = 'shape='+str(model.weights[0].shape)
                charts.append(scatter_base(weights, label, title, subtitle))
    page = Page().add(*charts)
    return page.render(path)

def image_base(img_src, title, subtitle):
    image = (Image()
             .add(src=img_src)
             .set_global_opts(title_opts=opts.ComponentTitleOpts(title=title, subtitle=subtitle)))
    return image

def visualize_layer(model, image, layer_name, layer_max_image=32, jupyter=True, path='visualize_layer.html'):
    """network layer visualize.
    
    Args:
        model: a tf.keras model or keras model.
        image: a image array with shape (1, height, width, channel).
        layer_name: a list of model layers name.
        layer_max_image: every layer max plot images.
        jupyter: if plot in jupyter, default True.
        path: if jupyter is False, result save a html file.
    Returns:
        A pyecharts polt object.
    """
    if tf.io.gfile.exists('feature_map'):
        tf.io.gfile.rmtree('feature_map')
    tf.io.gfile.makedirs('feature_map')
    temp_model = tf.keras.backend.function(model.inputs, [i.output for i in model.layers if i.name in layer_name])
    temp_name = [i.name for i in model.layers if i.name in layer_name]
    result = temp_model(image)
    images_per_row = 16
    count = 0
    name_dict = {}
    tab = Tab()
    for feature, name in zip(result, temp_name):
        if feature.ndim==4:
            if feature.shape[-1]==3:
                display_grid = feature[0,:,:,:].astype('uint8')
            else:
                n_features = feature.shape[-1] if feature.shape[-1]<layer_max_image else layer_max_image
                size = feature.shape[1]
                n_cols = int(np.ceil(n_features/images_per_row))
                display_grid = np.ones((size * n_cols, images_per_row * size),dtype=np.uint8)*255
                for col in range(n_cols):
                    for row in range(images_per_row):
                        if (col+1)*(row+1)>n_features:
                            break
                        channel_image = feature[0, :, :, col * images_per_row + row]
                        channel_image -= channel_image.mean()
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
                display_grid = np.expand_dims(display_grid, axis=-1)
            name_dict[name] = f'./feature_map/{count}.png'
            ImageAug(display_grid).save_image(name_dict[name])
            tab.add(image_base(name_dict[name], name, 'shape='+str(feature.shape)), name)
        count += 1
    return tab.render_notebook() if jupyter else tab.render(path)

def visualize_heatmaps(model, image, layer_name, jupyter=True, path='visualize_heatmaps.html'):
    """network layer visualize.
    
    Args:
        model: a tf.keras model or keras model.
        image: a image array with shape (1, height, width, channel).
        layer_name: a list of model layers name.
        jupyter: if plot in jupyter, default True.
        path: if jupyter is False, result save a html file.
    Returns:
        A pyecharts polt object.
    """
    if tf.io.gfile.exists('feature_map'):
        tf.io.gfile.rmtree('feature_map')
    tf.io.gfile.makedirs('feature_map')
    temp_model = tf.keras.backend.function(model.inputs, [i.output for i in model.layers if i.name in layer_name])
    temp_name = [i.name for i in model.layers if i.name in layer_name]
    result = temp_model(image)
    images_per_row = 16
    count = 0
    name_dict = {}
    tab = Tab()
    for feature, name in zip(result, temp_name):
        if feature.ndim==4:
            if model.get_layer(name).__class__.__name__=='InputLayer':
                out = np.squeeze(image, 0).astype('uint8')
            else:
                out = tf.image.resize(tf.expand_dims(tf.squeeze(tf.reduce_sum(tf.abs(feature), axis=-1)), axis=-1), (image.shape[1], image.shape[2]))
                out = 255-tf.cast(out/tf.reduce_max(out)*255., tf.uint8)
    #             print(np.squeeze(image, 0).shape)
                out = addWeighted(applyColorMap(out.numpy(), COLORMAP_JET), 0.7, np.squeeze(image, 0).astype('uint8'), 0.3, 0)
    #             out = np.expand_dims(out, axis=-1)
            name_dict[name] = f'./feature_map/{count}.png'
            ImageAug(out).save_image(name_dict[name])
            tab.add(image_base(name_dict[name], name, 'shape='+str(feature.shape)), name)
        count += 1
    return tab.render_notebook() if jupyter else tab.render(path)
