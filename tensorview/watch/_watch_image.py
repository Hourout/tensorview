import os
import PIL
from pyecharts.charts import Page
from pyecharts.components import Image, Table
from pyecharts import options as opts
from linora.sample import ImageDataset
from linora.image import read_image

__all__ = ['image_fold', 'image_info']

def image_base(img_src, title, subtitle, width, height):
    image = (Image()
             .add(src=img_src, style_opts={"width": width, "height": height, "style": "margin-top: 20px"})
             .set_global_opts(title_opts=opts.ComponentTitleOpts(title=title, subtitle=subtitle))
            )
    return image

def image_fold(root, show_name=False, show_size=False, show_shape=False, path='image_fold.html', max_num=None):
    """Show fold image visualize.
    
    Args:
        root: image dataset file root.
        show_name: whether show image name.
        show_size: whether show image size.
        show_shape: whether show image shape.
        path: result save a html file.
        max_num: max show image nums
    Returns:
        A pyecharts polt object.
    """
    image_series = ImageDataset(root).data.image
    image_list = []
    for r, i in enumerate(image_series):
        if max_num is not None:
            if r>max_num:
                break
        shape = read_image(i).shape
        title = i.split('/')[-1] if show_name else ''
        width = str(int(200/shape[0]*shape[1]))+'px'
        subtitle = 'size:'+str(round(os.stat(i).st_size/1024/1024, 2))+'M' if show_size else ''
        subtitle = subtitle+' shape:'+str(shape) if show_shape else subtitle+''
        image_list.append(image_base(i, title, subtitle, width, '200px'))
    page = Page(layout=Page.SimplePageLayout).add(*image_list)
    return page.render(path)

def image_info(root, path='image_info.html', show_original=False):
    """Show image visualize.
    
    Args:
        root: An image file root.
        path: result save a html file.
        show_original: whether show original image shape.
    Returns:
        A pyecharts polt object.
    """
    shape = read_image(root).shape
    info = PIL.Image.open(root).info
    width = '' if show_original else str(360)+'px'
    height = '' if show_original else str(int(360/shape[1]*shape[0]))+'px'
    image_charts = image_base(root, '', '', width, height)
    headers = ["Attribute", "Info"]
    rows = [['path', root],
            ['size', str(round(os.stat(root).st_size/1024/1024, 2))+'M'],
            ['shape', str(read_image(root).shape)],
            ['jfif', info['jfif']],
            ['jfif_version', info['jfif_version']],
            ['dpi', info['dpi']],
            ['jfif_unit', info['jfif_unit']],
            ['jfif_density', info['jfif_density']]
    ]
    table_charts = Table().add(headers, rows)
    page = Page(layout=Page.SimplePageLayout).add(*[image_charts, table_charts])
    return page.render(path)
           
           
