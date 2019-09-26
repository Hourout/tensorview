import PIL
from pyecharts.charts import Page
from pyecharts.components import Image, Table
from pyecharts import options as opts
from linora.sample import ImageDataset
from linora.image import read_image

__all__ = ['image_fold', 'image_info']

def image_base(img_src, title, subtitle):
    image = (Image()
             .add(src=img_src, style_opts={"width": "200px", "height": "200px", "style": "margin-top: 20px"})
             .set_global_opts(title_opts=opts.ComponentTitleOpts(title=title, subtitle=subtitle))
            )
    return image

def image_fold(root, show_name=True, show_size=False, show_shape=True, path='image_visualize.html'):
    """Show fold image visualize.
    
    Args:
        root: image dataset file root.
        show_name: whether show image name.
        show_size: whether show image size.
        show_shape: whether show image shape.
        path: result save a html file.
    Returns:
        A pyecharts polt object.
    """
    image_series = ImageDataset(root).data.image
    image_list = []
    for i in image_series:
        title = i.split('/')[-1] if show_name else ''
        subtitle = 'size:'+str(round(os.stat(i).st_size/1024/1024, 2))+'M' if show_size else ''
        subtitle = subtitle+' shape:'+str(read_image(i).shape) if show_shape else subtitle+''
        image_list.append(image_base(i, title, subtitle))
    page = Page(layout=Page.SimplePageLayout).add(*image_list)
    return page.render(path)

def image_info(root, path='image.html', show_original=False):
    """Show image visualize.
    
    Args:
        root: An image file root.
        show_original: whether show original image shape.
        path: result save a html file.
    Returns:
        A pyecharts polt object.
    """
    shape = read_image(root).shape
    info = PIL.Image.open(root).info
    width = '' if show_original else str(360)+'px'
    height = '' if show_original else str(int(360/shape[1]*shape[0]))+'px'
    image_charts = Image().add(src=root, style_opts={"width": width, "height": height, "style": "margin-top: 20px"})
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
    page = Page(layout=pe.charts.Page.SimplePageLayout).add(*[image_charts, table_charts])
    return page.render(path)
           
           
