from pyecharts.charts import Page
from pyecharts.components import Image
from pyecharts import options as opts
from linora.sample import ImageDataset
from linora.image import read_image

__all__ = ['

def image_base(img_src, title, subtitle):
    image = (Image()
             .add(src=img_src, style_opts={"width": "200px", "height": "200px", "style": "margin-top: 20px"})
             .set_global_opts(title_opts=opts.ComponentTitleOpts(title=title, subtitle=subtitle))
            )
    return image

def image_fold(root, show_name=True, show_size=False, show_shape=True, path='image_visualize.html'):
    image_series = ImageDataset(root).data.image
    image_list = []
    for i in image_series:
        title = i.split('\\')[-1] if show_name else ''
        subtitle = 'size:'+str(round(os.stat(i).st_size/1024/1024, 2))+'M' if show_size else ''
        subtitle = subtitle+' shape:'+str(read_image(i).shape) if show_shape else subtitle+''
        image_list.append(image_base(i, title, subtitle))
    page = Page(layout=Page.SimplePageLayout).add(*image_list)
    return page.render(path)
