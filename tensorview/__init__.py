from tensorview import train
from tensorview import model
from tensorview import watch

__version__ = '0.4.0'
__author__ = 'JinQing Lee'


def _hello():
    print("""
------------------------------------------------------------------------------------
      Linora
--------------------
      Version      : --  {}  --
      Author       : --  {}  --
      License      : Apache-2.0
      Homepage     : https://github.com/Hourout/tensorview
      Description  : Dynamic visualization training service in Jupyter Notebook for Keras, tf.keras and others.
------------------------------------------------------------------------------------""".format(__version__, __author__))
