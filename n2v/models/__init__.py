from __future__ import absolute_import, print_function

# checks
try:
    import tensorflow
    del tensorflow
except ModuleNotFoundError as e:
    from six import raise_from
    raise_from(RuntimeError('Please install TensorFlow: https://www.tensorflow.org/install/'), e)

try:
    import keras
    del keras
except ModuleNotFoundError as e:
    if e.name in {'theano','cntk'}:
        from six import raise_from
        raise_from(RuntimeError(
            "Keras is configured to use the '%s' backend, which is not installed. "
            "Please change it to use 'tensorflow' instead: "
            "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % e.name
        ), e)
    else:
        raise e

import keras.backend as K
if K.backend() != 'tensorflow':
    raise NotImplementedError(
            "Keras is configured to use the '%s' backend, which is currently not supported. "
            "Please configure Keras to use 'tensorflow' instead: "
            "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.backend()
        )
if K.image_data_format() != 'channels_last':
    raise NotImplementedError(
        "Keras is configured to use the '%s' image data format, which is currently not supported. "
        "Please change it to use 'channels_last' instead: "
        "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.image_data_format()
    )
del K


# imports
from .n2v_config import N2VConfig
from .n2v_standard import N2V