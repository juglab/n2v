from typing import Union, List, Dict
from pathlib import Path
from enum import Enum

import tensorflow as tf
from zipfile import ZipFile
from csbdeep.utils import save_json

from ..models.n2v_config import N2VConfig


class Extensions(Enum):
    BIOIMAGE_EXT = ".bioimage.io.zip"
    KERAS_EXT = ".h5"
    TF_EXT = ".zip"


class Format(Enum):
    H5 = "h5"
    TF = "tf"


class Algorithm(Enum):
    N2V = 0
    StructN2V = 1
    N2V2 = 2

    @staticmethod
    def get_name(algorithm: int) -> str:
        if algorithm == 1:
            return "structN2V"
        elif algorithm == 2:
            return "N2V2"
        else:
            return "Noise2Void"


class PixelManipulator(Enum):
    UNIFORM_WITH_CP = "uniform_withCP"
    UNIFORM_WITHOUT_CP = "uniform_withoutCP"
    NORMAL_WITHOUT_CP = "normal_withoutCP"
    NORMAL_ADDITIVE = "normal_additive"
    NORMAL_FITTED = "normal_fitted"
    IDENTITY = "identity"
    MEAN = "mean"
    MEDIAN = "median"


def which_algorithm(config: N2VConfig):
    """
    Checks which algorithm the model is configured for (N2V, N2V2, structN2V).
    """
    if config.structN2Vmask is not None:
        return Algorithm.StructN2V
    elif (
        config.n2v_manipulator == PixelManipulator.MEDIAN.value
        and not config.unet_residual
        and config.blurpool
        and config.skip_skipone
    ):
        return Algorithm.N2V2
    else:
        return Algorithm.N2V


def generate_bioimage_md(name: str, cite: list, path: Path):
    """
    Generate a generic document.md file for the bioimage.io format.
    """
    # create doc
    file = path / "napari-n2v.md"
    with open(file, 'w') as f:
        text = cite[0]['text']

        content = f'## {name}\n' \
                  f'This network was trained using [napari-n2v](https://pypi.org/project/napari-n2v/).\n\n' \
                  f'## Cite {name}\n' \
                  f'{text}'
        f.write(content)

    return file.absolute()


def get_algorithm_details(algorithm: Algorithm):
    """
    Returns name, authors and citation related to the algorithm, formatted as expected by bioimage.io
    model builder.
    """
    if algorithm == Algorithm.StructN2V:
        citation = [{'text': 'C. Broaddus, A. Krull, M. Weigert, U. Schmidt and G. Myers, \"Removing Structured Noise '
                             'with Self-Supervised Blind-Spot Networks,\" 2020 IEEE 17th International Symposium on '
                             'Biomedical Imaging (ISBI), 2020, pp. 159-163',
                     'doi': '10.1109/ISBI45749.2020.9098336'}]
    elif algorithm == Algorithm.N2V2:
        citation = [{'text': 'E. Hoeck, T.-O. Buchholz, A. Brachmann, F. Jug and A. Freytag, '
                             '\"N2V2--Fixing Noise2Void Checkerboard Artifacts with Modified Sampling Strategies and a '
                             'Tweaked Network Architecture.\" arXiv preprint arXiv:2211.08512 (2022).',
                     'doi': '10.48550/arXiv.2211.08512'}]
    else:
        citation = [{'text': 'A. Krull, T.-O. Buchholz and F. Jug, \"Noise2Void - Learning Denoising From Single '
                             'Noisy Images,\" 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition  '
                             '(CVPR), 2019, pp. 2124-2132',
                     'doi': '10.48550/arXiv.1811.10980'}]

    return citation


def build_modelzoo(
    result_path: Union[str, Path],
    weights_path: Union[str, Path],
    bundle_path: Union[str, Path],
    inputs: str,
    outputs: str,
    preprocessing: list,
    postprocessing: list,
    doc: Union[str, Path],
    name: str,
    authors: list,
    algorithm: Algorithm,
    tf_version: str,
    cite: List[Dict],
    axes: str = "byxc",
    files: list = [],
    **kwargs,
):
    from bioimageio.core.build_spec import build_model

    tags_dim = "3d" if len(axes) == 5 else "2d"

    build_model(
        root=bundle_path,
        weight_uri=weights_path,
        test_inputs=[inputs],
        test_outputs=[outputs],
        input_axes=[axes],
        output_axes=[axes],
        output_path=result_path,
        name=name,
        description="Self-supervised denoising.",
        authors=authors,
        license="BSD-3-Clause",
        documentation=doc,
        tags=[
            tags_dim,
            "unet",
            "denoising",
            Algorithm.get_name(algorithm.value),
            "tensorflow",
            "napari",
        ],
        preprocessing=[preprocessing],
        postprocessing=[postprocessing],
        tensorflow_version=tf_version,
        attachments={"files": files},
        cite=cite,
        **kwargs,
    )


def save_model_tf(model, config, model_path, config_path):
    # save bundle without including optimizer
    # (otherwise the absence of the custom functions cause errors upon loading)
    model_folder_path = model_path.parent / model_path.stem
    tf.keras.models.save_model(
        model,
        model_folder_path,
        save_format=Format.TF.value,
        include_optimizer=False,
    )

    # save configuration
    save_json(vars(config), config_path)

    # zip it and save to destination
    final_archive = model_path.absolute()
    with ZipFile(final_archive, mode="w") as archive:
        for file_path in model_folder_path.rglob("*"):
            archive.write(file_path, arcname=file_path.relative_to(model_folder_path))

    return final_archive
