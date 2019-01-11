from __future__ import print_function, unicode_literals, absolute_import, division

import argparse
import sys
from pprint import pprint

import numpy as np
from tqdm import tqdm

from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.utils import _raise, axes_check_and_normalize
from csbdeep.utils.six import Path


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--quiet',            metavar='', type=str2bool, required=False,  const=True, nargs='?', default=False,                                            help="don't show status messages")
    parser.add_argument('--gpu-memory-limit', metavar='', type=float,    required=False,                         default=None,                                             help="limit GPU memory to this fraction (0...1)")

    data = parser.add_argument_group("input")
    data.add_argument('--input-dir',          metavar='', type=str,      required=False,                         default=None,                                             help="path to folder with input images")
    data.add_argument('--input-pattern',      metavar='', type=str,      required=False,                         default='*.tif*',                                         help="glob-style file name pattern of input images")
    data.add_argument('--input-axes',         metavar='', type=str,      required=False,                         default=None,                                             help="axes string of input images")
    data.add_argument('--norm-pmin',          metavar='', type=float,    required=False,                         default=2,                                                help="'pmin' for PercentileNormalizer")
    data.add_argument('--norm-pmax',          metavar='', type=float,    required=False,                         default=99.8,                                             help="'pmax' for PercentileNormalizer")
    data.add_argument('--norm-undo',          metavar='', type=str2bool, required=False,  const=True, nargs='?', default=True,                                             help="'do_after' for PercentileNormalizer")
    data.add_argument('--n-tiles',            metavar='', type=int,      required=False,              nargs='+', default=None,                                             help="number of tiles for prediction")

    model = parser.add_argument_group("model")
    model.add_argument('--model-basedir',     metavar='', type=str,      required=False,                         default=None,                                             help="path to folder that contains CARE model")
    model.add_argument('--model-name',        metavar='', type=str,      required=False,                         default=None,                                             help="name of CARE model")
    model.add_argument('--model-weights',     metavar='', type=str,      required=False,                         default=None,                                             help="specific name of weights file to load (located in model folder)")

    output = parser.add_argument_group("output")
    output.add_argument('--output-dir',       metavar='', type=str,      required=False,                         default=None,                                             help="path to folder where restored images will be saved")
    output.add_argument('--output-name',      metavar='', type=str,      required=False,                         default='{model_name}/{file_path}/{file_name}{file_ext}', help="name pattern of restored image (special tokens: {file_path}, {file_name}, {file_ext}, {model_name}, {model_weights})")
    output.add_argument('--output-dtype',     metavar='', type=str,      required=False,                         default='float32',                                        help="data type of the saved tiff file")
    output.add_argument('--imagej-tiff',      metavar='', type=str2bool, required=False,  const=True, nargs='?', default=True,                                             help="save restored image as ImageJ-compatible TIFF file")
    output.add_argument('--dry-run',          metavar='', type=str2bool, required=False,  const=True, nargs='?', default=False,                                            help="don't save restored images")

    return parser, parser.parse_args()


def main():
    if not ('__file__' in locals() or '__file__' in globals()):
        print('running interactively, exiting.')
        sys.exit(0)

    # parse arguments
    parser, args = parse_args()
    args_dict = vars(args)

    # exit and show help if no arguments provided at all
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # check for required arguments manually (because of argparse issue)
    required = ('--input-dir','--input-axes', '--norm-pmin', '--norm-pmax', '--model-basedir', '--model-name', '--output-dir')
    for r in required:
        dest = r[2:].replace('-','_')
        if args_dict[dest] is None:
            parser.print_usage(file=sys.stderr)
            print("%s: error: the following arguments are required: %s" % (parser.prog,r), file=sys.stderr)
            sys.exit(1)

    # show effective arguments (including defaults)
    if not args.quiet:
        print('Arguments')
        print('---------')
        pprint(args_dict)
        print()
        sys.stdout.flush()

    # logging function
    log = (lambda *a,**k: None) if args.quiet else tqdm.write

    # get list of input files and exit if there are none
    file_list = list(Path(args.input_dir).glob(args.input_pattern))
    if len(file_list) == 0:
        log("No files to process in '%s' with pattern '%s'." % (args.input_dir,args.input_pattern))
        sys.exit(0)

    # delay imports after checking to all required arguments are provided
    from tifffile import imread, imsave
    import keras.backend as K
    from csbdeep.models import CARE
    from csbdeep.data import PercentileNormalizer
    sys.stdout.flush()
    sys.stderr.flush()

    # limit gpu memory
    if args.gpu_memory_limit is not None:
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(args.gpu_memory_limit)

    # create CARE model and load weights, create normalizer
    K.clear_session()
    model = CARE(config=None, name=args.model_name, basedir=args.model_basedir)
    if args.model_weights is not None:
        print("Loading network weights from '%s'." % args.model_weights)
        model.load_weights(args.model_weights)
    normalizer = PercentileNormalizer(pmin=args.norm_pmin, pmax=args.norm_pmax, do_after=args.norm_undo)

    n_tiles = args.n_tiles
    if n_tiles is not None and len(n_tiles)==1:
        n_tiles = n_tiles[0]

    processed = []

    # process all files
    for file_in in tqdm(file_list, disable=args.quiet or (n_tiles is not None and np.prod(n_tiles)>1)):
        # construct output file name
        file_out = Path(args.output_dir) / args.output_name.format (
            file_path = str(file_in.relative_to(args.input_dir).parent),
            file_name = file_in.stem, file_ext = file_in.suffix,
            model_name = args.model_name, model_weights = Path(args.model_weights).stem if args.model_weights is not None else None
        )

        # checks
        (file_in.suffix.lower()  in ('.tif','.tiff') and
         file_out.suffix.lower() in ('.tif','.tiff')) or _raise(ValueError('only tiff files supported.'))

        # load and predict restored image
        img = imread(str(file_in))
        restored = model.predict(img, axes=args.input_axes, normalizer=normalizer, n_tiles=n_tiles)

        # restored image could be multi-channel even if input image is not
        axes_out = axes_check_and_normalize(args.input_axes)
        if restored.ndim > img.ndim:
            assert restored.ndim == img.ndim + 1
            assert 'C' not in axes_out
            axes_out += 'C'

        # convert data type (if necessary)
        restored = restored.astype(np.dtype(args.output_dtype), copy=False)

        # save to disk
        if not args.dry_run:
            file_out.parent.mkdir(parents=True, exist_ok=True)
            if args.imagej_tiff:
                save_tiff_imagej_compatible(str(file_out), restored, axes_out)
            else:
                imsave(str(file_out), restored)

        processed.append((file_in,file_out))


    # print summary of processed files
    if not args.quiet:
        sys.stdout.flush()
        sys.stderr.flush()
        n_processed   = len(processed)
        len_processed = len(str(n_processed))
        log('Finished processing %d %s' % (n_processed, 'files' if n_processed > 1 else 'file'))
        log('-' * (26+len_processed if n_processed > 1 else 26))
        for i,(file_in,file_out) in enumerate(processed):
            len_file = max(len(str(file_in)),len(str(file_out)))
            log(('{:>%d}. in : {:>%d}'%(len_processed,len_file)).format(1+i,str(file_in)))
            log(('{:>%d}  out: {:>%d}'%(len_processed,len_file)).format('',str(file_out)))


if __name__ == '__main__':
    main()
