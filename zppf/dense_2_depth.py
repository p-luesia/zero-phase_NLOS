import numpy as np
from argparse import ArgumentParser
import sys
import os


def parse_args(argv):
    parser = ArgumentParser('dense_2_coords',
            description = 'Transform the dense reconstruction into coords')
    parser.add_argument('filenames', nargs='+', type = str, 
                        help = 'Dense reconstruction in npy.')
    parser.add_argument('--delta_z', nargs = '+', type = float, required = True,
                        help = 'Spacing between planes. Can be 1 or same'\
                            +' number of input filenames')
    parser.add_argument('--offset', nargs = '+', type = float, required = True,
                        help = 'Distance to the first plane. Can be 1 or same'\
                            +' number of input filenames')
    parser.add_argument('--output', nargs = '+', type = str, required = True,
                        help = 'Output path. Can be 1 or same number of '\
                              +'input filenames')
    
    return parser.parse_args(argv)
    

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    delta_z = args.delta_z
    offset = args.offset
    output = args.output

    assert len(delta_z) == len(args.filenames) or len(delta_z) == 1,\
        f"Wrong number of delta_z parameter. Expected {len(args.filenames)} or 1, received {len(delta_z)}."
    assert len(offset) == len(args.filenames) or len(offset) == 1,\
        f"Wrong number of offset parameter. Expected {len(args.filenames)} or 1, received {len(offset)}."
    assert len(output) == len(args.filenames) or len(output) == 1,\
        f"Wrong number of output parameter. Expected {len(args.filenames)} or 1, received {len(output)}."

    if len(delta_z) == 1:
        delta_z = delta_z * len(args.filenames)
    if len(offset) == 1:
        offset = offset * len(args.filenames)
    if len(output) == 1:
        output = output * len(args.filenames)

    for file, d_z, z_off, opath in zip(args.filenames, delta_z, offset, output):
        data = np.load(file)
        max_coords = np.argmax(data, axis = 0)
        coords = max_coords*d_z + z_off
        ofile = opath
        if os.path.isdir(opath):
            ofile = os.path.join(opath, ".".join(os.path.split(file)[-1].split('.')[:-1]))

        print(ofile)
        np.save(ofile, coords)
        