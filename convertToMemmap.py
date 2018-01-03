import os
import argparse
import numpy as np
import time
import sys
import multiprocessing
from sima import ImagingDataset
import itertools as it
import warnings

from sima.misc.progressbar import ProgressBar
#from sima.sequence import _NaNMeanSequence as dims_reducer
from sima.sequence import _MIPSequence as dims_reducer
from sima.sequence import _IndexedSequence
from sima.sequence import _MotionCorrectedSequence


def _first_obs(seq):
    print 'searching for first observed values for each pixel'
    p = ProgressBar(seq.shape[0])
    frame_iter1 = iter(seq)
    first_obs = next(frame_iter1)
    for i, frame in enumerate(frame_iter1):
        p.update(i)
        for frame_chan, fobs_chan in zip(frame, first_obs):
            fobs_chan[np.isnan(fobs_chan)] = frame_chan[np.isnan(fobs_chan)]
        if all(np.all(np.isfinite(chan)) for chan in first_obs):
            break
    p.end()

    return first_obs


def _fill_gaps(first_obs, frame_iter2):
    """Fill missing rows in the corrected images with data from nearby times.

    Parameters
    ----------
    frame_iter1 : iterator of list of array
        The corrected frames (one list entry per channel).
    frame_iter2 : iterator of list of array
        The corrected frames (one list entry per channel).

    Yields
    ------
    list of array
        The corrected and filled frames.

    """
    most_recent = [x * np.nan for x in first_obs]
    for i, frame in enumerate(frame_iter2):
        for fr_chan, mr_chan in zip(frame, most_recent):
            mr_chan[np.isfinite(fr_chan)] = fr_chan[np.isfinite(fr_chan)]
        yield np.array([np.nan_to_num(mr_ch) + np.isnan(mr_ch) * fo_ch
               for mr_ch, fo_ch in zip(most_recent, first_obs)])


def _writer(in_file, out_file, x1, x2, first_obs, channel, trim, z_plane,
            skip_init):
    chunk_size = x2-x1
    ds = ImagingDataset.load(in_file)
    # slicing seems to be not working properly for for some
    # _MotionCorrectedSequences
    #if type(ds.sequences[0]) == _MotionCorrectedSequence:
    #    seq = _IndexedSequence(
    #        ds.sequences[0], (slice(skip_init, None, None), slice(None),
    #        slice(None), slice(None), ds._resolve_channel(channel)))

    seq = ds.sequences[0][skip_init:, :, :, :, ds._resolve_channel(channel)]
    num_f = seq.shape[0]
    seq = seq[x1:x2]

    if trim != 0:
        seq = seq[:, :, trim:-trim, trim:-trim]

    if z_plane is not None:
        seq = seq[:, z_plane]
    elif seq.shape[1] != 1:
        seq = dims_reducer(seq)

    iterator = _fill_gaps(first_obs, iter(seq))
    shape = (np.prod(seq.shape[2:]), chunk_size)

    f = np.memmap(out_file, mode='r+', dtype=np.float32, order='C',
                  shape=(shape[0], num_f))

    p = ProgressBar(chunk_size)
    for f_idx,frame in enumerate(iterator):
        p.update(f_idx)
        f[:,x1+f_idx] = np.nan_to_num(np.reshape(frame, shape[0], order='F'))
    p.end()
    del f
    return out_file


def pool_helper(args):
    return _writer(*args)


def convert(in_file, out_file, chunk_size=500, processes=4, channel='Ch2',
            first_obs=150, trim=0, z_plane=None, skip_init=0):

    ds = ImagingDataset.load(in_file)
    if type(ds.sequences[0]) == _MotionCorrectedSequence:
        ds.sequences[0] = _IndexedSequence(
            ds.sequences[0], tuple([slice(None)]*5))
        ds.save()
    try:
        seq = ds.sequences[0][skip_init:, :, :, :, ds._resolve_channel(channel)]
    except:
        import pdb; pdb.set_trace()

    blocks = np.linspace(
        0, seq.shape[0], np.ceil(float(seq.shape[0])/chunk_size)+1,
        dtype=int)
    if trim != 0:
        seq = seq[:, :, trim:-trim, trim:-trim]

    if z_plane is not None:
        seq = seq[:, z_plane]

    if seq.shape[1] != 1:
        seq = dims_reducer(seq)

    first_obs = _first_obs(seq[:first_obs])
    shape = (np.prod(seq.shape[2:]), seq.shape[0])
    del seq
    del ds

    f = np.memmap(out_file, mode='w+', dtype=np.float32, order='C', shape=shape)

    start = time.time()
    num_blocks = len(blocks)-1
    print 'exporting in %i blocks' % (num_blocks)
    pool = multiprocessing.Pool(processes=processes)
    pool.map(pool_helper, zip([in_file]*num_blocks, [out_file]*num_blocks,
        blocks[:-1], blocks[1:], [first_obs]*num_blocks, [channel]*num_blocks,
        [trim]*num_blocks, [z_plane]*num_blocks, [skip_init]*num_blocks))
    print 'finished in %i seconds' % (time.time() - start)

    pool.close()
    pool.join()


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-s", "--scratch_drive", action="store", type=str,
        default="/scratch/caiman", help="destination drive for memmap file")
    argParser.add_argument(
        "-c", "--channel", action="store", type=str,
        default="Ch2", help="channel name. default is Ch2")
    argParser.add_argument(
        '-k', '--skip_init', action='store', type=int, default=0,
        help="skip initial frames when creating the memmap file"
    )
    argParser.add_argument(
        "filename", action="store", type=str, default="",
        help=("Process any experiment that has a tSeriesDirectory containing" +
              "'directory'"))
    args = argParser.parse_args()

    ds = ImagingDataset.load(args.filename)
    trim = 0

    out_file = 'Yr_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap'.format(
        ds.frame_shape[1]-2*trim, ds.frame_shape[2]-2*trim,
        ds.num_frames-args.skip_init)

    directory = os.path.join(
        args.scratch_drive, os.path.split(args.filename)[0][1:])
    if not os.path.exists(directory):
        os.makedirs(directory)

    del ds # HDF5 file cannot be open when starting pools

    out_file = os.path.join(directory, out_file)
    print "writing to %s" % out_file
    convert(args.filename, os.path.join(directory, out_file), chunk_size=1000,
            channel=args.channel, trim=trim, skip_init=args.skip_init,
            first_obs=150)


if __name__ == '__main__':
    main()
