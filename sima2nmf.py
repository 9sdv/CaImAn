from builtins import str
from builtins import range

import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import numpy as np
import psutil
import glob
import os
import json
import hashlib
import scipy
import argparse
from ipyparallel import Client
import pylab as pl
pl.ion()

from sima import ImagingDataset
from sima.ROI import ROI
from sima.ROI import ROIList
from sima import imaging_parameters

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar
from caiman.base.rois import extract_binary_masks_blob

from plot_results import signals_plotter


def find_datasets(search_directory, overwrite=False):
    for directory, folders, files in os.walk(search_directory):
        if directory.endswith('.sima'):
            if overwrite or not os.path.exists(
                    os.path.join(directory, 'cnmf_results.npz')):
                yield directory


def exportRois(sima_path, contours, z_plane=None):
    ds = ImagingDataset.load(sima_path)
    ds_hash = str(int(hashlib.sha1(ds.savedir).hexdigest(), 16) % (10 ** 8))
    im_shape = ds.frame_shape[:3]
    if z_plane is None:
        z_plane = range(ds.frame_shape[0])
    else:
        z_plane = [z_plane]
    roi_list = []
    for roi in contours:
        coords = roi['coordinates']
        segment_boundaries = np.where(np.isnan(coords[:, 0]))[0]
        polys = []
        for i in xrange(1, len(segment_boundaries)):
            for j in z_plane:
                segment = coords[
                    (segment_boundaries[i-1]+1):segment_boundaries[i], :]
                segment = np.hstack((segment, j*np.ones((segment.shape[0], 1))))
                polys.append(segment)
        roi_label = ds_hash + '_' + str(roi['neuron_id'])
        roi_list.append(ROI(
            polygons=polys, label=roi_label,
            id=roi_label, im_shape=im_shape))

    ds.add_ROIs(ROIList(roi_list), label='cnmf_rois')

    return roi_list


def main(argv):
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        '-d', '--dendrites', action='store_true',
        help='is dendrites, possibly inconsistant results on somas')
    argParser.add_argument(
        '-a', '--alph_snmf', action='store', type=float, default=1e10,
        help='controls the sparsity')
    argParser.add_argument(
        '-P', '--patch_size', action='store', type=int,
        default=200, help='set patch size to run the algorithm on patches, 0 \
        indicates not to use patches, default=200')
    argParser.add_argument(
        "-s", "--stride", action="store", type=int, default=20,
        help="amount of overlap between the patches")
    argParser.add_argument(
        '-p', '--system_order', action='store', type=int, default=2,
        help='order of the autoregressive system, default=2')
    argParser.add_argument(
        "-c", "--channel", action="store", type=str,
        default="Ch2", help="channel name. default is Ch2")
    argParser.add_argument(
        '-n', '--neuron_count', action='store', type=int, default=20,
        help='expected neuron count per patch default=20')
    argParser.add_argument(
        '-m', '--deconvolution_method', action='store', type=str,
        default='cvxpy', help='deconvolution method used. cvxpy or oaisis, \
        default=cvxpy')
    argParser.add_argument(
        '-g', '--gSig', action='store', type=int, default=4.5,
        help='expected half size of the neurons')
    argParser.add_argument(
       '-z', '--z_plane', action='store', type=int, default=None,
       help='z-plane of the sima file to extract, -1 will \
       compute the max intensity accross all planes. default=None')
    argParser.add_argument(
        "-t", "--scratch_drive", action="store", type=str,
        default="/scratch/caiman", help="destination drive for memmap file")
    argParser.add_argument(
        '-k', '--keep_mmap', action='store_true',
        help='set to prevent remavl of memmory mapped file used during cnmf \
        calculation')
    argParser.add_argument(
        '-o', '--overwrite', action='store_true',
        help='if running on on multiple files, force overwrite of \
        cnmf_results.npz')
    argParser.add_argument(
        '-D', '--defaults', action='store', type=str,
        help="load defualt arguments from configs.json, argument is the label \
        of the settings to load")
    argParser.add_argument(
        "filename", action="store", type=str, default="",
        help=("Process any experiment that has a tSeriesDirectory containing" +
              "'directory'"))
    if '-D' in argv:
        settings_dict = json.load(
            open('configs.json'))['sima2nmf'][argv[argv.index('-D') + 1]]
        argParser.set_defaults(**settings_dict)

    args = argParser.parse_args()

    if os.path.splitext(args.filename)[1] not in ['.sima', '.mmap']:
        datasets = find_datasets(args.filename, args.overwrite)
    else:
        datasets = [args.filename]
    if args.patch_size:
        is_patches = True
    else:
        is_patches = False

    is_dendrites = args.dendrites

    if is_dendrites is True:
        # THIS METHOd CAN GIVE POSSIBLY INCOSISTENT RESULTS ON SOMAS WHEN NOT
        # USED WITH PATCHES
        init_method = 'sparse_nmf'
        alpha_snmf = args.alpha_snmf  # this controls sparsity
    else:
        init_method = 'greedy_roi'
        alpha_snmf = None  # 10e2  # this controls sparsity

    gSig = [args.gSig, args.gSig]
    for dataset in datasets:
        ext = os.path.splitext(dataset)[1]
        if ext == '.sima':
            import convertToMemmap
            fname = dataset
            print 'converting %s' % fname

            ds = ImagingDataset.load(fname)
            trim = 0

            if ds.frame_shape[0] != 1:
                warnings.warn("3D datasets not supported, calculating MIP of \
                              z-axis")
            original_n_planes = ds.frame_shape[0]

            out_file = 'Yr_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap'.format(
                ds.frame_shape[1]-2*trim, ds.frame_shape[2]-2*trim,
                ds.num_frames)

            directory = os.path.join(
                args.scratch_drive, os.path.split(dataset)[0][1:])
            if not os.path.exists(directory):
                os.makedirs(directory)

            del ds # HDF5 file cannot be open when starting pools

            fname_new = os.path.join(directory, out_file)
            print "writing to %s" % out_file
            convertToMemmap.convert(
                dataset, fname_new, chunk_size=1000, channel=args.channel,
                trim=0, z_plane=args.z_plane)

        elif ext == '.mmap':
            fname_new = dataset
            sima_path = os.path.split(fname_new.split(args.scratch_drive)[1])[0]
            sima_files = glob.glob(os.path.join(sima_path, '*.sima'))
            if not len(sima_files):
                raise Exception('sima file not found')
            elif len(sima_files) > 1:
                raise Exception('multiple sima files in TSeries Directory')
            fname = sima_files[0]
            ds = ImagingDataset.load(fname)
            original_n_planes = ds.frame_shape[0]
            del ds
            print fname
        else:
            raise Exception('filetype not supported')

        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=8, single_thread=False)

        Yr, dims, T = cm.load_memmap(fname_new)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        Y = np.reshape(Yr, dims + (T,), order='F')

        if np.min(images)<0:
            raise Exception('Movie too negative, add_to_movie should be larger')
        if np.sum(np.isnan(images))>0:
            raise Exception(
                'Movie contains nan! You did not remove enough borders')

        Cn = cm.local_correlations(Y[:,:,:3000])

        K = args.neuron_count
        p = args.system_order
        prairie_xml = glob.glob(os.path.join(
            os.path.abspath(os.path.join(fname, os.pardir)),'*.xml'))[0]

        if not is_patches:
            merge_thresh = 0.8  # merging threshold, max correlation allowed
            cnm = cnmf.CNMF(n_processes, method_init=init_method, k=K,
                gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview,
                Ain=None, method_deconvolution=args.deconvolution_method,
                skip_refinement = False)
            cnm = cnm.fit(images)
            crd = plot_contours(cnm.A, Cn, thr=0.9)
        else:
            # half-size of the patches in pixels. rf=25, patches are 50x50
            rf = int(args.patch_size/2)
            stride = args.stride  # amount of overlap between the patches in
                                  # pixels
            merge_thresh = 0.8  # merging threshold, max correlation allowed
            save_results = False
            # RUN ALGORITHM ON PATCHES

            cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0,
                dview=dview, Ain=None, rf=rf, stride=stride, memory_fact=1,
                method_init=init_method, alpha_snmf=alpha_snmf,
                only_init_patch=True, gnb=1,
                method_deconvolution=args.deconvolution_method)
            cnm = cnm.fit(images)

            A_tot = cnm.A
            C_tot = cnm.C
            YrA_tot = cnm.YrA
            b_tot = cnm.b
            f_tot = cnm.f
            sn_tot = cnm.sn

            print ('Number of components:' + str(A_tot.shape[-1]))
            pl.figure()
            crd = plot_contours(A_tot, Cn, thr=0.9)

            # approx final rate  (after eventual downsampling)
            final_frate = \
                1/imaging_parameters.prairie_imaging_parameters(
                    prairie_xml)['framePeriod']/original_n_planes
            Npeaks = 10
            traces = C_tot + YrA_tot
            fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, \
                significant_samples = evaluate_components(Y, traces, A_tot,
                    C_tot, b_tot, f_tot, final_frate, remove_baseline=True, N=5,
                    robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

            idx_components_r = np.where(r_values >= .5)[0]
            idx_components_raw = np.where(fitness_raw < -40)[0]
            idx_components_delta = np.where(fitness_delta < -20)[0]

            idx_components = np.union1d(idx_components_r, idx_components_raw)
            idx_components = np.union1d(idx_components, idx_components_delta)
            idx_components_bad = np.setdiff1d(list(range(len(traces))),
                                              idx_components)

            print ('Keeping ' + str(len(idx_components)) +
                   ' and discarding  ' + str(len(idx_components_bad)))

            pl.figure()
            crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)

            A_tot = A_tot.tocsc()[:, idx_components]
            C_tot = C_tot[idx_components]

            save_results = False
            if save_results:
                np.savez('results_analysis_patch.npz', A_tot=A_tot, C_tot=C_tot,
                    YrA_tot=YrA_tot, sn_tot=sn_tot, d1=d1, d2=d2, b_tot=b_tot,
                    f=f_tot)

            cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig,
                merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot,
                Cin=C_tot, f_in=f_tot, rf=None, stride=None,
                method_deconvolution=args.deconvolution_method)
            cnm = cnm.fit(images)

        A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
        S = cnm.S

        final_frate = \
            1/imaging_parameters.prairie_imaging_parameters(
                prairie_xml)['framePeriod']/original_n_planes

        Npeaks = 10
        traces = C + YrA
        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, \
            significant_samples = evaluate_components(Y, traces, A, C, b, f,
                final_frate, remove_baseline=True, N=5, robust_std=False,
                Athresh=0.1, Npeaks=Npeaks, thresh_C=0.3)

        idx_components_r = np.where(r_values >= .95)[0]
        idx_components_raw = np.where(fitness_raw < -100)[0]
        idx_components_delta = np.where(fitness_delta < -100)[0]

        idx_components = np.union1d(idx_components_r, idx_components_raw)
        idx_components = np.union1d(idx_components, idx_components_delta)
        idx_components_bad = np.setdiff1d(
            list(range(len(traces))), idx_components)

        print ' ***** '
        print (len(traces))
        print (len(idx_components))

        np.savez(os.path.join(fname, 'cnmf_results.npz'),
                 Cn=Cn, A=A.todense(), C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1,
                 d2=d2, S=S, idx_components=idx_components,
                 idx_components_bad=idx_components_bad, params=vars(args))

        crd = plot_contours(A.tocsc(), Cn, thr=0.9)
        print 'finished plotting'

        rois = exportRois(fname, crd, z_plane=args.z_plane)
        print 'roi sima export complete'

        ph = PlotHelper(fname, 'cnmf.pdf')
        if len(idx_components) > 0:
            print 'plotting components good'
            ph.addROIs([rois[i] for i in idx_components], 'GREEN')
        if len(idx_components_bad) > 0:
            print 'plotting components bad'
            ph.addROIs([rois[i] for i in idx_components_bad], 'RED')
        ph.saveFigure()

        signals_plotter(fname, C, idx_components_bad)

        # STOP CLUSTER and clean up log files
        cm.stop_server()

        log_files = glob.glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        if not args.keep_mmap:
            os.remove(fname_new)
            try:
                os.removedirs(os.path.split(fname_new)[0])
            except OSError:
                pass

if __name__ == '__main__':
    main(sys.argv[1:])
