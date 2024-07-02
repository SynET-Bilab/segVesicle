#!/usr/bin/env python3
import logging
import sys
import os
import fire
import mrcfile
import numpy as np
from fire import core



class Ves_seg:
    '''
    Vesicle segmentation, use pretrained model based on U-net
    for tomograms, including preprocessing, segmentation and postprocessing,
    
    '''
    def resample(self,tomo,
                   pixel_size,
                   out_name=None,
                   outspacing=17.142):
        '''

        '''
        from resampling import resample_image, measure
        # from resampling import main, measure

        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
                            datefmt="%m-%d %H:%M:%S", level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
        
        logging.info("\n######Start resampling process######\n")

        [original_spacing, original_size, out_spacing, out_size] = measure(tomo, pixel_size, outspacing)
        logging.info("resample_tomo: {}| original_spacing: {}| original_size: {}| out_spacing: {}| out_size: {}".format(
            tomo, original_spacing, original_size, out_spacing, out_size
        ))
        resample_image(tomo, pixel_size, out_name, outspacing)
        logging.info("\n######Done resampling process######\n")


    def predict(self,mrc,
                model,
                dir='.',
                sidelen=128,
                neighbor_in=7,
                neighbor_out=3,
                batch_size=8,
                gpuID:str='0'):
        '''

        '''
        import mrcfile
        from sgmt_predict import predict_new
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
                            datefmt="%m-%d %H:%M:%S", level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

        logging.info("\n######Start prediction process######\n")
        root_name = mrc.split('/')[-1].split('.')[0]
        if not os.path.isdir(dir):
            os.mkdir(dir)

        model = model.split(',')
        if len(model) == 1:
            output = dir+'/'+root_name+'-mask.mrc'
            predict_new(mrc, output, model[0], sidelen, neighbor_in, neighbor_out, batch_size, gpuID)
        elif len(model) > 1:
            output_list = []
            #alias_name = []
            data = []
            for i in range(len(model)):
                logging.info('start predict with model {}'.format(i))
                output = dir+'/'+root_name+'-mask'+str(i)+'.mrc'
                predict_new(mrc, output, model[i], sidelen, neighbor_in, neighbor_out, batch_size, gpuID)
                output_list.append(output)
                # name = 'mask' + str(i)
                # alias_name.append(name)
            for j in range(len(output_list)):
                with mrcfile.open(output_list[j]) as mask:
                    data.append(mask.data)
            with mrcfile.open(mrc) as orig_m:
                shape = orig_m.data.shape
            mask = np.zeros(shape, dtype=np.uint16)
            for k in data:
                mask = mask | k
            final_name = dir + '/' + root_name + '-mask.mrc'
            with mrcfile.new(final_name, overwrite=True) as f:
                f.set_data(mask)

        logging.info("\n######Done prediction process######\n")


    def morph(self,mask_file,
              area_file,
              dir='.',
              radius=10):
        '''

        '''
        from morph import morph_process, vesicle_measure, vesicle_rendering
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
                            datefmt="%m-%d %H:%M:%S", level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
        root_name = mask_file.split('/')[-1].split('-')[0]

        logging.info("\n######Start morphological process######\n")
        with mrcfile.open(mask_file) as m:
            bimask = m.data
        shape = bimask.shape
        vesicle_list, vesicle_list_sup, shape_morph_process = morph_process(mask_file, radius)
        logging.info("\n######Done morphological process######\n")

        logging.info("\n######Start vesicle measuring######\n")
        output_file = dir + '/' + root_name + '_vesicle.json'
        output_file_in_area = dir + '/' + root_name + '_vesicle_in.json'
        [vesicle_info, in_vesicle_info] = vesicle_measure(vesicle_list, vesicle_list_sup, shape, radius,
                                                          output_file, output_file_in_area, area_file)
        logging.info("\n######Done vesicle measuring######\n")

        logging.info("\n######Start vesicle rendering######\n")
        render = dir + '/' + root_name + '_vesicle.mrc'
        render_in = dir + '/' + root_name + '_vesicle_in.mrc'
        ves_tomo = vesicle_rendering(output_file, shape)
        with mrcfile.new(render, overwrite=True) as m:
            m.set_data(ves_tomo)
        print('Rendering vesicles in roi')
        ves_tomo_in = vesicle_rendering(output_file_in_area, shape)
        with mrcfile.new(render_in, overwrite=True) as m_in:
            m_in.set_data(ves_tomo_in)
        logging.info("\n######Done vesicle rendering######\n")


def check_parse(args_list):
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    if args_list[0] in ['resample', 'predict', 'morph', 'gui']:
        if args_list[0] in ['resample','predict','morph']:
            check_list = eval(args_list[0]+'_param') + ['help']
        else:
            check_list = None
    else:
        check_list = None
    if check_list is not None:
        for arg in args_list:
            if type(arg) is str and arg[0:2]=='--':
                if arg[2:] not in check_list:
                    logger.error(" '{}' not recognized!".format(arg[2:]))
                    sys.exit(0)


def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)


if __name__ == '__main__':
    #core.Display = Display
    #if len(sys.argv) > 1:
    #    check_parse(sys.argv[1:])
    fire.Fire(Ves_seg)













