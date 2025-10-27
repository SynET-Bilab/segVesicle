#!/bin/bash
#SBATCH --partition=tao
#SBATCH --nodes=1
#SBATCH --nodelist=t000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=veseg
#SBATCH --output=slurm-%j.out

#source /usr/share/Modules/init/bash
module purge
module load imod/4.12.16
module load isonet/20220823


pixelsize=17.14  #change every time
suffix="-bin4-wbp.rec"   #wbp file 后缀
areafile="area.mod"
snrfalloff=0.7   #0.3 for vpp, 0.7 for novpp

run_correct(){

    resample.py ../$3 $2 ./tomoset/$1_wbp_resample.mrc

    isonet.py prepare_star ./tomoset --pixel_size 17.14 
    isonet.py deconv tomograms.star --snrfalloff ${snrfalloff}  --deconv_folder tomo_deconv    
    isonet.py predict tomograms.star --output_dir ./ /share/data/CryoET_Data/zhenhang/software/segVesicle/pretrained/vesicle_corrected_model.h5 --gpuID 0,1,2,3
    mv $1_wbp_resample_corrected.mrc $1_wbp_corrected.mrc
}

run_segment(){

    tomofile="$1_wbp_corrected.mrc"
    deconvfile="$1_wbp_resample.mrc"
    maskfile="$1_segment.mrc"
    segment.py --tomo_file ${tomofile} --tomo_deconv_file ./tomo_deconv/${deconvfile} --mask_file ${maskfile} 

    if [ -f "../$2" ]; then
        morph.py --tomo ${tomo} --tomo_file ${tomofile} --mask_file ${maskfile} --pixelsize $3 --area_file ../$2
    else
        morph.py --tomo ${tomo} --tomo_file ${tomofile} --mask_file ${maskfile} --pixelsize $3
    fi
}



for str in `cat segVesicle.batch`
do
    IFS='/'
    read -ra arr <<< "$str"
    tomoidx=${arr}
    tomo=${tomoidx%-*}
    echo "-----------------do $tomoidx-----------------"
    
    mkdir -p ./$tomoidx/ves_seg/tomoset
    cd $tomoidx/ves_seg
    tomo_wbp=$tomo$suffix

    ###isonet correct
    #run_correct $tomo $pixelsize $tomo_wbp
    ###segment and fit
    run_segment $tomo $areafile $pixelsize

    cd ../../
    echo "-----------------$tomoidx done!-----------------"
done




