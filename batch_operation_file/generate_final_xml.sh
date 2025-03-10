#!/bin/bash
#SBATCH --partition=tao
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-12:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=veseg
#SBATCH --output=slurm-%j.out

# Load required modules
#source /usr/share/Modules/init/bash
export MODULEPATH=/share/root/users/luzh/modulefiles:$MODULEPATH
module purg
module load imod/4.12.16
module load segVesicle/20250309

# Input: Original Pixel Size
ORIGINAL_PIXEL_SIZE=$1

# Get the current directory where the script is executed
CURRENT_PATH=$(pwd)

# Loop over each folder in the current directory
for folder in */ ; do
    # Extract the folder name without the trailing slash
    tomo_name=$(basename "$folder")

    # Extract base_tomo_name
    if [[ "$tomo_name" == *"-1"* ]]; then
        base_tomo_name=${tomo_name%%-1*}
    else
        base_tomo_name=$tomo_name
    fi

    # Define the class XML path
    class_xml_path="${CURRENT_PATH}/${tomo_name}/ves_seg/vesicle_analysis/${base_tomo_name}_vesicle_class.xml"

    # Check if the class XML file exists
    if [[ -f "$class_xml_path" ]]; then
        # Call the Python script to process the XML
        echo "Processing $class_xml_path with original pixel size $ORIGINAL_PIXEL_SIZE"
        python /share/data/CryoET_Data/software/segVesicle/bin/util/generate_xml.py "$class_xml_path" "$ORIGINAL_PIXEL_SIZE"
    else
        echo "Warning: $class_xml_path does not exist, skipping."
    fi
done

echo "All processing complete."e
