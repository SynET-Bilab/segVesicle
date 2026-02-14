#!/bin/bash
#SBATCH --partition=tao
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-12:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=veseg
#SBATCH --output=slurm-%j.out

export MODULEPATH=/share/root/users/luzh/modulefiles:$MODULEPATH
module purg
module load imod/4.12.16
module load segVesicle/20250309

# Define the current path
current_path=$(pwd)

# Add the path to vesicle_distance_and_filter script to PATH
# export PATH=/share/data/CryoET_Data/software/segVesicle/bin/:$PATH

# Define distance_nm, can be passed via SBATCH parameter or set here
distance_nm=${distance_nm:-50}  # Default to 50 if not passed
# Enable 2D fitting when fit_2d is set to true
fit_2d=${fit_2d:-true}

# Define the path for segVesicle_heart_broken.json
heart_broken_json="$current_path/segVesicle_heart_broken.json"

# Check if heart_broken_json exists, if not, skip the is_broken check
heart_broken_json_exists=false
if [ -f "$heart_broken_json" ]; then
    heart_broken_json_exists=true
fi

# Iterate over all directories in the current directory
for tomo_dir in "$current_path"/*/; do
    # Get the tomo_name
    tomo_name=$(basename "$tomo_dir")
    
    # Calculate base_tomo_name
    if [[ "$tomo_name" == *-1* ]]; then
        base_tomo_name="${tomo_name%%-1*}"
    else
        base_tomo_name="$tomo_name"
    fi

    # Define key paths
    json_path="$current_path/$tomo_name/ves_seg/${base_tomo_name}_vesicle.json"
    mod_path="$current_path/$tomo_name/ves_seg/membrane/${base_tomo_name}.mod"
    xml_output_path="$current_path/$tomo_name/ves_seg/vesicle_analysis/${base_tomo_name}_ori.xml"
    filter_xml_path="$current_path/$tomo_name/ves_seg/vesicle_analysis/${base_tomo_name}_filter.xml"
    isonet_tomo_path="$current_path/$tomo_name/ves_seg/${base_tomo_name}_wbp_corrected.mrc"

    # Check if heart_broken_json exists and if the corresponding tomo_name value is true
    if [ "$heart_broken_json_exists" = true ]; then
        is_broken=$(jq -r --arg TOMO "$tomo_name" '.[$TOMO]' "$heart_broken_json")
        if [ "$is_broken" == "true" ]; then
            echo "Skipping $tomo_name because segVesicle_heart_broken.json is marked as true."
            continue
        fi
    else
        echo "Skipping is_broken check because $heart_broken_json does not exist."
    fi

    # Check if json_path, mod_path, and isonet_tomo_path exist
    if [ ! -f "$json_path" ]; then
        echo "Skipping $tomo_name because file is missing: $json_path"
        continue
    fi

    if [ ! -f "$mod_path" ]; then
        echo "Skipping $tomo_name because file is missing: $mod_path"
        continue
    fi

    if [ ! -f "$isonet_tomo_path" ]; then
        echo "Skipping $tomo_name because file is missing: $isonet_tomo_path"
        continue
    fi

    # Call the Python script for processing
    fit_2d_flag=""
    if [ "$fit_2d" = true ]; then
        fit_2d_flag="--fit_2d"
    fi

    python /share/data/CryoET_Data/software/segVesicle/bin/vesicle_distance_and_filter.py --json_path "$json_path" --mod_path "$mod_path" --xml_output_path "$xml_output_path" --filter_xml_path "$filter_xml_path" --distance_nm "$distance_nm" --isonet_tomo_path "$isonet_tomo_path" $fit_2d_flag

    # Check if the Python script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error: Problem encountered while processing $tomo_name."
    else
        echo "Successfully processed $tomo_name."
    fi
done
