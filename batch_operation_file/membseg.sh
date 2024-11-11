#!/usr/bin/env bash

echo "----segsbatch----"
echo "Generate membseg.sbatch"

fsbatch="membseg.sbatch"

if [[ -f ${fsbatch} ]]
then
    mv ${fsbatch} ${fsbatch}~
fi

cat > ${fsbatch} <<EOL
#!/bin/bash
#SBATCH --partition=tao
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --job-name=segmem
#SBATCH --output=slurm-%j.out

source /usr/share/Modules/init/bash

module purge
module load etsynseg 

# 遍历当前文件夹中的所有子文件夹
for sub_folder in */
do
    # 移除路径末尾的斜杠，获取子文件夹的名称
    sub_folder_name="\${sub_folder%/}"
    
    # 检查是否存在prompt.mod文件
    prompt_file="./\${sub_folder_name}/ves_seg/membrane/prompt.mod"
    if [[ -f "\$prompt_file" ]]
    then
        echo "Found prompt.mod in \${sub_folder_name}, running segprepost.py..."

        # 处理命令的文件路径，确保没有多余的空格
        input_file="\${sub_folder_name}_wbp_corrected.mrc"
        output_path="./\${sub_folder_name}/ves_seg/membrane/\${sub_folder_name}"

        # 执行segprepost.py命令，包含输出路径
        segprepost.py run "./\${sub_folder_name}/ves_seg/""\$input_file" "\$prompt_file" -o "\$output_path"
    else
        echo "No prompt.mod found in \${sub_folder_name}, skipping."
    fi
done
EOL

echo "template.sbatch generated."

# 提交生成的sbatch作业
sbatch ${fsbatch}

