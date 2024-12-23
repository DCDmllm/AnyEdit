import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import numpy as np
import argparse
import os
import json
import AnyEdit_Collection.adaptive_editing_pipelines.tools.tool as tool
from PIL import Image
from tqdm import tqdm
def determine_rotation(direction, axis):
    """
    Determine whether the rotation is to the left or to the right.
    
    Args:
    direction (float): The rotation angle in degrees (not used in this determination).
    axis (list of float): The rotation axis as [x, y, z].
    
    Returns:
    str: 'left' if the rotation is to the left, 'right' if the rotation is to the right.
    """
    # Extract the y-component of the axis
    y_component = axis[1]
    
    # Determine the direction based on the y-component
    if y_component > 0:
        return 'left'  # 正面看逆时针
    else:
        return 'right'  # 正面看顺时针

# Example usage:
# axis = [0.05430099, -0.79789922, -0.6003401]  # Example axis from your earlier input
# direction = 83.77152889853502  # Example angle in degrees
# rotation_direction = determine_rotation(direction, axis)
# print(f"The rotation is to the {rotation_direction}.")

def read_data_from_txt(file_path):
    # 创建一个空列表来存储提取的数据
    extracted_data = []

    # 打开文件并读取每一行
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行尾的换行符并通过空格分割每个元素
            elements = line.strip().split()

            # 检查是否是包含数据的行（通过元素数量判断，数据行应多于10个元素）
            if len(elements) == 10 and elements[0] != '#':
                # 提取四元数 qx, qy, qz, qw（调整顺序为 qx, qy, qz, qw）
                quaternion = [float(elements[2]), float(elements[3]), float(elements[4]), float(elements[1])]

                # 提取图片名称
                image_name = elements[-1]

                # 将四元数和图片名称作为一个元组添加到列表中
                extracted_data.append((quaternion, image_name))

    # 返回提取的数据
    return extracted_data

def parse_args():
    parser = argparse.ArgumentParser(description="Rotation Change Pipeline")
    parser.add_argument("--text_file", required=True, help="path to MVImgNet txt file")
    parser.add_argument("--image_root", required=True, help="path to MVImgNet image file")
    parser.add_argument("--category", required=True, help="current category object for MVImgNet")
    parser.add_argument(
        "--options",
        nargs="+",
        help="in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--root_dir",
        help="The root dir you store your camera model and images, should be root_dir/single_image_folder/sparse .",
    )
    parser.add_argument(
        "--output_dir",
        help="output directory, output_dir/input, output_dir/output ...",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    root_dir = args.root_dir
    text_file_path_suffix = args.text_file
    obj = args.category
    output_root_dir = args.output_dir
    
    
    tool.prepare_output_dir(output_root_dir)
    
    total_edited_images = []
    
    # for every sub directory in root_dir
    # don't put any file 
    for subdir_name in tqdm(os.listdir(root_dir)):
        text_file_path = os.path.join(os.path.join(root_dir,subdir_name), text_file_path_suffix)
        if os.path.exists(text_file_path):
            data = read_data_from_txt(text_file_path)
        else:
            continue
        has_clockwise = False
        has_counterclockwise = False
        for i in range(2):
            q1 = random.choice(data)
            q2 = random.choice(data)
            while q1[1] == q2[1]:
                q1 = random.choice(data)
                q2 = random.choice(data)
            
            q_diff = R.from_quat(q1[0]) * R.from_quat(q2[0]).inv()
            q1_name = q1[1]
            q2_name = q2[1]
            q1_file = os.path.join(args.image_root, q1_name)
            q2_file = os.path.join(args.image_root, q2_name)
            
            angle = q_diff.magnitude() * 2  # 转换为角度
            axis = q_diff.as_rotvec() / np.linalg.norm(q_diff.as_rotvec())

            rotation_direction = determine_rotation(angle, axis)

            instruction_word = random.choice(['Turn', 'Make', 'Rotate'])

            if rotation_direction == 'right':
                if has_clockwise:
                    # 如果已经有一个顺时针旋转的指令，就将文件名反过来并变成逆时针旋转的指令
                    q1_file, q2_file = q2_file, q1_file
                    instruction = f"{instruction_word} the {obj} counterclockwise"
                else:
                    instruction = f"{instruction_word} the {obj} clockwise"
                    has_clockwise = True
            else:
                if has_counterclockwise:
                    # 如果已经有一个逆时针旋转的指令，就将文件名反过来并变成顺时针旋转的指令
                    q1_file, q2_file = q2_file, q1_file
                    instruction = f"{instruction_word} the {obj} clockwise"
                else:
                    instruction = f"{instruction_word} the {obj} counterclockwise"
                    has_counterclockwise = True
            
            
            # store q1 and q2 to target dir
            q1_image = Image.open(os.path.join(os.path.join(root_dir,subdir_name),q1_file))
            q2_image = Image.open(os.path.join(os.path.join(root_dir,subdir_name),q2_file))
            q1_image.save(os.path.join(output_root_dir,"init",subdir_name + "_" +q1_name))
            q2_image.save(os.path.join(output_root_dir,"edited_img",subdir_name + "_"+q2_name))
            
            cur_data = {"edit": instruction, "edit object": None, "input": None, "output": None, "edit_type": "rotation_change", "visual_input": None, "image_file": subdir_name + "_"+q1_name, "edited_file": subdir_name + "_"+q2_name}
            
            total_edited_images.append(cur_data)
            
        root_dir = args.root_dir
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        output_dir = root_dir

    jsonl_file_path = os.path.join(output_root_dir, 'final_edit_results_all.jsonl')
    with open(jsonl_file_path, 'w') as jsonl_file:
        for data in total_edited_images:
            jsonl_file.write(json.dumps(data) + '\n')

    print("JSONL 文件已成功导出到 {}".format(jsonl_file_path))