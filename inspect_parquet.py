import pandas as pd
import argparse
from PIL import Image
import io
import numpy as np
import os

def inspect_and_extract_first_entry(file_path, output_dir='output_data'):
    """
    Reads a Parquet file, extracts the first data entry, and saves its components.

    Args:
        file_path (str): The path to the .parquet file.
        output_dir (str): The directory where the extracted data will be saved.
    """
    # --- 1. Validate Input and Setup Output Directory ---
    if not os.path.exists(file_path):
        print(f"错误：文件未找到于 '{file_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"数据将被保存到: '{output_dir}'")

    # --- 2. Read Parquet File ---
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"读取 Parquet 文件时出错: {e}")
        return

    if df.empty:
        print("错误: Parquet 文件为空。")
        return

    # --- 3. Extract the First Row ---
    first_row = df.iloc[0]
    print("\n成功读取第一条数据。开始提取并保存...")

    # --- 4. Process and Save Each Field ---

    # Process 'image_1'
    if 'image_1' in first_row and first_row['image_1'] is not None:
        try:
            image_bytes = first_row['image_1']
            image = Image.open(io.BytesIO(image_bytes))
            save_path = os.path.join(output_dir, 'image_1.png')
            image.save(save_path)
            print(f"  - 'image_1' 已成功保存为: {save_path}")
        except Exception as e:
            print(f"  - 处理 'image_1' 时出错: {e}")
    else:
        print("  - 'image_1' 未在数据中找到或为空。")

    # Process 'image_2' (optional)
    if 'image_2' in first_row and first_row['image_2'] is not None:
        try:
            image_bytes = first_row['image_2']
            image = Image.open(io.BytesIO(image_bytes))
            save_path = os.path.join(output_dir, 'image_2.png')
            image.save(save_path)
            print(f"  - 'image_2' 已成功保存为: {save_path}")
        except Exception as e:
            print(f"  - 处理 'image_2' 时出错: {e}")
    else:
        print("  - 'image_2' 未在数据中找到或为空。")
        
    # Process 'state'
    if 'state' in first_row and first_row['state'] is not None:
        try:
            state_vector = np.array(first_row['state'])
            save_path = os.path.join(output_dir, 'state.txt')
            np.savetxt(save_path, state_vector, fmt='%.6f', header='State Vector:')
            print(f"  - 'state' 已成功保存为: {save_path}")
            print(f"    State 内容: {state_vector}")
        except Exception as e:
            print(f"  - 处理 'state' 时出错: {e}")
    else:
        print("  - 'state' 未在数据中找到或为空。")

    # Process 'action'
    if 'action' in first_row and first_row['action'] is not None:
        try:
            action_vector = np.array(first_row['action'])
            save_path = os.path.join(output_dir, 'action.txt')
            np.savetxt(save_path, action_vector, fmt='%.6f', header='Action Vector:')
            print(f"  - 'action' 已成功保存为: {save_path}")
            print(f"    Action 内容: {action_vector}")
        except Exception as e:
            print(f"  - 处理 'action' 时出错: {e}")
    else:
        print("  - 'action' 未在数据中找到或为空。")

    # Process 'prompt'
    if 'prompt' in first_row and first_row['prompt'] is not None:
        try:
            prompt_text = first_row['prompt']
            save_path = os.path.join(output_dir, 'prompt.txt')
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(prompt_text)
            print(f"  - 'prompt' 已成功保存为: {save_path}")
            print(f"    Prompt 内容: '{prompt_text}'")
        except Exception as e:
            print(f"  - 处理 'prompt' 时出错: {e}")
    else:
        print("  - 'prompt' 未在数据中找到或为空。")
        
    print("\n提取完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="从 Parquet 文件中提取第一条数据的 image, state, action, 和 prompt 信息。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'file_path', 
        type=str, 
        help="需要读取的 .parquet 文件的路径。"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='output_data', 
        help="保存提取出数据的目录名称。\n(默认: 'output_data')"
    )

    args = parser.parse_args()
    inspect_and_extract_first_entry(args.file_path, args.output_dir) 