import os
from pydub import AudioSegment
import sys

# 定义路径
source_dir = '/data/liangjh/LLaMA-Factory/data/FuseLLM/voice/'
output_dir = '/data/liangjh/LLaMA-Factory/data/FuseLLM/voice_split/'
file_extension = '.amr'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def split_audio(file_path, out_dir):
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_filename_a = os.path.join(out_dir, f"{base_filename}_A{file_extension}")
    output_filename_b = os.path.join(out_dir, f"{base_filename}_B{file_extension}")

    try:
        # 明确指定amr格式给from_file
        audio = AudioSegment.from_file(file_path, format="amr")
    except Exception as e:
        print(f"错误：无法加载或处理文件 {file_path}。可能是ffmpeg不支持AMR或文件损坏。错误：{e}")
        return

    length_ms = len(audio)
    if length_ms == 0:
        print(f"警告：文件 {file_path} 时长为0，无法切分。")
        return
        
    midpoint_ms = length_ms // 2

    part_a = audio[:midpoint_ms]
    part_b = audio[midpoint_ms:]

    try:
        # 尝试显式指定AMR-NB编码器，并设置采样率为8000Hz
        part_a.export(output_filename_a, format="amr", codec="libopencore_amrnb", parameters=["-ar", "8000"])
        part_b.export(output_filename_b, format="amr", codec="libopencore_amrnb", parameters=["-ar", "8000"])
        print(f"已切分 {file_path} -> {output_filename_a}, {output_filename_b}")
    except Exception as e:
        # 如果 libopencore_amrnb 失败，尝试备选的 amr_nb (某些ffmpeg版本可能用这个别名)
        try:
            print(f"使用 libopencore_amrnb 编码失败，尝试使用 'amr_nb' 作为编码器 (同样设置8kHz采样率)... ({e})")
            part_a.export(output_filename_a, format="amr", codec="amr_nb", parameters=["-ar", "8000"])
            part_b.export(output_filename_b, format="amr", codec="amr_nb", parameters=["-ar", "8000"])
            print(f"已使用 'amr_nb' 编码器切分 {file_path} -> {output_filename_a}, {output_filename_b}")
        except Exception as e2:
            print(f"错误：无法导出切分后的文件 {os.path.basename(output_filename_a)} / {os.path.basename(output_filename_b)}。尝试了 libopencore_amrnb 和 amr_nb (均设置8kHz) 均失败。错误：{e2}")
            print("这通常意味着您的ffmpeg版本没有编译AMR编码支持，或者指定的采样率转换存在问题。")

print(f"开始处理目录 {source_dir} 中的 {file_extension} 文件...")
print(f"切分后的文件将保存到 {output_dir}")

found_files = 0
processed_files = 0
for filename in os.listdir(source_dir):
    if filename.endswith(file_extension) and not filename.startswith('.'): # 忽略隐藏文件
        found_files += 1
        full_file_path = os.path.join(source_dir, filename)
        split_audio(full_file_path, output_dir)
        processed_files +=1


if found_files == 0:
    print(f"在 {source_dir} 中没有找到任何 {file_extension} 文件。")
else:
    print(f"处理完成。共找到并尝试处理 {processed_files}/{found_files} 个文件。")

print("\n重要提示:")
print("此脚本依赖 pydub 和 ffmpeg (需支持AMR编解码, 如 libopencore-amrnb/libopencore-amrwb)。")
print("请确保已安装这些依赖。例如:")
print("  pip install pydub")
print("  # (对于 ffmpeg, 请根据您的操作系统进行安装, 例如 Ubuntu: sudo apt install ffmpeg)") 