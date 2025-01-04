import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        # 初始化自定义数据集类
        self.questions = questions  # 问题列表
        self.image_folder = image_folder  # 图像文件夹路径
        self.tokenizer = tokenizer  # 分词器
        self.image_processor = image_processor  # 图像处理器
        self.model_config = model_config  # 模型配置

    def __getitem__(self, index):
        # 获取指定索引处的数据项
        line = self.questions[index]  # 获取问题行
        image_file = line["image"]  # 获取图像文件名
        qs = ""  # 获取问题文本
        if self.model_config.mm_use_im_start_end:
            # 如果模型配置使用图像起始和结束标记
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            # 否则只使用图像标记
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # 获取对话模板并添加消息
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()  # 获取提示文本

        # 打开图像并转换为RGB格式
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        # 处理图像并转换为张量
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # 将提示文本转换为输入ID
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        # 返回输入ID、图像张量和图像尺寸
        return input_ids, image_tensor, image.size

    def __len__(self):
        # 返回数据集的长度
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def eval_model(args):
    # 禁用torch初始化
    disable_torch_init()
    # 展开用户路径
    model_path = os.path.expanduser(args.model_path)
    # 从路径中获取模型名称
    model_name = get_model_name_from_path(model_path)
    # 加载预训练模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # 读取问题文件
    declarative_questions = [json.loads(q) for q in open(os.path.expanduser(args.declarative_question_file), "r")]
    # 获取指定的块
    declarative_questions = get_chunk(declarative_questions, args.num_chunks, args.chunk_idx)

    # 创建数据加载器
    data_loader = create_data_loader(declarative_questions, args.image_folder, tokenizer, image_processor, model.config)

    # 展开答案文件路径
    representations_file = os.path.expanduser(args.representations_file)
    # 创建答案文件目录
    os.makedirs(os.path.dirname(representations_file), exist_ok=True)

    # 创建一个空的 .pt 文件，准备存储流式写入的数据
    with open(representations_file, 'wb') as f:
        # 遍历数据加载器和问题
        for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, declarative_questions), total=len(declarative_questions)):
            # 获取问题ID
            idx = line["question_id"]

            # 推理模式
            with torch.inference_mode():
                # 获取 multimodal encoder 的 image 的 representation
                images = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

            image_features, pro_image_features = model.encode_images_steps(images)

            # 构建每组数据的字典
            data = {
                'question_id': idx,
                'image_features': image_features,
                'pro_image_features': pro_image_features,
                'model_id': model_name
            }

            # 将每组数据逐条保存
            torch.save(data, f)
            f.flush()  # 强制刷新缓存，以便及时写入磁盘

    # 读取保存的 .pt 文件并验证
    loaded_data_list = []
    with open(representations_file, 'rb') as f:
        while True:
            try:
                # 逐个读取每个数据字典
                data = torch.load(f)
                loaded_data_list.append(data)
            except EOFError:
                break

    # 打印读取的数据
    for idx, data in enumerate(loaded_data_list):
        print(f"Data {idx}:")
        print(data.keys())

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*copying from a non-meta parameter.*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--declarative-question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--representations-file", type=str, default="representations.pt")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
