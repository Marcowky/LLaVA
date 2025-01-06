import argparse
import torch
import os
import json
from tqdm import tqdm
from transformers import CLIPModel, AutoProcessor

from PIL import Image


def eval_model(args):

    # 读取问题文件
    declarative_questions = [json.loads(q) for q in open(os.path.expanduser(args.declarative_question_file), "r")]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型并移动到设备
    model = CLIPModel.from_pretrained(args.model_path).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 遍历数据加载器和问题
    for line in tqdm(declarative_questions, total=len(declarative_questions)):
        # 获取问题ID
        image = Image.open(os.path.join(args.image_folder, line["image"]))

        # 处理文本和图像数据并将图像数据转移到适当的设备
        inputs = processor(text=[line['text_yes'], line['text_no']], images=image, return_tensors="pt", padding=True)

        # 将输入数据（文本和图像）移到设备
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # 模型推理
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # 这是图像-文本的相似度分数
        probs = logits_per_image.softmax(dim=1)  # 使用 softmax 得到标签概率

        # 根据概率选择答案
        ans = 'Yes' if probs[0][0] >= probs[0][1] else 'No'

        # 将答案写入文件
        ans_file.write(json.dumps({
            "question_id": line["question_id"],
            "text": ans,
            "text_yes": line['text_yes'],
            "text_no": line['text_no'],
            "probs_yes": probs[0][0].item(),
            "probs_no": probs[0][1].item(),
            "model_id": args.model_path,
            "metadata": {}
        }) + "\n")

    ans_file.close()

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*copying from a non-meta parameter.*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--declarative-question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    eval_model(args)
