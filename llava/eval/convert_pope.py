import argparse
import json

def convert_to_declarative(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            question_text = data.get('text', '')
            
            # 删除多余的部分 "\\nAnswer the question using a single word or phrase."
            question_text = question_text.split("\\n")[0]
            
            # 提取"yes"和"no"的陈述句
            declarative_yes = question_text.replace('Is there', 'There is').replace('?', '.')
            declarative_no = declarative_yes.replace('There is a', 'There is no')
            
            # 生成新的 JSON 对象
            new_data = {
                'question_id': data.get('question_id'),
                'image': data.get('image'),
                'text_yes': declarative_yes,
                'text_no': declarative_no,
                'category': data.get('category', '')
            }
            
            # 将新的数据写入输出文件
            outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert question text to declarative sentences")
    parser.add_argument('--question-file', type=str, required=True, help="Input JSONL file with questions")
    parser.add_argument('--declarative-question-file', type=str, required=True, help="Output JSONL file with declarative sentences")
    
    args = parser.parse_args()
    
    convert_to_declarative(args.question_file, args.declarative_question_file)
