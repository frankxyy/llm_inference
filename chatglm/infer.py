import argparse
from statistics import mean
import time
from transformers import AutoTokenizer, AutoModel

def add_parser_arguments(parser):
    parser.add_argument("--req_cnt", type=int, default=1)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
        
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    from utils import load_model_on_gpus
    model = load_model_on_gpus("THUDM/chatglm-6b", num_gpus=2)

    model = model.eval()
    ques = "你好，请帮我计算一个数学题：两个苹果加上六个苹果，等于多少个苹果"
    ques_tokens = tokenizer(ques)
    print("ques_tokens = {}".format(ques_tokens))
    input_len = len(ques_tokens['input_ids'])
    print("len of ques {}".format(input_len))

    cnt = 0
    tot_time = 0
    tot_tokens = 0
    response_list = []
    for i in range(args.req_cnt):
        start_time = time.time()
        input_ids = tokenizer(ques, return_tensors="pt").input_ids.cuda()
        # response = model.chat(ques)
        outputs = model.generate(
            input_ids,
            num_beams=1,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            max_length = 200,
            remove_invalid_values=True,
        )
        print("Output:\n" + 100 * '-')
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        end_time = time.time()

        cnt+=1
        len_resp = len(outputs[0])
        print("len_resp = {}".format(len_resp))
        tot_tokens+= (len_resp - input_len)
        tot_time+=float(end_time - start_time)
        response_list.append(response)
        
    print("outputs = {}".format(outputs))
    response_tokens = tokenizer(response)
    print("response_tokens = {}".format(response_tokens))
    print(response)
    
    response_tokens_list = [tokenizer(response) for response in response_list]
    ave_resp_len = mean([len(response_tokens['input_ids']) for response_tokens in response_tokens_list]) - input_len
    ave_resp_len = len_resp - input_len  # 以这个数值为准，上一行的计算方法可能有文本转换带来的偏差
    print("ave_resp_len {}".format(ave_resp_len))
    
        
    print("ave tot time cost: {} {}".format((tot_time * 1000.0 / cnt), "ms"))
    print("ave time cost by token: {} {}".format((tot_time * 1000.0 / cnt / ave_resp_len), "ms"))
    print("throughput: {} {}".format((tot_tokens / tot_time), "tokens/s"))