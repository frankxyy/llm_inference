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
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model = model.eval()
    ques = "你好，请帮我计算一个数学题：两个苹果加上六个苹果，等于多少个苹果"
    ques_tokens = tokenizer(ques)
    print("ques_tokens = {}".format(ques_tokens))
    print("len of ques {}".format(len(ques_tokens['input_ids'])))

    cnt = 0
    tot_time = 0
    response_list = []
    for i in range(args.req_cnt):
        start_time = time.time()
        # logging.warning('start')
        response, history = model.chat(tokenizer, ques, history=[])
        # logging.warning('end')
        end_time = time.time()

        cnt+=1
        tot_time+=float(end_time - start_time)
        response_list.append(response)
        

    response_tokens = tokenizer(response)
    print("response_tokens = {}".format(response_tokens))
    print(response)
    
    response_tokens_list = [tokenizer(response) for response in response_list]
    ave_resp_len = mean([len(response_tokens['input_ids']) for response_tokens in response_tokens_list])
    print("ave_resp_len {}".format(ave_resp_len))
    
    
    print("ave tot time cost: {} {}".format((tot_time * 1000.0 / cnt), "ms"))
    print("ave time cost by token: {} {}".format((tot_time * 1000.0 / cnt / ave_resp_len), "ms"))