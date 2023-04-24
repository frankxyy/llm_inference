import argparse
from statistics import mean
import time
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

def add_parser_arguments(parser):
    parser.add_argument("--model_name", type=str, default=1)
    parser.add_argument("--quan", type=str, default='fp16')
    parser.add_argument("--req_cnt", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--tp", action='store_true')
    parser.add_argument("--pp", action='store_true')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
        
    if args.model_name == 'decapoda-research/llama-7b-hf':
        from transformers import LlamaTokenizer, LlamaForCausalLM
        # with init_empty_weights():
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", trust_remote_code=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        if args.pp:
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
        else:
            model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", trust_remote_code=True)
            model = model.half().cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        if args.pp:
            with init_empty_weights():
                model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
        else:
            model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
            model = model.half().cuda()
    
    if args.pp:
        from utils import load_model_on_gpus
        model.tie_weights()
        model = load_model_on_gpus(args.model_name, model, num_gpus=args.num_gpus, device_map=None)
        
    # while 1:
    #     pass
        
    if args.tp:
        import tensor_parallel as tp
        if args.num_gpus == 2:
            model = tp.tensor_parallel(model, device_ids=["cuda:0", "cuda:1"])
        elif args.num_gpus == 4:
            model = tp.tensor_parallel(model, device_ids=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        else:
            raise Exception('num_gpus error')
        model = model.half()
    else:
        if args.quan == 'fp16' and not args.pp:
            model = model.half().cuda()
        elif args.quan == 'int8':
            model = model.quantize(8).half().cuda()
        elif args.quan == 'int4':
            model = model.quantize(4).half().cuda()
            
    
    print("type of model is {}".format(type(model)), flush=True)

    model = model.eval()
    ques = "你好,请帮我计算一个数学题:两个苹果加上六个苹果,等于多少个苹果"
    ques_tokens = tokenizer(ques)
    print("ques_tokens = {}".format(ques_tokens))
    input_len = len(ques_tokens['input_ids'])
    print("len of ques {}".format(input_len))

    cnt = 0
    tot_time = 0
    tot_tokens = 0
    response_list = []
    # while 1:
    for i in range(args.req_cnt):
        start_time = time.time()
        input_encoded = tokenizer.batch_encode_plus([ques], return_tensors="pt", padding=True)
        # print("input_encoded = {}".format(input_encoded), flush=True)
        for t in input_encoded:
            if torch.is_tensor(input_encoded[t]):
                input_encoded[t] = input_encoded[t].cuda()
                
        # response = model.chat(ques)
        outputs = model.generate(
            **input_encoded,
            num_beams=1,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            max_length = 200,
            remove_invalid_values=True,
            max_new_tokens = args.max_new_tokens
        )
        # print("Output:\n" + 100 * '-')
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
    response_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("response_tokens = {}".format(response_tokens))
    print(response)
    
    ave_output_seq_len = mean([ len(output) for output in outputs ]) - input_len
    # ave_output_seq_len = len_resp - input_len  # 以这个数值为准，上一行的计算方法可能有文本转换带来的偏差
    print("ave_output_seq_len {}".format(ave_output_seq_len))
    
        
    print("ave tot time cost: {} {}".format((tot_time * 1000.0 / cnt), "ms"))
    print("ave time cost by token: {} {}".format((tot_time * 1000.0 / cnt / ave_output_seq_len), "ms"))
    print("throughput: {} {}".format((tot_tokens / tot_time), "tokens/s"))