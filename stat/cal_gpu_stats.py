from pynvml import *

def nvidia_info():
    # pip install nvidia-ml-py
    nvidia_dict = {
        "state": True,
        "nvidia_version": "",
        "nvidia_count": 0,
        "gpus": []
    }
    try:
        nvmlInit()
        nvidia_dict["nvidia_version"] = nvmlSystemGetDriverVersion()
        nvidia_dict["nvidia_count"] = nvmlDeviceGetCount()
        for i in range(nvidia_dict["nvidia_count"]):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "gpu_name": nvmlDeviceGetName(handle),
                "total": memory_info.total,
                "free": memory_info.free,
                "used": memory_info.used,
                "temperature": f"{nvmlDeviceGetTemperature(handle, 0)}℃",
                "powerStatus": nvmlDeviceGetPowerState(handle)
            }
            nvidia_dict['gpus'].append(gpu)
    except NVMLError as _:
        nvidia_dict["state"] = False
    except Exception as _:
        nvidia_dict["state"] = False
    finally:
        try:
            nvmlShutdown()
        except:
            pass
    return nvidia_dict

def check_gpu_mem_usedRate():
    max_rate = 0.0
    num_gpu = 2
    while True:
        info = nvidia_info()
        # print(info)
        
        for gpu_id in range(num_gpu):
            used = info['gpus'][gpu_id]['used']
            tot = info['gpus'][gpu_id]['total']
            print(f"GPU {gpu_id} used: {used / 1024 / 1024 / 1024}, \
                  tot: {tot / 1024 / 1024 / 1024}, 使用率：{used/tot}")
            if used/tot > max_rate:
                max_rate = used/tot
            print("GPU {} 最大使用率：{}".format(gpu_id, max_rate) )
            print("GPU {} 最大使用：{}".format(gpu_id, max_rate*tot / 1024 / 1024 / 1024) )
        
if __name__ == "__main__":
    check_gpu_mem_usedRate()