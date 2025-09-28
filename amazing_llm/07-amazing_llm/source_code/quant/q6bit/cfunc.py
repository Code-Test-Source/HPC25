from ._C import unsqueeze_from_6bit, calculate_dequant
import torch
from tqdm import tqdm
import concurrent.futures

def c_unsqueeze_from_6bit(qweight: torch.Tensor) -> torch.Tensor:
    return unsqueeze_from_6bit(qweight)

def c_calculate_dequant(quant_param: torch.Tensor, zero_point: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return calculate_dequant(quant_param, zero_point, scale)

def c_recover_from_quant(qweight_lst : list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    '''
    a weight named `X` will be recoverd from `X.qweight`(uint8), `X.zero_point`(int8) and `X.scales`(float16)
    '''
    key_set = set()
    qweight_dic = {}
    res_lst = []
    
    # 分离量化权重和普通权重
    for namei, ti in qweight_lst:
        if 'norm' not in namei:
            key_set.add(namei[:namei.rfind('.')])
            qweight_dic[namei] = ti
        else:
            res_lst.append((namei, ti))
    
    # 使用多线程并行处理量化权重
    def process_key(ki):
        qweighti = qweight_dic[ki+".qweight"]
        zero_pointi = qweight_dic[ki+".zero_point"] 
        scalesi = qweight_dic[ki+".scales"]
        return (ki, c_calculate_dequant(qweighti, zero_pointi, scalesi))
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_key, ki): ki for ki in key_set}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Dequantizing"):
            ki, tensor = future.result()
            res_lst.append((ki, tensor))
    
    return res_lst