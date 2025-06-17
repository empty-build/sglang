import os
import sys

def func(label_value, a, b, c, d, e, f):
    print("""    case {}: {{
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME({},{},{},{},{},{})::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
        break;
    }}""".format(label_value, a, b, c, d, e, f))

    # print("GENERATE_SM90_FP8_CONFIG({},{},{},{},{},{})".format(a, b, c, d, e, f))

def func_co(label_value, a, b, c, d, e, f):
    print("""    case {}: {{
        using Cutlass3xW4A8GemmKSelected = typename JOIN_STRUCT_NAME_CO({},{},{},{},{},{})::Cutlass3xW4A8Gemm;
        cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmKSelected>(
            d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
            problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
        break;
    }}""".format(label_value, a, b, c, d, e, f))

    # print("GENERATE_SM90_FP8_CO_CONFIG({},{},{},{},{},{})".format(a, b, c, d, e, f))



def func_2(label):
    print("export label={}".format(label))
    print("echo ${label}")
    print("export gemm_id={}".format(1))
    print("python3 new_bench.py")

def call_func_with_all_combinations(lists, func):

    if len(lists) != 6:
        return
    list1, list2, list3, list4, list5, list6 = lists
    #print("export gemm_id={}".format(1))
    #print("export CUDA_VISIBLE_DEVICES=6")
    for a in list1:
        for b in list2:
            for c in list3:
                for d in list4:
                    for e in list5:
                        for f in list6:
                            
                            label_value = int((a/64) * 1e5 + (b/64) * 1e4 + (c /64) * 1e3 + (d/1) * 1e2 + (e/1) * 1e1  + (f/1) * 1e0) 
                            print(f"a {a}, b {b}, c {c}, d {d}, e {e}, f {f}, label:{label_value}")
                            # func_2(label_value)
                            # func(label_value, a, b, c, d, e, f)
                            # func_co(label_value, a, b, c, d, e, f)
    #print("unset label")
    #print("python3 new_bench.py")
                            
                       
                          

if __name__ == "__main__":

#std::vector<std::vector<int>> c_options = {{1,2},{1,2},{1}};
# 7168121  :<64, 64, 128>, <1, 2, 1>
# 9088121  :<64, 256, 128>, <1, 2, 1>
#std::vector<std::vector<int>> t_options = {{64, 128, 256}, {32, 64, 128, 256}, {128, 256}};    
    # pingpong
    # list1 = [64, 128]
    # list2 = [16, 32, 64, 128]
    # list1 = [256]
    # list2 = [16, 32]
    # coorperative
    list1 = [128]
    list2 = [16, 32, 64]
    # list1 = [256]
    # list2 = [16, 32]
    list3 = [512]
    list4 = [1, 2]
    list5 = [1, 2]
    list6 = [1]

    all_lists = [list1, list2, list3, list4, list5, list6]
    
    call_func_with_all_combinations(all_lists, func)
