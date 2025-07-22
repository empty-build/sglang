from numpy import int32
import torch
import triton

from sglang.srt.layers.moe.ep_moe.kernels import (
    run_moe_ep_preproess,
    run_cutlass_moe_ep_preproess,
)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        # x_vals=[1024],
        x_log=False,
        line_arg="provider",
        line_vals=["ep_preprocess", "preprocess"],
        line_names=["ep_preprocess", "preprocess"],
        styles=[("blue", "-"), ("orange", "-")],
        ylabel="ms",
        plot_name="ep_preprocess vs preprocess",
        args={},
    )
)
def benchmark(batch_size, provider):
    quantiles=[0.5, 0.2, 0.8]

    M, K, N = batch_size, 7168, 2048
    E = 256
    local_e = 32
    topk_ids = torch.randint(0, E-1, (M, E), dtype=torch.int32, device="cuda")
    if provider == "ep_preprocess":
        expert_map = torch.arange(E, dtype=torch.int32, device="cuda")
        expert_map[local_e:] = E
        local_topk_ids = torch.where(expert_map[topk_ids] != E,
                                    expert_map[topk_ids], E)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_cutlass_moe_ep_preproess(
                local_topk_ids,
                local_e,
            ),
            quantiles=quantiles,
        )
    if provider == "preprocess":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run_moe_ep_preproess(topk_ids, E),
            quantiles=quantiles,
        )
    return ms, min_ms, max_ms

def main():
    benchmark.run(
        show_plots=True,
        print_data=True,
    )


if __name__ == "__main__":
    main()
