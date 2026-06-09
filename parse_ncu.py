import csv


def parse_ncu_csv(file_path):
    kernels = {}
    current_kernel = None

    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kname = row.get("Kernel Name", "").strip()
            if kname:
                if "cutlass_k2_kernel" in kname:
                    current_kernel = "cula"
                elif "_flash_kda_fwd_recurrence" in kname:
                    current_kernel = "cpp"
                else:
                    current_kernel = None

            if current_kernel:
                metric_name = row.get("Metric Name", "").strip()
                metric_value = row.get("Metric Value", "").strip()
                if metric_name and metric_value:
                    if current_kernel not in kernels:
                        kernels[current_kernel] = {}
                    try:
                        val = float(metric_value.replace(",", ""))
                        kernels[current_kernel][metric_name] = val
                    except ValueError:
                        pass
    return kernels


def print_comparison(kernels):
    if "cula" not in kernels or "cpp" not in kernels:
        return

    cula_data = kernels["cula"]
    cpp_data = kernels["cpp"]

    mapping = {
        "Total Instructions": ["smsp__inst_executed.sum", "Executed Instructions"],
        "ALU Pipe": ["smsp__inst_executed_pipe_alu.sum"],
        "LSU Pipe": ["smsp__inst_executed_pipe_lsu.sum"],
        "XU Pipe": ["smsp__inst_executed_pipe_xu.sum"],
        "Tensor Pipe": ["smsp__inst_executed_pipe_tensor.sum"],
        "Bank Conflicts": ["l1tex__data_bank_conflicts_pipe_lsu.sum"],
        "MMA Pipe Util": ["smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"],
        "ALU Pipe Util": ["smsp__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed"],
        "Shared Store Bytes": ["l1tex__t_bytes_pipe_lsu_mem_shared_op_st.sum"],
        "Warp Utilization": ["smsp__warps_active.avg.pct_of_peak_sustained_active"],
        "Eligible Warps": ["smsp__warps_eligible.avg.per_cycle_active", "Eligible Warps Per Scheduler"],
        "Waves/SM": ["launch__waves_per_multiprocessor", "Waves Per SM"],
    }

    print("| Metric | cula | cpp | Diff (%) |")
    print("| :--- | :---: | :---: | :---: |")

    for label, names in mapping.items():
        v_cula = 0.0
        for n in names:
            if n in cula_data:
                v_cula = cula_data[n]
                break

        v_cpp = 0.0
        for n in names:
            if n in cpp_data:
                v_cpp = cpp_data[n]
                break

        diff = 0.0
        if v_cpp != 0:
            diff = (abs(v_cula - v_cpp) / v_cpp) * 100
        elif v_cula != 0:
            diff = 100.0

        print(f"| {label} | {v_cula:,.2f} | {v_cpp:,.2f} | {diff:.1f}% |")


if __name__ == "__main__":
    data = parse_ncu_csv("/tmp/ncu_n_metrics.csv")
    print_comparison(data)
