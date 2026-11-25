# Calibration Pipeline and Analytical Cost Model Architecture (`mgcm`)

Unified specification of the analytical total-cost model $C(P, \theta, R)$ and calibration microbenchmark suite (`MB1..MB6`) for subgraph boundary selection (`MatMul -> Add(bias) -> Activation`) across the `OpenVINO -> MLIR (GraphCompiler) / Slang -> SPIR-V -> Vulkan Runtime` compilation stack.

---

## 1. Problem Formulation and Objective Function

For an operator block occurring $K_{\text{same}} \ge 1$ times in a model graph with an identical structural signature `(M, N, K, trA, trB, dtype, epilogue)` over $R \ge 1$ model inferences, the effective execution reuse count of the compiled kernel is $R_{\text{eff}} = K_{\text{same}} \cdot R$.

Given a legal subgraph partition $P = \{k_1, \dots, k_{|P|}\}$ and matrix kernel schedule $\theta = (BM, BN, BK, TM, TN, U_K, \text{split\_k}, \mathbb{I}_{\text{smem}})$, the total execution and compilation cost is:

$$C(P, \theta, R_{\text{eff}}) = T_{\text{compile}}^{\text{total}}(P, \theta) + R_{\text{eff}} \cdot T_{\text{execute}}(P, \theta)$$

---

## 2. Stage 1: Total Compilation Cost $T_{\text{compile}}^{\text{total}}(P, \theta)$

### 2.1. Two-Stage Linear Compilation Model with Pipeline Cache Lookup
Each kernel $k \in P$ undergoes host frontend lowering $c_k \in \{\text{GC}, \text{Slang}\}$ (`IR/Source -> SPIR-V`) followed by GPU driver compilation (`SPIR-V -> Native ISA` + deferred first-dispatch JIT), or incurs a cache lookup cost `t_cache_hit_ms` upon a pipeline cache hit:

$$T_{\text{compile}}^{\text{total}}(P, \theta) = \sum_{k \in P} \Bigg[ \mathbb{I}_{\text{fe\_miss}}(k)\Big(\alpha_0^{(c_k)} + \alpha_1^{(c_k)} \widehat{L}_{\text{spv}}^{(c_k)}(k, \theta)\Big) + \mathbb{I}_{\text{drv\_miss}}(k)\Big(\beta_0^{\text{drv}} + \beta_1^{\text{drv}} \widehat{L}_{\text{spv}}^{(c_k)}(k, \theta)\Big) + \big(1 - \mathbb{I}_{\text{drv\_miss}}(k)\big) t_{\text{cache\_hit}} \Bigg]$$

* **Theoretical Justification & Applicability Domain:** Single-pass SSA construction and Linear Scan register allocation in JIT backends exhibit $O(L_{\text{IR}})$ linear scaling in instruction length (*Poletto & Sarkar, ACM TOPLAS 1999*; *Braun et al., CC 2013*), while amortized operator-fusion trade-offs follow $T_C + R \cdot T_X$ (*Niu et al., DNNFusion, ACM PLDI 2021*; *Kamei & Costa, CBSoft 2026*). Linearity over $L_{\text{spv}} \le 12\,000$ words is empirically verified per device via $R^2$ stored in `DeviceProfile` (validity threshold: $R^2 \ge 0.95$).

### 2.2. Analytical Pre-Compilation SPIR-V Length Estimator $\widehat{L}_{\text{spv}}^{(c)}(k, \theta)$
Because boundary selection runs prior to invoking `GC` or `Slang`, the 32-bit SPIR-V word length is predicted deterministically via syntax-directed translation rules (*Khronos SPIR-V Spec v1.3, §3*; *Aho et al., Compilers, 2nd ed., Ch. 5*):

$$S_{\text{ld}}(\theta) = \left\lceil \frac{BM \cdot BK}{N_{\text{threads}}(\theta)} \right\rceil + \left\lceil \frac{BK \cdot BN}{N_{\text{threads}}(\theta)} \right\rceil$$

$$\widehat{L}_{\text{spv}}^{(c)}(k, \theta) = \big(l_0^{(c)} + l_{\text{buf}}^{(c)} B_k\big) + \mathbb{I}_{\text{mm}}(k)\Big(l_{\text{skel}}^{(c)} + \mathbb{I}_{\text{smem}} S_{\text{ld}}(\theta) l_{\text{smem}}^{(c)} + U_K\big[(TM + TN)l_{\text{rld}}^{(c)} + (TM \cdot TN)l_{\text{fma}}^{(c)}\big]\Big) + U_{\text{out}}(k, \theta)\Big(l_{\text{store}}^{(c)} + \sum_{o \in \text{ep}(k)} \Delta l_{\text{op}}^{(c)}(o)\Big)$$

where $B_k$ is the bound `StorageBuffer` count, $U_K \in \{1, BK\}$ is the K-loop unroll factor, and $U_{\text{out}}(k, \theta)$ equals $1$ for standalone elementwise kernels and $TM \cdot TN$ for register-fused epilogues.

---

## 3. Stage 2: Execution Cost $T_{\text{execute}}(P, \theta)$ (Intra-GPU Execution)

### 3.1. Command Dispatch, Synchronization, and Smooth $L_2$-Roofline
For GPU-resident subgraphs batched into a single `VkCommandBuffer`:

$$T_{\text{execute}}(P, \theta) = |P| \cdot T_{\text{launch}} + (|P| - 1) \cdot T_{\text{barrier}} + \sum_{k \in P} \sqrt{T_{\text{comp}}(k, \theta)^2 + T_{\text{mem}}(k, \theta)^2}$$

* **References:** Serial command launch and RAW barrier drain overheads (*Volkov, UC Berkeley PhD Thesis 2016*; *Lustig & Martonosi, IEEE HPCA 2013*); balanced $L_2$-norm compute/memory pipeline overlap (*Williams et al., CACM 2009*; *Czechowski et al., ACM PPoPP 2012*; *Konstantinidis & Cotronis, IEEE PDP 2015*).

### 3.2. Kernel Compute Time $T_{\text{comp}}(k, \theta)$ (Waves, Tails, and Register Pressure)
* **2D Tiled Matrix Kernel ($\mathbb{I}_{\text{mm}}(k) = 1$):**
  $$G_M = \left\lceil \frac{M}{BM} \right\rceil,\quad G_N = \left\lceil \frac{N}{BN} \right\rceil,\quad G_k = G_M \cdot G_N \cdot \text{split\_k},\quad M_{\text{pad}} = G_M BM,\quad N_{\text{pad}} = G_N BN,\quad K_{\text{pad}} = \left\lceil \frac{K}{BK} \right\rceil BK$$
  $$\text{FLOPs}_{\text{mm}}^{\text{pad}} = 2 M_{\text{pad}} N_{\text{pad}} K_{\text{pad}},\qquad E_{\text{pad}} = M_{\text{pad}} N_{\text{pad}}$$
* **1D Elementwise Kernel ($\mathbb{I}_{\text{mm}}(k) = 0$, $W_{\text{wg}} = 256$):**
  $$G_k = \left\lceil \frac{M \cdot N}{W_{\text{wg}}} \right\rceil,\qquad \text{FLOPs}_{\text{mm}}^{\text{pad}} = 0,\qquad E_{\text{pad}} = G_k \cdot W_{\text{wg}}$$
* **Wave Quantization, Latency Hiding, and Register Fusion Penalty:**
  With $N_{\text{waves}} = \lceil G_k / G_w \rceil$ and $\kappa_{\text{lat}} = \max\!\big(0,\, \frac{\Pi_{\text{FMA}}}{\Pi_{\text{wave1}}} - 1\big)$ (*Hong & Kim, ISCA 2009*; *Zhu et al., ROLLER, USENIX OSDI 2022*):
  $$\eta_{\text{occ}}(G_k) = \frac{G_k}{G_w \cdot N_{\text{waves}}} \cdot \frac{N_{\text{waves}}}{N_{\text{waves}} + \kappa_{\text{lat}}}$$
  When a transcendental activation is fused into a matrix kernel ($n_{\text{act,ep}} > 0$ and $\mathbb{I}_{\text{mm}}(k) = 1$), the inner matrix loop throughput is scaled by the empirically calibrated register-pressure factor $\eta_{\text{fuse}} = \text{fusion\_reg\_factor} \in (0, 1]$ from `MB4` ($\eta_{\text{fuse}} = 1$ otherwise), capturing occupancy drops or register spilling at large $K$ on register-constrained architectures.
* **Total Heterogeneous Pipe Compute Time (Dimensionally Matched):**
  $$T_{\text{comp}}(k, \theta) = \frac{1}{\eta_{\text{occ}}(G_k)} \left( \frac{\text{FLOPs}_{\text{mm}}^{\text{pad}}}{\eta_{\text{tail}}(k) \cdot \eta_{\text{fuse}}(k) \cdot \Pi_{\text{GEMM}}(\theta)} + \frac{E_{\text{pad}} \cdot n_{\text{fma,ep}}}{\Pi_{\text{FMA}}} + \frac{E_{\text{pad}} \cdot n_{\text{act,ep}}}{\Pi_{\text{ACT}}(\text{ep}_k)} \right)$$
  where $n_{\text{fma,ep}}$ is the elementary vector op count per element, $n_{\text{act,ep}} \in \{0, 1\}$ is the macro-activation invocation count per element, and $\Pi_{\text{ACT}} \in \{\text{pi\_gelu\_erf\_gelem\_s},\, \text{pi\_gelu\_tanh\_gelem\_s}\}$ is measured directly in `GElem/s`.

### 3.3. Kernel Memory Time $T_{\text{mem}}(k, \theta)$
Using piecewise log-linear ($\log_2 W$) interpolation over the 22-point empirical `bw_curve` from `MB2` (*Mei & Chu, IEEE TPDS 2017*; *Ilic et al., IEEE CAL 2014*) yields $\text{BW}_{\text{emp}}(W)$:
* **1D Elementwise Kernel ($\mathbb{I}_{\text{mm}}(k) = 0$):**
  $$T_{\text{mem}}(k) = \frac{V_{\text{in}}(k) + V_{\text{out}}(k)}{\text{BW}_{\text{emp}}(4MN)}$$
* **2D Tiled Matrix Kernel ($S_A = 4MK, S_B = 4KN, S_C = 4MN$):**
  $$V_{\text{cold}} = S_A + S_B + S_{\text{ep\_in}} + S_C,\qquad V_{\text{reuse}} = (G_N - 1) S_A + (G_M - 1) S_B$$
  $$T_{\text{mem}}(k, \theta) = \frac{V_{\text{cold}}}{\text{BW}_{\text{emp}}(S_A + S_B + S_C)} + \frac{V_{\text{reuse}}}{\text{BW}_{\text{emp}}\!\big(\min(S_A + S_B,\, G_w \cdot 4K(BM + BN))\big)} + \mathbb{I}_{\text{split\_k}>1}\frac{\text{split\_k}\cdot S_C}{\text{BW}_{\text{emp}}(S_C)}$$
* **Untiled Naive Matrix Kernel:** applies the measured coalescing penalty $\gamma_{\text{coal}}(S, S_B)$ from `MB3` to strided $B$ reads ($M \cdot S_B$).

*(Heterogeneous Extension Note: For cross-device `GPU <-> CPU` fallback cuts of size $S_e$, $T_{\text{execute}}$ adds $T_{\text{submit}}$ plus $S_e / \text{BW}_{\text{host}} + S_e / \text{BW}_{\text{PCIe}}$ on `DesktopDiscrete`, or cache-coherency sync $T_{\text{coherency\_sync}}$ on `UnifiedUma` Zero-Copy).*

### 3.4. Exact Horizon Decision Map via Lower Envelope
Substituting block dimensions yields straight lines $C_i(R_{\text{eff}}) = b_i + k_i \cdot R_{\text{eff}}$. Computing the lower envelope (*de Berg et al., Computational Geometry, Ch. 8/11*) prunes Pareto-dominated plans and yields exact switching thresholds $R^*_{A \to B} = (b_B - b_A) / (k_A - k_B)$. By **Common-Mode Cancellation**, the shared $T_{\text{MatMul}}$ bulk cancels out in $C_a - C_b$, isolating the directly calibrated boundary terms.

---

## 4. Unified `DeviceProfile` Schema (Fields 1–24)

| # | C++ / JSON Key (`DeviceProfile`) | Unit | Calibration Microbenchmark & Extraction Protocol |
| :-: | :--- | :---: | :--- |
| **1–2** | `alpha_0_slang_ms`, `alpha_1_slang_ms_per_word` | `ms`, `ms/w` | `MB6`: OLS fit ($M=6$ lengths, median of $N=7$ salted in-process `libslang` runs) |
| **3–4** | `alpha_0_gc_ms`, `alpha_1_gc_ms_per_word` | `ms`, `ms/w` | `MB6`: OLS fit for host `GraphCompiler` MLIR passes + `LLJIT` |
| **5–6** | `beta_0_drv_ms`, `beta_1_drv_ms_per_word` | `ms`, `ms/w` | `MB6`: OLS fit ($M=6$ lengths, median of $N=7$ SPIR-V constant mutations, `create + lazy`) |
| **7** | `t_cache_hit_ms` | `ms` | `MB6`: warm `vkCreateComputePipelines` lookup latency on populated `VkPipelineCache` |
| **8–9** | `r2_fe`, `r2_drv` *(+ `calibration_wall_time_ms`)* | `[0..1]`, `ms` | `MB6`: linearity validation ($R^2 \ge 0.95$) and end-to-end calibration budget |
| **10** | `t_submit_cpu_us` | `us` | `MB1`: `cpu_wall(1) - gpu_exec(1)` |
| **11–12** | `t_launch_gpu_us`, `t_barrier_gpu_us` | `us` | `MB1`: OLS slope over $N \in \{1, 8, 32, 128\}$ chains without and with RAW barriers |
| **13** | `bw_curve` *(22 points)* | `(B, GB/s)` | `MB2`: bijective permutation scan `32 KB .. 256 MB` (`P50` median and `IQR`) |
| **14–16** | `l2_effective_bytes`, `bw_l2_gbps`, `bw_dram_gbps` | `B`, `GB/s` | `MB2`: diagnostic L2 cliff capacity and plateau bandwidths |
| **17–18** | `coal_factor_l2_s32`, `coal_factor_dram_s32` | `(0..1]` | `MB3`: useful bandwidth ratio at `stride=32` vs `stride=1` |
| **19–21** | `gflops_matmul_smem`, `gflops_matmul_cache`, `prefer_shared_mem` | `GFLOPS`, `bool` | `MB4`: `64x64` tiled GEMM (`1024x1024x512`) via `SMEM` vs direct L1/L2 cache |
| **22** | `tail_efficiency` | `(0..1]` | `MB4`: bounds-checked unaligned `1000x1000x500` efficiency $\eta_{\text{tail}}$ (in padded FLOPs) |
| **23** | `fusion_reg_factor` | `(0..1]` | `MB4`: throughput ratio of `mb6_matmul_fused_gelu_erf` to `mb4_matmul_smem` ($\eta_{\text{fuse}}$) |
| **24** | `pi_fma_gflops`, `pi_gelu_erf_gelem_s`, `pi_gelu_tanh_gelem_s`, `g_wave_quantum`, `pi_wave1_gflops` | `GFLOPS`, `GElem/s`, `WGs` | `MB5`: 16-acc `FMA` peak, `GELU` throughput, hardware wave quantum $G_w$, and 1-wave throughput |
