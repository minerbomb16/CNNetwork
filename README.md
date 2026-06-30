# Architectural Memory and Computational Bottlenecks in CNN Training in Julia

**📖 [READ THE FULL ARTICLE: Architectural Memory and Computational Bottlenecks in CNN Training in Julia (PDF)](docs/Architectural_Memory_and_Computational_Bottlenecks_in_CNN_Training_in_Julia.pdf)**

---

## 👥 Authors
This project was developed collaboratively by a 2-person team:
* **Michał Bibrzycki**
* **[Mikołaj Tradecki](https://github.com/s3r10us3r)**

---

## 📝 Project Overview
Modern deep learning frameworks often prioritize flexibility over raw performance, leading to significant memory and computational overhead on CPUs. This project investigates these bottlenecks, focusing on dynamic memory allocation and type instability in standard Automatic Differentiation (AD) engines. 

To address these issues, we developed a custom AD framework in Julia from scratch. By using a single, pre-allocated memory pool and flat array indexing, our architecture completely bypasses the Garbage Collector (GC) during the training loop. We validated our framework by training a Convolutional Neural Network (CNN) on the FashionMNIST dataset and benchmarked it against Flux.jl and PyTorch. Our results show that this zero-allocation approach drastically reduces memory overhead and execution time, demonstrating that strict, static memory management is highly effective for CPU-bound deep learning.

---

## 📊 Benchmark Results
The evaluations were performed on an AMD Ryzen 7 processor. The models were trained on the FashionMNIST dataset utilizing a batch size of 10.

### Table I: Memory Pre-allocation Benchmark
| Framework | Time (ms) | Allocated Memory (KiB) |
| :--- | :--- | :--- |
| **Ours** | 2.138 | 7 656.4 |
| **Flux.jl** | 0.081 | 533.0 |
| **PyTorch** | 1.695 | 264.5 |

### Table II: Convolutional Layer Benchmark
| Framework | Pass | Time (us) | Alloc. (KiB) |
| :--- | :--- | :--- | :--- |
| **Ours** | Forward<br>Backward | 30.72<br>42.25 | 0<br>0 |
| **Flux.jl** | Forward<br>Backward | 196.98<br>329.35 | 217.88<br>88.57 |
| **PyTorch** | Forward<br>Backward | 101.00<br>374.71 | 183.75<br>30.84 |

### Table III: Max Pooling Layer Benchmark
| Framework | Pass | Time (us) | Alloc. (KiB) |
| :--- | :--- | :--- | :--- |
| **Ours** | Forward<br>Backward | 126.17<br>11.90 | 0<br>0 |
| **Flux.jl** | Forward<br>Backward | 17.44<br>212.03 | 51.71<br>185.20 |
| **PyTorch** | Forward<br>Backward | 137.81<br>204.96 | 91.88<br>38.27 |

### Table IV: Flatten Layer Benchmark
| Framework | Pass | Time (us) | Alloc. (KiB) |
| :--- | :--- | :--- | :--- |
| **Ours** | Forward<br>Backward | 0.02<br>0.02 | 0<br>0 |
| **Flux.jl** | Forward<br>Backward | 4.31<br>0.72 | 1.05<br>0.20 |
| **PyTorch** | Forward<br>Backward | 5.27<br>18.11 | 0<br>0 |

### Table V: Dense Layer Benchmark
| Framework | Pass | Time (us) | Alloc. (KiB) |
| :--- | :--- | :--- | :--- |
| **Ours** | Forward<br>Backward | 12.05<br>113.58 | 0<br>0 |
| **Flux.jl** | Forward<br>Backward | 19.60<br>355.53 | 6.90<br>289.03 |
| **PyTorch** | Forward<br>Backward | 43.72<br>76.13 | 3.28<br>288.20 |

### Table VI: ReLU Layer Benchmark
| Framework | Pass | Time (us) | Alloc. (KiB) |
| :--- | :--- | :--- | :--- |
| **Ours** | Forward<br>Backward | 0.09<br>0.10 | 0<br>0 |
| **Flux.jl** | Forward<br>Backward | 8.68<br>1.51 | 4.29<br>3.54 |
| **PyTorch** | Forward<br>Backward | 10.28<br>16.36 | 3.28<br>3.28 |

### Table VII: Dropout Layer Benchmark
| Framework | Pass | Time (us) | Alloc. (KiB) |
| :--- | :--- | :--- | :--- |
| **Ours** | Forward<br>Backward | 0.38<br>0.11 | 0<br>0 |
| **Flux.jl** | Forward<br>Backward | 8.97<br>4.75 | 9.30<br>3.57 |
| **PyTorch** | Forward<br>Backward | 33.90<br>20.10 | 6.56<br>0 |

### Table VIII: Fused Softmax/Loss Benchmark
| Framework | Pass | Time (us) | Alloc. (KiB) |
| :--- | :--- | :--- | :--- |
| **Ours** | Forward<br>Backward | 1.03<br>0.07 | 0<br>0 |
| **Flux.jl** | Forward<br>Backward | 6.70<br>3.09 | 3.78<br>2.30 |
| **PyTorch** | Forward<br>Backward | 18.54<br>51.12 | 0.40<br>0 |

### Table IX: Full Training Benchmark (3 Epochs)
| Framework | Time (s) | Total Allocations (GiB) |
| :--- | :--- | :--- |
| **Ours** | 13.12 | 0 |
| **Flux.jl** | 38.77 | 24.24 |
| **PyTorch** | 18.30 | 18.60 |