# Dokumentacja biblioteki CNNetwork

## Spis treści

1. [Wstęp](#1-wstęp)
2. [autodiff.jl — pamięć i propagacja gradientów](#2-autodiffjl--pamięć-i-propagacja-gradientów)
   - 2.1 [MemoryPool](#21-memorypool)
   - 2.2 [GraphNode](#22-graphnode)
   - 2.3 [alloc_weight! i alloc_act!](#23-alloc_weight-i-alloc_act)
   - 2.4 [StaticChain i @generated](#24-staticchain-i-generated)
   - 2.5 [Operacje pomocnicze](#25-operacje-pomocnicze)
3. [clothesolver.jl — warstwy sieci](#3-clothesolverjl--warstwy-sieci)
   - 3.1 [System Blueprint](#31-system-blueprint)
   - 3.2 [CompiledModel i build_model](#32-compiledmodel-i-build_model)
   - 3.3 [Warstwa Dense (w pełni połączona)](#33-warstwa-dense-w-pełni-połączona)
   - 3.4 [Warstwa ReLU](#34-warstwa-relu)
   - 3.5 [Warstwa Flatten (spłaszczanie)](#35-warstwa-flatten-spłaszczanie)
   - 3.6 [Warstwa Conv (splot)](#36-warstwa-conv-splot)
   - 3.7 [Warstwa MaxPool](#37-warstwa-maxpool)
   - 3.8 [Warstwa Dropout](#38-warstwa-dropout)
   - 3.9 [Funkcja straty LogitCrossEntropy](#39-funkcja-straty-logitcrossentropy)
4. [Optymalizacje wydajności — podsumowanie](#4-optymalizacje-wydajności--podsumowanie)

---

## 1. Wstęp

Biblioteka implementuje konwolucyjną sieć neuronową **od zera**, bez korzystania z
silnika automatycznego różniczkowania (Flux, Zygote, itp.). Zamiast tego ręcznie
wyprowadzimy i zakodujemy pochodne dla każdej warstwy. Celem jest pełna kontrola
nad układem pamięci i kosztami obliczeń.

Architektura sieci użyta w projekcie to LeNet-5 dostosowany do zbioru FashionMNIST:

```
Wejście (28×28×1×B)
  └─ Conv1   : 3×3, 1→6,  brak biasu  →  (26×26×6×B)
  └─ MaxPool1: 2×2                     →  (13×13×6×B)
  └─ Conv2   : 3×3, 6→16, brak biasu  →  (11×11×16×B)
  └─ MaxPool2: 2×2                     →  (5×5×16×B)
  └─ Flatten                           →  (400×B)
  └─ Dense1  : 400→84, ReLU           →  (84×B)
  └─ Dropout : p=0.4
  └─ Dense2  : 84→10                  →  (10×B)
  └─ Strata  : LogitCrossEntropy
```

gdzie `B` to rozmiar batcha. Parametry (`W`, `b`) oraz aktywacje (`x`, `y`) są
przechowywane w dwóch jednowymiarowych tablicach (pule pamięci), a każda warstwa
trzyma jedynie widoki (`view`) na odpowiednie fragmenty.

---

## 2. autodiff.jl — pamięć i propagacja gradientów

### 2.1 MemoryPool

```julia
struct MemoryPool
    weights::Vector{Float32}   # wagi i biasy
    w_grad ::Vector{Float32}   # gradienty wag
    w_offset::Base.RefValue{Int}

    acts   ::Vector{Float32}   # aktywacje (wyjścia warstw)
    a_grad ::Vector{Float32}   # gradienty aktywacji
    a_offset::Base.RefValue{Int}
end
```

**Po co?**
Standardowy kod Julia tworzy nową tablicę przy każdym `zeros(Float32, ...)`.
Przy 1875 batchach na epokę i kilkunastu warstwach oznaczałoby to dziesiątki tysięcy
alokacji — każda z osobną blokadą alokatora, osobnym GC-trackowaniem i osobnymi
odwołaniami do pamięci rozrzuconymi po stercie.

`MemoryPool` rezerwuje dwa ciągłe `Vector{Float32}` (jeden na wagi, jeden na aktywacje)
i przydziela każdej warstwie wycinek za pomocą `view`. Dzięki temu:

- **zero alokacji** podczas propagacji w przód/wstecz — całe obliczenia nadpisują
  istniejącą pamięć,
- wagi i aktywacje leżą w ciągłych blokach → lepsza lokalność pamięci podręcznej,
- zerowanie gradientów = jedno `fill!(pool.w_grad, 0)` zamiast N osobnych operacji.

Przykład działania:

```
weights: [  W1_flat  |  b1  |  W2_flat  |  b2  | … ]
          ^           ^       ^
          offset=1    +324    +330
```

Każda warstwa przechowuje swoje wagi jako `reshape(view(pool.weights, start:end), kształt)`.

---

### 2.2 GraphNode

```julia
struct GraphNode{T}
    data::T   # wartość (forward pass)
    grad::T   # gradient (backward pass)
end
```

`GraphNode{T}` łączy tensor wartości z tensorem gradientu tego samego kształtu.
Typ `T` jest parametryczny — Julia kompiluje osobną wersję dla każdego kształtu
(`ReshapedArray{Float32,2,...}`, `ReshapedArray{Float32,4,...}` itd.), eliminując
dynamiczny dispatch i indirekcje.

Ważne: `data` i `grad` to **widoki** na pulę, nie własne tablice. Zapis do
`node.data[i]` bezpośrednio modyfikuje odpowiedni element w `pool.acts`.

---

### 2.3 alloc_weight! i alloc_act!

```julia
function alloc_weight!(pool::MemoryPool, dims...)
    len   = prod(dims)
    start = pool.w_offset[]
    pool.w_offset[] += len
    append!(pool.weights, zeros(Float32, len))
    append!(pool.w_grad,  zeros(Float32, len))
    return GraphNode(
        reshape(view(pool.weights, start:(start+len-1)), dims),
        reshape(view(pool.w_grad,  start:(start+len-1)), dims)
    )
end
```

Funkcja działa jak inkrementalny alokator (ang. *bump allocator*):
1. Zapamiętuje bieżący offset.
2. Przesuwa offset o `len`.
3. Rozszerza wektor (amortyzowane O(1)).
4. Zwraca `GraphNode` zawierający widok na nowo przydzielony fragment.

`alloc_act!` działa identycznie, ale korzysta z pól `acts`/`a_grad`.

Wszystkie alokacje muszą nastąpić **przed** pierwszym obliczeniem — `append!`
przebudowuje wewnętrzny bufor Vectora i unieważnia stare widoki.

---

### 2.4 StaticChain i @generated

```julia
struct StaticChain{T <: Tuple} layers::T end
```

`StaticChain` przechowuje krotkę (`Tuple`) warstw. Krotki w Julii są typowane
statycznie — `StaticChain{Tuple{Conv,MaxPool,Dense,...}}` jest jednym konkretnym
typem. Kompilator zna liczbę i typy wszystkich warstw w czasie kompilacji.

Gdybyśmy użyli `Vector{Operator}`, Julia musiałaby sprawdzać typ każdej warstwy
w czasie działania programu i wywoływać `primal!` przez tabelę wirtualną.

Zamiast tego korzystamy z `@generated`:

```julia
@generated function forward!(chain::StaticChain{T}, x::GraphNode, is_training::Bool) where T
    N = length(T.parameters)
    exprs = Expr[:(curr_x = x)]
    for i in 1:N
        push!(exprs, :(primal!(chain.layers[$i], curr_x, is_training)))
        push!(exprs, :(curr_x = chain.layers[$i].out))
    end
    push!(exprs, :(return curr_x))
    return Expr(:block, exprs...)
end
```

`@generated` to **metaprogramowanie w czasie kompilacji**: funkcja przyjmuje *typy*
argumentów i zwraca wyrażenie (AST), które Julia kompiluje jako normalny kod.
Dla naszej sieci 9-warstwowej generuje:

```julia
curr_x = x
primal!(chain.layers[1], curr_x, is_training)   # Conv1
curr_x = chain.layers[1].out
primal!(chain.layers[2], curr_x, is_training)   # MaxPool1
curr_x = chain.layers[2].out
# … (9 warstw)
return curr_x
```

Cały `forward!` to **jedna inlined (wywoływana jako kod w głównym programie, nie jako funkcja z osobnym miejscem w pamięci), bezzwrotna sekwencja wywołań** — brak pętli,
brak dynamicznego dispatchu, brak tablicy wirtualnej. Każde `primal!` jest

specjalizowane dla konkretnego typu warstwy.

`backward!` robi to samo, ale iteruje warstwy **od tyłu**:

```julia
@generated function backward!(chain::StaticChain{T}, x::GraphNode, is_training::Bool) where T
    N = length(T.parameters)
    exprs = Expr[]
    for i in N:-1:1
        input_expr = i == 1 ? :(x) : :(chain.layers[$(i-1)].out)
        push!(exprs, :(adjoint!(chain.layers[$i], $input_expr, is_training)))
    end
    # …
end
```

---

### 2.5 Operacje pomocnicze

```julia
zero_w_grad!(pool) = fill!(pool.w_grad, 0.0f0)
zero_a_grad!(pool) = fill!(pool.a_grad, 0.0f0)
```

Zamiast zerować gradienty w każdej warstwie osobno, `fill!` operuje na całym ciągłym
wektorze jednym wywołaniem — procesor może użyć `memset`-like SIMD do wyzerowania
setek kilobajtów w kilka mikrosekund.

```julia
function optimize!(pool::MemoryPool, η::Float32)
    @inbounds @simd for i in eachindex(pool.weights)
        pool.weights[i] -= η * pool.w_grad[i]
    end
end
```

Klasyczny SGD (stochastic gradient descent):

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

Pętla SIMD po ciągłym wektorze wag — procesor wykonuje tę operację wektorowo
(8 elementów Float32 naraz na AVX2).

---

## 3. clothesolver.jl — warstwy sieci

### 3.1 System Blueprint

Użytkownik definiuje architekturę tak samo jak w bibliotece Flux:

```julia
my_net_def = Chain(
    Conv((3,3), 1 => 6, bias=false),
    MaxPool((2,2)),
    Dense(400 => 84, relu),
    Dropout(0.4),
    Dense(84 => 10)
)
```

`Conv(...)` nie tworzy od razu warstwy z buforami — zwraca lekką strukturę `ConvSpec`
(tzw. *blueprint*), która jedynie opisuje parametry:

```julia
struct ConvSpec <: Blueprint
    filter::Tuple{Int,Int}
    ch::Pair{Int,Int}
    bias::Bool
end
```

`Chain(...)` zbiera wszystkie blueprinty w krotkę, automatycznie rozwijając krotki
(np. `Dense(400=>84, relu)` zwraca `(DenseSpec(...), ReLUSpec())`):

```julia
function Chain(args...)
    flat = []
    for a in args
        a isa Tuple ? push!(flat, a...) : push!(flat, a)
    end
    return ChainDef(Tuple(flat))
end
```

Faktyczna alokacja pamięci i inicjalizacja wag następuje dopiero przy wywołaniu
`build_model`. Dzięki temu ten sam opis architektury można skompilować z różnymi
rozmiarami batcha bez żadnych zmian.

---

### 3.2 CompiledModel i build_model

```julia
struct CompiledModel{T, I, TG, P, O}
    chain      ::StaticChain{T}
    pool       ::MemoryPool
    input      ::GraphNode{I}          # bufor wejściowy [28×28×1×B]
    target     ::GraphNode{TG}         # bufor etykiet   [10×B]
    loss       ::LogitCrossEntropy{P,O}
    batch_size ::Int
end
```

`build_model` jest **kompilatorem sieci** — przechodzi przez listę blueprintów,
wywołuje `build_layer` dla każdego, zbiera gotowe warstwy w `StaticChain` i alokuje
bufor wejściowy, bufor etykiet oraz funkcję straty:

```julia
function build_model(def::ChainDef, input_shape; batch_size::Int=1)
    pool = MemoryPool()
    current_shape = input_shape
    compiled_layers = []

    for bp in def.blueprints
        layer, current_shape = build_layer(bp, pool, current_shape, batch_size)
        push!(compiled_layers, layer)
    end

    chain       = StaticChain(Tuple(compiled_layers))
    n_classes   = current_shape[1]          # rozmiar wyjścia ostatniej warstwy
    input_node  = alloc_act!(pool, input_shape..., batch_size)
    target_node = alloc_act!(pool, n_classes, batch_size)
    loss_fn     = LogitCrossEntropy(pool, n_classes, batch_size)
    return CompiledModel(chain, pool, input_node, target_node, loss_fn, batch_size)
end
```

Każde `build_layer` zwraca krotkę `(warstwa, output_shape)` — rozmiar wyjścia
jest wejściem do następnej warstwy. Kształty propagują się przez całą sieć
automatycznie, bez jawnego podawania wymiarów każdej warstwy.

**Dlaczego `batch_size::Int` jest ważne?**
Przechowywanie `batch_size` jako pole z konkretnym typem `Int` (a nie jako
argument słowa kluczowego `Any`) sprawia, że każda funkcja odczytująca
`model.batch_size` jest w pełni typowalna przez kompilator Julii. Poprzednia
wersja przekazywała `batch_size` jako nieokreślony argument słowa kluczowego:

```julia
# STARA wersja — batch_size::Any → boxing w pętli
function train!(model, X, Y; batch_size, η)
    for b_start in 1:batch_size:N   # typ zakresu nieznany = Any
```

```julia
# NOWA wersja — typ konkretny = brak boxingu
function train!(model::CompiledModel, X, Y; η::Float32)
    bs = model.batch_size           # Julia wie: bs::Int64
    for b_start in 1:bs:N          # zakres StepRange{Int64} — specjalizowany
```

Różnica zmierzona w benchmarku: **~2× szybciej** (5 s/epoka zamiast 10 s/epoka).

---

### 3.3 Warstwa Dense (w pełni połączona)

#### Matematyka

Warstwa Dense wykonuje transformację afiniczną:

$$Y = WX + b$$

gdzie:
- $W \in \mathbb{R}^{n_{out} \times n_{in}}$ — macierz wag
- $X \in \mathbb{R}^{n_{in} \times B}$ — batch wejść (każda kolumna to jeden przykład)
- $b \in \mathbb{R}^{n_{out}}$ — bias (rozgłaszany na każdą kolumnę)
- $Y \in \mathbb{R}^{n_{out} \times B}$ — batch wyjść

#### Propagacja wsteczna

Jeśli znamy gradient straty względem wyjścia $\frac{\partial L}{\partial Y}$,
gradienty względem $W$, $b$ i $X$ wynoszą:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \cdot X^T$$

$$\frac{\partial L}{\partial b} = \sum_{b=1}^{B} \frac{\partial L}{\partial Y_{:,b}}$$

$$\frac{\partial L}{\partial X} = W^T \cdot \frac{\partial L}{\partial Y}$$

Wszystkie trzy operacje to **mnożenia macierzy** — biblioteka BLAS (`mul!`) wykonuje
je z cache-blockingiem i optymalizacjami na poziomie assemblera.

#### Implementacja

```julia
function primal!(layer::DenseLayer, x::GraphNode, is_training::Bool)
    # Y = W × X  (jeden batched GEMM: [out×B] = [out×in] × [in×B])
    mul!(layer.out.data, layer.w.data, x.data)
    # dodaj bias do każdej kolumny
    @inbounds for b in axes(layer.out.data, 2)
        @simd for i in axes(layer.b.data, 1)
            layer.out.data[i, b] += layer.b.data[i]
        end
    end
end

function adjoint!(layer::DenseLayer, x::GraphNode, is_training::Bool)
    # dW += dY × X^T       (α=1, β=1 → akumulacja)
    mul!(layer.w.grad, layer.out.grad, x.data', 1, 1)
    # db += suma kolumn dY
    @inbounds for b in axes(layer.out.grad, 2)
        @simd for i in axes(layer.b.grad, 1)
            layer.b.grad[i] += layer.out.grad[i, b]
        end
    end
    # dX += W^T × dY
    mul!(x.grad, layer.w.data', layer.out.grad, 1, 1)
end
```

`mul!(C, A, B, α, β)` oblicza `C = α·A·B + β·C`. Wartość `β=1` zapewnia
**akumulację** gradientów (a nie nadpisanie) — istotne gdy wiele ścieżek
prowadzi do tego samego węzła.

#### Inicjalizacja He

```julia
function he_uniform(shape...; fan_in)
    return (rand(Float32, shape...) .- 0.5f0) .* Float32(sqrt(24.0 / fan_in))
end
```

Inicjalizacja He (Kaiming) zapewnia, że wariancja aktywacji pozostaje stała
przez głębokość sieci przy aktywacji ReLU:

$$W \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{in}}},\; \sqrt{\frac{6}{n_{in}}}\right)$$

Czynnik $24/n_{in}$ odpowiada wzorowi $6/n_{in}$ ponieważ:
$\text{Var}(\mathcal{U}[-a,a]) = a^2/3$, a $a = \sqrt{6/n_{in}}$ daje $\text{Var} = 2/n_{in}$.

---

### 3.4 Warstwa ReLU

#### Matematyka

Funkcja aktywacji Rectified Linear Unit jest elementowa:

$$y_i = \max(0,\; x_i)$$

Pochodna:

$$\frac{\partial L}{\partial x_i} = \begin{cases} \frac{\partial L}{\partial y_i} & \text{jeśli } y_i > 0 \\ 0 & \text{w przeciwnym razie} \end{cases}$$

#### Implementacja

```julia
function primal!(layer::ReLULayer, x::GraphNode, is_training::Bool)
    @inbounds @simd for i in eachindex(layer.out.data)
        layer.out.data[i] = max(0f0, x.data[i])
    end
end

function adjoint!(layer::ReLULayer, x::GraphNode, is_training::Bool)
    od = layer.out.data; og = layer.out.grad; xg = x.grad
    @inbounds @simd for i in eachindex(xg)
        xg[i] += ifelse(od[i] > 0f0, og[i], 0f0)
    end
end
```

Kluczowe: używamy `ifelse` zamiast `if/else`. Funkcja `ifelse(cond, a, b)` generuje
instrukcję `CMOV` (conditional move) — **bezgałęziową** selekcję wartości na poziomie
procesora. Umożliwia to wektoryzację SIMD całej pętli, bo procesor nie musi
przewidywać warunków gałęzi.

Przechowujemy wartość wyjścia `out.data` (a nie wejścia `x.data`) żeby sprawdzić
aktywność neuronu. Wynik forward-passu jest dostępny za darmo — nie trzeba go
ponownie obliczać ani przechowywać osobno.

---

### 3.5 Warstwa Flatten (spłaszczanie)

#### Matematyka

Flatten zmienia widok tensora z wielowymiarowego na wektor, **nie przenosząc danych**.

Tensor `[H, W, C, B]` (w pamięci: H×W×C×B elementów Float32, w kolejności
column-major) po spłaszczeniu staje się `[H*W*C, B]`. Kolejność elementów w pamięci
pozostaje **dokładnie taka sama** — tylko kształt `reshape` zmienia się.

#### Implementacja

```julia
function primal!(layer::FlattenLayer, x::GraphNode, is_training::Bool)
    copyto!(layer.out.data, x.data)
end
```

`copyto!` działa jak `memcpy` — kopiuje bajty bez żadnej reinterpretacji. Dzięki
temu nie ma żadnej transpozycji ani permutacji, a operacja jest ograniczona przez
przepustowość pamięci.

W propagacji wstecznej gradient z tensoru `[H*W*C, B]` trafia z powrotem do
`[H, W, C, B]` — ta sama sztuczka:

```julia
function adjoint!(layer::FlattenLayer, x::GraphNode, is_training::Bool)
    og = layer.out.grad; xg = x.grad
    @inbounds @simd for i in eachindex(xg)
        xg[i] += og[i]
    end
end
```

---

### 3.6 Warstwa Conv (splot)

Splot jest obliczeniowo najdroższą warstwą sieci (ok. 70% czasu backward, ok. 40%
czasu forward). Biblioteka używa techniki **im2col + BLAS GEMM**, która przekształca
splot w mnożenie macierzy.

#### Matematyka splotu

Klasyczny splot 2D (bez paddingu) dla jednego przykładu:

$$Y[h, w, c_{out}] = \sum_{c_{in}} \sum_{f_h=1}^{F_H} \sum_{f_w=1}^{F_W}
  W[f_h, f_w, c_{in}, c_{out}] \cdot X[h + f_h - 2,\; w + f_w - 2,\; c_{in}]$$

gdzie:
- $W \in \mathbb{R}^{F_H \times F_W \times C_{in} \times C_{out}}$ — filtry
- $X \in \mathbb{R}^{H_{in} \times W_{in} \times C_{in}}$ — wejście
- $Y \in \mathbb{R}^{H_{out} \times W_{out} \times C_{out}}$ — wyjście
- $H_{out} = H_{in} - F_H + 1$ (brak paddingu)

Dla filtru 3×3 i wejścia 28×28: $H_{out} = 28 - 3 + 1 = 26$.

Suma po $c_{in}, f_h, f_w$ to **iloczyn skalarny** między spłaszczonym filtrem
a spłaszczoną paczką wejścia (ang. *receptive field patch*) dla każdej pozycji
$(h, w)$ w wyjściu.

#### Technika im2col

Idea: spakować **wszystkie** paczki wejścia naraz do jednej macierzy `col_buf`,
a następnie wykonać jedno mnożenie macierzy.

```
col_buf [N × K]   gdzie N = H_out × W_out,  K = F_H × F_W × C_in

Wiersz n = (h_out-1)*H_out + w_out  ←→  pozycja wyjściowa (h_out, w_out)
Kolumna k ←→  jedna wartość paczki   (fh, fw, c_in)

col_buf[n, k] = X[h_out + fh - 2, w_out + fw - 2, c_in]
```

Po zbudowaniu `col_buf`:

$$Y_{\text{flat}} = \text{col\_buf} \cdot W_{\text{flat}}$$

gdzie $W_{\text{flat}} \in \mathbb{R}^{K \times C_{out}}$ to **zero-copy reshape**
istniejącej tablicy wag. Wynik $Y_{\text{flat}} \in \mathbb{R}^{N \times C_{out}}$
to zero-copy reshape bufora wyjściowego.

**Rozmiary buforów dla tej sieci:**

| Warstwa | N (pozycji) | K (paczka) | col_buf  | W_flat   | GEMM N×K×C_out |
|---------|-------------|------------|----------|----------|-----------------|
| Conv1   | 26×26 = 676 | 3×3×1 = 9  | 24 KB    | 9×6      | 676 × 9 × 6     |
| Conv2   | 11×11 = 121 | 3×3×6 = 54 | 26 KB    | 54×16    | 121 × 54 × 16   |

Oba bufory mieszczą się w pamięci L2 (typowo 256–512 KB).

#### Implementacja — im2col (forward)

```julia
fill!(layer.col_buf, 0.0f0)                    # zerowanie (dla paddingu)
@inbounds for c_in in 1:C_in, fw in 1:fW
    w_lo = max(1, 3 - fw)                      # analityczne granice
    w_hi = min(W_out, W_in + 2 - fw)           # (brak gałęzi w pętli)
    for fh in 1:fH
        h_lo = max(1, 3 - fh)
        h_hi = min(H_out, H_in + 2 - fh)
        k = fh + fH*(fw-1) + fH*fW*(c_in-1)   # indeks kolumny w col_buf

        for w_out in w_lo:w_hi
            in_w   = w_out + fw - 2
            n_base = H_out * (w_out - 1)
            @simd for h_out in h_lo:h_hi       # SIMD po h — stride-1
                col_buf[n_base + h_out, k] = x.data[h_out + fh - 2, in_w, c_in, b]
            end
        end
    end
end
W_flat  = reshape(layer.w.data, K, C_out)     # zero-copy
out_b   = reshape(@view(out.data[:,:,:,b]), N, C_out)
mul!(out_b, layer.col_buf, W_flat)            # GEMM
```

Granice `h_lo`/`h_hi` wyznaczane są raz na parę `(fh, fw)` — pętla `@simd for h_out`
jest **bezwarunkowa** (brak `if` sprawdzającego czy `h_out + fh - 2 ∈ [1, H_in]`).

#### Implementacja — propagacja wsteczna

Propagacja wsteczna splotu wymaga gradientów względem $W$ i $X$:

$$\frac{\partial L}{\partial W_{\text{flat}}} = \text{col\_buf}^T \cdot \frac{\partial L}{\partial Y_{\text{flat}}}$$

$$\frac{\partial L}{\partial \text{col\_buf}} = \frac{\partial L}{\partial Y_{\text{flat}}} \cdot W_{\text{flat}}^T$$

Gradient po `col_buf` jest następnie rozproszony z powrotem do tensora `x.grad`
operacją odwrotną do im2col (ang. *col2im*):

```julia
dy_b = reshape(@view(out.grad[:,:,:,b]), N, C_out)

# gradient wag (GEMM + akumulacja)
mul!(W_gflat, layer.col_buf', dy_b, 1f0, 1f0)

# gradient wejścia: col_buf ← dy × W^T
mul!(layer.col_buf, dy_b, W_flat')

# col2im: rozproszenie col_buf → x.grad
@inbounds for c_in in 1:C_in, fw in 1:fW
    for fh in 1:fH
        k = fh + fH*(fw-1) + fH*fW*(c_in-1)
        for w_out in w_lo:w_hi
            @fastmath @simd for h_out in h_lo:h_hi
                x.grad[h_out+fh-2, in_w, c_in, b] += col_buf[n_base + h_out, k]
            end
        end
    end
end
```

Kluczowa oszczędność: pętla col2im iteruje `C_in × F_W × F_H = 54` razy
(dla Conv2). Naiwna implementacja iterowałaby `C_out × C_in × F_W × F_H = 864` razy —
redukcja po $C_{out}$ jest **wbudowana w GEMM**.

---

### 3.7 Warstwa MaxPool

#### Matematyka

Max pooling wybiera maksimum w każdym oknie $P_H \times P_W$:

$$Y[h_{out}, w_{out}, c, b] = \max_{\substack{0 \le dp_h < P_H \\ 0 \le dp_w < P_W}}
  X[h_{out} \cdot P_H + dp_h + 1,\; w_{out} \cdot P_W + dp_w + 1,\; c,\; b]$$

W propagacji wstecznej gradient przechodzi tylko przez pozycję, która osiągnęła
maksimum (tzw. *przełącznik*):

$$\frac{\partial L}{\partial X[i_h, i_w, c, b]} \mathrel{+}=
  \frac{\partial L}{\partial Y[h_{out}, w_{out}, c, b]}
  \cdot \mathbf{1}[i_h, i_w \text{ jest argmax}]$$

Aby backward działał, forward musi zapamiętać, **gdzie** było maksimum (`argmax`).

#### Kodowanie argmax

Standardowe przechowywanie pozycji jako `Tuple{Int,Int}` zajmuje **16 bajtów**
na element. Dla MaxPool1 (rozmiar wyjścia 14×14×6×32) daje to:
$14 \times 14 \times 6 \times 32 \times 16 = 591\,360 \text{ B} \approx 578 \text{ KB}$

To prawie cały L2! Zamiast tego używamy **spakowanego Int32**:

```julia
# Kodowanie: dolne 8 bitów = in_h,  górne 8 bitów = in_w
mi[h] = Int32(in_h) | (Int32(in_w) << 8)

# Dekodowanie:
in_h = Int(idx & 0xFF)
in_w = Int((idx >> 8) & 0xFF)
```

Zajętość: `4 B/element × 37632 = 147 KB` dla MaxPool1 — **4× mniej**.

#### Bezgałęziowa pętla SIMD

Naiwna implementacja z gałęzią `if val > max_val` wewnątrz pętli po `h`
powoduje ~50% misprediction (wartości maksymalne rozkładają się równomiernie
po oknie) — mierzony koszt: 2–3× wolniej przy dużych batchach.

Rozwiązanie: przenieść iterację po oknie poolingu **na zewnątrz** pętli po `h`,
a porównanie zastąpić `ifelse`:

```julia
mv = layer.max_val_buf   # wektor długości H_out
mi = layer.max_idx_buf

@inbounds for b in 1:B, c in 1:C, w_out in 1:W_out
    @simd for h in 1:H_out
        mv[h] = -Inf32; mi[h] = Int32(0)     # inicjalizacja kolumny
    end

    for dpw in 0:pw-1, dph in 0:ph-1         # okno PONAD pętlą h
        iw = (w_out - 1) * pw + dpw + 1
        @simd for h in 1:H_out               # pętla SIMD — brak gałęzi
            ih  = (h - 1) * ph + dph + 1
            val = x.data[ih, iw, c, b]
            upd = val > mv[h]
            mv[h] = ifelse(upd, val,                           mv[h])
            mi[h] = ifelse(upd, Int32(ih) | (Int32(iw) << 8), mi[h])
        end
    end

    @simd for h in 1:H_out                   # zapis do out i argmax
        layer.out.data[h, w_out, c, b] = mv[h]
        layer.argmax[h, w_out, c, b]   = mi[h]
    end
end
```

Wewnętrzna pętla `@simd for h` jest w pełni wektoryzowalna:
- dostępy do `x.data` są stride-`P_H` w `h` (stały krok, kompilator może to wektoryzować),
- `mv[h]` i `mi[h]` to stride-1,
- brak gałęzi — `ifelse` generuje `VMAXPS`/`VBLENDVPS` lub `VCMPPS`+`VPBLEND`.

---

### 3.8 Warstwa Dropout

#### Matematyka

Dropout (Srivastava et al., 2014) losowo zeruje neurony podczas treningu
z prawdopodobieństwem `p`, skalując pozostałe przez `1/(1-p)` (odwrócony dropout):

$$y_i = \begin{cases} \frac{x_i}{1-p} & \text{z prawdopodobieństwem } 1-p \\ 0 & \text{z prawdopodobieństwem } p \end{cases}$$

Cel: redukcja nadmiernego dopasowania (overfitting) poprzez zmuszanie sieci do
uczenia się redundantnych reprezentacji. Przy ewaluacji dropout jest wyłączony
(`copyto!` bez skalowania).

W propagacji wstecznej gradient przechodzi tylko przez aktywne neurony:

$$\frac{\partial L}{\partial x_i} = \begin{cases} \frac{1}{1-p} \cdot \frac{\partial L}{\partial y_i} & \text{jeśli maska}[i] = 1 \\ 0 & \text{w przeciwnym razie} \end{cases}$$

#### Stabilność typów (kluczowa optymalizacja)

Wczesna wersja struktury:

```julia
# ZŁA wersja — mask::Array{Bool} to UnionAll (brak wymiaru = nieokreślony)
struct DropoutLayer{O} <: Operator
    mask    ::Array{Bool}     # ← UnionAll, nie konkretny typ
    rand_buf::Array{Float32}  # ← UnionAll
    out     ::GraphNode{O}
    p       ::Float32
end
```

`Array{Bool}` bez podanego wymiaru jest typem `UnionAll` — Julia nie zna liczby
wymiarów, więc nie może wygenerować specjalizowanego kodu dla pętli. Każdy element
`mask[i]` musi być *opakowany* (boxing) w tymczasowy obiekt sterty.
Wynik: **147 KB alokacji na jeden forward pass** dla batcha 32.

```julia
# DOBRA wersja — typy konkretne poprzez parametry
struct DropoutLayer{M <: AbstractArray{Bool}, R <: AbstractArray{Float32}, O} <: Operator
    mask    ::M   # M = Matrix{Bool} — znany rozmiar i krok
    rand_buf::R   # R = Matrix{Float32}
    out     ::GraphNode{O}
    p       ::Float32
end
```

`zeros(Bool, 84, 32)` zwraca `Matrix{Bool}` = `Array{Bool,2}` — konkretny typ.
Julia specjalizuje `DropoutLayer{Matrix{Bool}, Matrix{Float32}, ...}`, eliminując
boxing. Alokacje: 147 KB → **0 bajtów** na forward pass.

#### Implementacja

```julia
function primal!(layer::DropoutLayer, x::GraphNode, is_training::Bool)
    if is_training
        rand!(layer.rand_buf)                    # losowanie in-place (brak alokacji)
        scale = 1.0f0 / (1.0f0 - layer.p)
        p = layer.p; mask = layer.mask; rb = layer.rand_buf
        xd = x.data; od = layer.out.data
        @inbounds @simd for i in eachindex(od)
            m      = rb[i] > p                   # Bool
            mask[i] = m
            od[i]  = ifelse(m, xd[i] * scale, 0.0f0)
        end
    else
        copyto!(layer.out.data, x.data)          # ewaluacja: bez skalowania
    end
end
```

`rand!` wypełnia istniejący bufor wartościami losowymi — brak alokacji tablicy
tymczasowej. `ifelse` zamiast `?:` → bezgałęziowe SIMD.

---

### 3.9 Funkcja straty LogitCrossEntropy

#### Matematyka

Sieć zwraca **logity** (surowe wartości przed softmax). Strata to
entropię krzyżową po softmax, ale obliczaną razem dla stabilności numerycznej.

**Softmax:**

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Problem:** jeśli $\max(z)$ jest duże (np. 100), to $e^{100}$ przepełnia Float32.

**Rozwiązanie — stabilna wersja:**

$$\text{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \quad m = \max_j z_j$$

Odjęcie $m$ nie zmienia wartości softmax ($e^{-m}$ dzieli zarówno licznik jak
i mianownik), ale gwarantuje $\max(z - m) = 0$, więc $e^{z_i - m} \le 1$.

**Entropia krzyżowa:**

$$L = -\sum_{i=1}^{C} y_i^{\text{true}} \cdot \log(\hat{y}_i + \varepsilon)$$

gdzie $\varepsilon = 10^{-10}$ zapobiega $\log(0)$.

**Gradient (logitów):**

$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i^{\text{true}}$$

To elegancki wynik — gradient logitów to po prostu różnica między predykcją
a prawdziwą wartością. Żadnych pochodnych softmax ani log.

#### Implementacja

```julia
function primal!(layer::LogitCrossEntropy, y_pred::GraphNode, y_true::GraphNode)
    C = size(y_pred.data, 1); B = size(y_pred.data, 2)
    total_loss = 0.0f0
    @inbounds for b in 1:B
        # krok 1: stabilny softmax
        m = -Inf32
        @simd for i in 1:C; m = max(m, y_pred.data[i, b]) end   # max logitu

        sum_exp = 0.0f0
        @fastmath @simd for i in 1:C
            layer.probs.data[i, b] = exp(y_pred.data[i, b] - m)
            sum_exp += layer.probs.data[i, b]
        end

        # krok 2: normalizacja + entropia krzyżowa
        l = 0.0f0
        @fastmath @simd for i in 1:C
            layer.probs.data[i, b] /= sum_exp
            l -= y_true.data[i, b] * log(layer.probs.data[i, b] + 1f-10)
        end
        total_loss += l
    end
    layer.out.data[1] = total_loss / B     # strata uśredniona po batchu
    return total_loss / B
end

function adjoint!(layer::LogitCrossEntropy, y_pred::GraphNode, y_true::GraphNode)
    B = size(y_pred.data, 2)
    g = layer.out.grad[1] / B   # dzielenie przez B: gradient średniej straty
    @fastmath @inbounds @simd for i in eachindex(y_pred.grad)
        y_pred.grad[i] += (layer.probs.data[i] - y_true.data[i]) * g
    end
end
```

Skalowanie przez `1/B` w `adjoint!` jest konieczne: `primal!` zwraca **średnią**
straty po batchu, więc gradient każdego przykładu musi być przez B podzielony,
żeby SGD zachował semantykę uczenia "jeden przykład = jedna jednostka gradientu".

---

## 4. Optymalizacje wydajności — podsumowanie

| Optymalizacja | Plik | Efekt |
|---|---|---|
| `mul!` zamiast `W*x` w Dense | clothesolver.jl | brak tymczasowych macierzy |
| `rand!` zamiast `rand()` w Dropout | clothesolver.jl | 0 alokacji na pass |
| `Array{Bool}` → parametryczny typ w Dropout | clothesolver.jl | 147 KB/pass → 0 B |
| im2col + BLAS GEMM w Conv | clothesolver.jl | 13× szybciej vs v1.0 |
| `Array{Tuple}` → `Array{Int32}` w MaxPool | clothesolver.jl | argmax 980 KB → 245 KB |
| Hoisting okna poolingu ponad `@simd h` + `ifelse` | clothesolver.jl | MaxPool1 1.68×, MaxPool2 1.91× |
| MemoryPool (dwie ciągłe tablice) | autodiff.jl | ~84 alokacji zamiast milionów |
| `@generated forward!/backward!` | autodiff.jl | brak dynamicznego dispatchu |
| `batch_size::Int` w `CompiledModel` | clothesolver.jl | 10 s/epoka → 5 s/epoka |
| `@simd`, `@inbounds`, `@fastmath` | wszędzie | wektoryzacja pętli |

### Wyniki końcowe (batch=32, CPU, brak wielowątkowości / CUDA)

```
Przepustowość     : 12 875 img/s
Czas na batch     : 2.49 ms / 32 obrazy
Alokacje/100 passów: 600 (stałe, niezależne od rozmiaru batcha)
Dokładność (3 epoki, η=0.02): 86.64%
```

### Pozostały bottleneck

Po wszystkich optymalizacjach profilery wskazują wsteczną propagację splotu
(Conv2 backward ~70% czasu backward). Jest to fundamentalne ograniczenie:
col2im (rozpraszanie gradientów do `x.grad`) wymaga `C_in × F_W × F_H = 54`
przebiegów po tablicy wejścia. Dalsze przyspieszenie wymagałoby:
- wielowątkowości (OpenBLAS już korzysta z wielu wątków dla dużych macierzy),
- GPU (CUDA przez NVIDIA cuDNN), lub
- zmiany układu danych na `[B, H, W, C]` (NHWC), co pozwoliłoby na jeden
  batched GEMM bez osobnej pętli po próbkach.
