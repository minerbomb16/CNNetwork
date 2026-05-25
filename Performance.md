# LOG
## Wersja 1.0
Mikołaj - 175s  
Michał - 247s  
Alokacje - 91.681 GiB  

---

## Wersja 2.0
### Zmiany
1. (197s, 681MiB) Zmiana w warstwie Dense, wykorzystanie funkcji mul!, która pozwala na unikniecie tworzenia tymaczasowych macierzy
``` julia
    # Przed
    function primal!(y::GraphNode{:dense, 3})
        W, b, x = y.args
        y.data .= W.data * x.data .+ b.data
    end

    function adjoint!(y::GraphNode{:dense, 3})
        W, b, x = y.args
        W.grad .+= y.grad * x.data'
        b.grad .+= y.grad
        x.grad .+= W.data' * y.grad
    end

    #Po
    function primal!(y::GraphNode{:dense, 3})
        W, b, x = y.args
        mul!(y.data, W.data, x.data)
        y.data .+= b.data
    end

    function adjoint!(y::GraphNode{:dense, 3})
        W, b, x = y.args
        mul!(W.grad, y.grad, x.data', 1, 1) 
        b.grad .+= y.grad
        mul!(x.grad, W.data', y.grad, 1, 1) 
    end
```
2. (201s, 525MiB) W Dropout dodanie bufora z pamięcią zmiennoprzecinkową, by go nadpisywać, a nie inicjować za każdym razem rand() -> rand!()
``` julia
    # Przed
    function (layer::Dropout)(x::GraphNode)
        return GraphNode(:dropout, (x,), zeros(size(x.data)), 
                        params=Dict(:p => layer.p, :mask => zeros(Bool, size(x.data))))
    end

    function primal!(y::GraphNode{:dropout, 1})
        # ...
            p = y.params[:p]
            mask = rand(size(x.data)...) .> p
            y.params[:mask] .= mask
            y.data .= (x.data .* mask) ./ (1.0 - p)
        # ...
    end

    #Po
    function (layer::Dropout)(x::GraphNode)
        return GraphNode(:dropout, (x,), zeros(size(x.data)), 
                        params=Dict(:p => layer.p,
                                    :mask => zeros(Bool, size(x.data)),
                                    :rand_buffer => zeros(size(x.data))))
    end

    function primal!(y::GraphNode{:dropout, 1})
        # ...
        p = y.params[:p]
        mask = y.params[:mask]
        rand_buffer = y.params[:rand_buffer]
        rand!(rand_buffer)
        mask .= rand_buffer .> p
        y.data .= (x.data .* mask) ./ (1.0 - p)
        # ...
    end
```

3. (198s, 394MiB) Usunięcie Any i przypisanie typów. Dodanie nadrzędnych funkcji typu _work oraz typów (AbstractNode)

4. (208s, 357MiB) Zmiana reverse(order) na Iterators.reverse(order) (nie alokuje)

5. Cofnięte, nie przyniosło efektu (205s, 357MiB) Wyrzucenie dodawania biasu do gradientu dla filtru w Conv przed for (SIMD)

6. (201s, 355MiB) Dodanie brakujących typów do Conv i Naxpool

7. (203s, 330MiB) Zmina Float64 na Flaot32

### Performance
Mikołaj -  161s  
Michał - 203s  
Alokacje - 330MiB  

---
## Wersja 3.0
Całkowita zmiana architektury, teraz oparta jest na MemoryPool i generowaniu kodu przy inicjalizacji łańcucha (warstw sieci). Całość nowej
architektury opisana jest w DOKUMENTACJA.md


### Performance
Mikołaj - 30.76s  
Michał - 24.4s  
Aloakcje - 583.404 MiB  

---
## Wersja 4.0
1. (21s, 17MiB) Dodanie @view w pętli testującej

``` julia
batch_idx = @view indices[b_start:b_end]

zero_w_grad!(model.pool)
zero_a_grad!(model.pool)

for (i, idx) in enumerate(batch_idx)
    copyto!(@view(input_node.data[:, :, :, i]), @view(X[:, :, :, idx]))
    copyto!(@view(target_node.data[:, i]), @view(Y[:, idx]))
end
```

2. (19.39s, 6.742MiB) W test.ipynb zastąpienie wysokopoziomowego wycinania widoków (copyto! z @view) w pętlach ładujących dane bezpośrednim, niskopoziomowym przepisywaniem bajtów za pomocą indeksowania liniowego na płaskich buforach parent(). Ustawienie na 1 wątek

### Performance
Mikołaj - xs  
Michał - 19.38s  
Aloakcje - 6.709 MiB 