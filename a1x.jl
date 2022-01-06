function d1(t, S, K, T, σ, r, q)
    return ((log(S / K) + (r - q + σ^2 / 2)) * (T - t)) / (σ * sqrt(T - t))
end

function d2(t, S, K, T, σ, r, q)
    return d1 - σ * sqrt(T - t)
end