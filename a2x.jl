using DataStructures

function Midpoint(a, b, n, f_int)::Float64
    h = (b - a) / n
    I_midpoint = 0
    for i in 1:n
        I_midpoint += f_int(a + (i - 0.5) ⋅ h)
    end
    return h ⋅ I_midpoint
end

function Simpson(a, b, n, f_int)::Float64
    h = (b - a) / n
    I_simpson = f_int(a) / 6 + f_int(b) / 6
    for i in 1:(n - 1)
        I_simpson += f_int(a + i ⋅ h) / 3
    end
    for i in 1:n
        I_simpson += 2 ⋅ f_int(a + (i - 0.5) ⋅ h) / 3
    end
    return h ⋅ I_simpson
end

function I_numerical(a, b, f_int, method, tol)
    no_intervals = []
    n = 4; I_old = method(a, b, n, f_int)
    append!(no_intervals, n)
    n = 2n; I_new = method(a, b, n, f_int)
    append!(no_intervals, n)
    I_approx = [I_old]
    while(abs(I_new - I_old) > tol)
        I_old = I_new
        n = 2n
        append!(no_intervals, n)
        append!(I_approx, I_new)
        I_new = method(a, b, n, f_int)
    end
    append!(I_approx, I_new)
    return no_intervals, I_approx
end

function cum_dist_normal(t)
    z = abs(t)
    y = 1 / (1 + 0.2316419z)
    a1 = 0.319381530; a2 = -0.356563782; a3 = 1.781477937; a4 = -1.821255978; a5 = 1.330274429
    
    m = 1 - exp(-t^2 / 2) * (a1 * y + a2 * y^2 + a3 * y^3 + a4 * y^4 + a5 * y^5) / √(2 * π)
    if t > 0
        nn = m
    else
        nn = 1 - m
    end
    return nn
end
    
function black_scholes_approx(t, S, K, T, σ, r, q)
    d_1 = d1(t, S, K, T, σ, r, q); d_2 = d2(t, S, K, T, σ, r, q)
    C = S * exp(-q * (T - t)) * cum_dist_normal(d_1) - K * exp(-r * (T - t)) * cum_dist_normal(d_2)
    P = K * exp(-r * (T - t)) * cum_dist_normal(-d_2) - S * exp(-q * (T - t)) * cum_dist_normal(-d_1)
    return [d_1, d_2], [C, P]
end

using QuadGK
approx_error = function(f, M, T, S)
    F = quadgk(f, 0, 2, rtol=1e-7)[1]
    return (abs(int(F - M)), abs(int(F - T)), abs(int(F - S)))
end

d1x(S, K, T, x, r, q) = (log(S / K) + (r - q + x^2 / 2) * T) / (x * sqrt(T))
d2x(S, K, T, x, r, q) = (log(S / K) + (r - q - x^2 / 2) * T) / (x * sqrt(T))

get_periods = function(m, p)
    n = Int64(ceil(m / p))
    t = Stack{Int64}()
    push!(t, m)
    I = n - 1
    for I in 1:n-1
        push!(t, m - Int64(ceil(p * I)))
        I = I - 1
    end
    return t
end

function bond_PrDuCv(T, n, t_cash_flow, v_cash_flow, y)
    B = 0; D = 0; C = 0
    disc = zeros(n)
    for i in 1:n
        disc[i] = exp(-t_cash_flow[i]y)
        B = B + v_cash_flow[i]disc[i]
        D = D + t_cash_flow[i]v_cash_flow[i]disc[i]
        C = C + t_cash_flow[i]^2 * v_cash_flow[i] * disc[i]
    end
    D = D / B; C = C / B
    return B, D, C
end

function get_cash_flow(n, m, T, F, C; M = 12)
    t_cash_flow = zeros(n)
    v_cash_flow = zeros(n)
    I = n; t = T
    while (I > 0)
        t_cash_flow[I] = t / M
        t = T - (M / m) ⋅ (n - I + 1) # (n - (I - 1))
        I -= 1
    end
    I = 1
    while (I < n)
        v_cash_flow[I] = F ⋅ (C / m)
        I += 1
    end
    v_cash_flow[n] = F ⋅ (1 + C / m)
    return t_cash_flow, v_cash_flow
end

function bond_price_zero_rate(n, t_cash_flow, v_cash_flow, r_zero; z = 0)
    disc = zeros(n)
    B = 0
    for i in 1:n
        disc[i] = exp(-t_cash_flow[i] ⋅ (r_zero(t_cash_flow[i]) + z))
        B = B + v_cash_flow[i] ⋅ disc[i]
    end
    return disc, B
end

function bond_price_inst_rate(n, t_cash_flow, v_cash_flow, r_inst, tol; z = 0)
    B = 0
    I_num = zeros(n)
    disc = zeros(n)
    for i in 1:n
        no_intervals, I_approx = I_numerical(0, t_cash_flow[i], r_inst, Simpson, tol[i])
        I_num[i] = I_approx[end]
        disc[i] = exp(-I_num[i])
        B = B + v_cash_flow[i] ⋅ disc[i]
    end
    return disc, B
end