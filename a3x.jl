using Calculus, ForwardDiff, LinearAlgebra

function VarCov1(n, X)
    x = X - ones(n) * ones(n)'X * (1 / n) # deviation scores
    return x'x * (1 / n) # each term in deviation sum of squares by n
end

function VarCov2(μ, σ, ρ)
    M = zeros(size(ρ)[1], size(ρ)[2])
    I = 1
    while I <= size(ρ)[1]
        J = I
        while J <= size(ρ)[2]
            M[I, J] = M[J, I] = σ[I]σ[J]ρ[I, J]
            J = J + 1
        end
        I = I + 1
    end
    return round.(M, digits=5)
end

function newton(x0, f; tol_approx = 10e-9, tol_consec = 10e-6)
    n = 0 # number of iterations
    xnew = x0; xold = x0 - 1
    while ( abs(f(xnew)) > tol_approx || abs(xnew - xold) > tol_consec )
        n = n + 1
        xold = xnew
        xnew = xold - f(xold) / derivative(f, xold)
    end
    return n, xnew
end

function nDimNewton(x0, F; tol_approx = 10e-9, tol_consec = 10e-6)
    t1 = [tol_approx for I in 1:size(F)[1]]; tol_approx = t1
    n = 0
    xnew = x0; xold = x0 - ones(size(x0)[1])
    
    while ( [norm(F[I](xnew)) for I in 1:size(F)[1]] > tol_approx || norm(xnew - xold) > tol_consec )
        n = n + 1
        xold = xnew
        DF = zeros(size(F)[1], size(F)[1])
        for I in 1:size(F)[1]
            DF[I, :] = ForwardDiff.gradient(F[I], xold)
        end
        xnew = xold - DF \ [F[I](xold) for I in 1:size(F)[1]]
    end
    return (n, xnew)
end

function approxNewton(x0, F; tol_approx = 10e-9, tol_consec = 10e-6)
    t1 = [tol_approx for I in 1:size(F)[1]]; tol_approx = t1
    n = 0
    h = tol_consec; xnew = x0; xold = x0 - ones(size(x0)[1])
    
    while ( [norm(F[I](xnew)) for I in 1:size(F)[1]] > tol_approx  || norm(xnew - xold) > tol_consec )
        n = n + 1
        xold = xnew
        ΔF = zeros(size(F)[1], size(F)[1])
        # e = [K == J ? 1 : 0 for K in 1:size(F)[1], J in 1:size(F)[1]]
        for I in 1:size(F)[1]
            for J in 1:size(F)[1]
                e = [K == J ? 1 : 0 for K in 1:size(F)[1]]
                ΔF[I, J] = (F[I](xold .+ h .⋅ e) - F[I](xold)) / h
            end
        end
        xnew = xold - ΔF \ [F[I](xold) for I in 1:size(F)[1]]
    end
    return (n, xnew)
end

function nDimNewtonLagrange(w0, G, M; λ1 = 1.0, λ2 = 1.0, tol_approx = 10e-9, tol_consec = 10e-6)
    n = 0; x0 = [w0; λ1; λ2]
    xnew = x0; xold = x0 - ones(size(x0)[1]);
    
    while ( norm(G(xnew)) > tol_approx  || norm(xnew - xold) > tol_consec )
        n = n + 1
        xold = xnew
        DG = __nablaG(xold, M)
        xnew = xold - DG^-1 * G(xold)
    end
    w0 = xnew[1:end-2]; λ0 = [xnew[end-1], xnew[end]]
    return (n, w0, λ0)
end

function __nablaF(x, M)
    μ = x[1:end-2]
    DF = zeros(size(x)[1], size(x)[1])
    DF[1:end-2, 1:end-2] = 2 * M
    DF[end-1, 1:end-2] = ones(size(μ))'
    DF[end, 1:end-2] = μ'
    DF[1:end-2, end-1] = ones(size(μ))
    DF[1:end-2, end] = μ
    return DF
end

function __nablaG(x, M)
    DG = zeros(size(x)[1], size(x)[1])
    DG[1:end-2, 1:end-2] = 2 * x[end] * M
    DG[end-1, 1:end-2] = ones(size(x)[1]-2)'
    DG[end, 1:end-2] = 2(M * x[1:end-2])'
    DG[1:end-2, end-1] = ones(size(x)[1]-2)
    DG[1:end-2, end] = 2M * x[1:end-2];
    return DG
end