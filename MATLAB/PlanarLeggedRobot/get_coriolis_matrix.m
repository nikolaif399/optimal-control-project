function C = get_coriolis_matrix(M,q,qdot)
% get_coriolis
% Computes coriolis matrix from mass matrix
    
    n = size(M,1);
    diffM = reshape(jacobian(reshape(M, [n*n, 1]), q), [n n n]);
    
    C = sym(zeros(size(M)));
    for i = 1:n
        for j = 1:n
            for k = 1:n
                C(i,j) = C(i,j) + 0.5*(diffM(i,j,k) + diffM(i,k,j) - diffM(k,j,i)) * qdot(k);
            end
        end
    end
    C = simplify(C);
end

