 function [f, df] = dcmldbeta(beta,phi)
% negative log-likelihood and its derivative for the DCMLDA model
% D = number of documents
% phi is a matrix with D columns, each being one multinomial
% beta is a column vector with W rows

[W D] = size(phi);   % size of vocabulary, number of documents

rowsums = sum(log(phi),2);  % sum of each row of phi
totsum = (beta'-1) * rowsums;

f = D*(gammaln(sum(beta)) - sum(gammaln(beta))) + totsum;
f = -f;

if nargout > 1        % calculate gradient wrt beta
    df = D*(psi(sum(beta)) - psi(beta)) + rowsums;
    df = -df;
end
