function [f, df] = dcmldalpha(alpha,theta)
% negative log-likelihood and its derivative for the DCMLDA model
% assumes alpha is scalar
% theta is a matrix with D columns, each being one multinomial

[K D] = size(theta);      % number of topics, documents
totsum = sum(sum(log(theta)));

% calculating overall log-likelihood. 
f = D*(gammaln(K*alpha) - K*gammaln(alpha)) + (alpha-1)*totsum;
f = -f;

if nargout > 1
    df = D*K*(psi(K*alpha) - psi(alpha)) + totsum;
    df = -df;
end
