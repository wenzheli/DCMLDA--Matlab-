% load the sparse text data. 
load nips.mat;
% train the DCMLDA model using nips data set. We choose number topic as 20, 
% start sampling after finishing first 100 iterations. And we sample 10
% times, where each sampling period requires 5 iteration of gibbs sampling.
[master,alphas,betas]=dcmlda(nips,20,100, 10,5);
% print the top words we learned from beta. 
topwords=mktopwords(20,betas{size(betas,2)}', wrds)

