function topwords = mktopwords(ntop,phi,wordlist)
% for each topic, find words with highest-probability. 
% Input: 
%    ntop - number of top words need to return
%    phi - learned topic word distribution. 
%    wordlist - it is a cell array, that stores actual string that
%               corresponds to each token index. 
[nwords ntopics] = size(phi);
if (nwords ~= length(wordlist))
    display('bad wordlist');
end

topwords = cell(ntop,ntopics);
for t=1:ntopics
   [sorted which] = sort(-phi(:,t));
   topwords(:,t) = wordlist(which(1:ntop));
end  
