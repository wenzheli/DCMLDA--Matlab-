% dcmlda.m implements a collapsed gibbs sampler for the basic DCMLDA model.
% and itt uses EM to learn alpha and beta parameter vectors
%
% The main advantage of DCMLDA over LDA is that DCMLDA model can capture
% the word burstiness, which means if a term is used once in a document,
% then it is likely to be used again. 
%
% INPUT:
%   wdmat - sparse word-document count matrix 
%   numtopics - desired number of topics
%   burnin - number of epochs before sampling starts
%   samples - number of M-steps of EM
%   samplewait - number of epochs between M-steps
%   alpha - [optional] initial value (default 50/numtopics)
%   beta - [optional] initial value (default 0.01)
%
% OUTPUT:
%   master - resulting master lists, which includes topic assignment for
%               each word
%   alphas - the learned prior distribution for alphas. Simply store all of
%               the learned results
%   betas -  the learnred prior distribution for beta. Simply store all of
%               the learned results
function [master,alphas,betas] = dcmlda(wdmat,numtopics,burnin, samples,samplewait,alpha,beta)
    tic
    if (nargin <= 5) alpha = 50/numtopics; end    
    if (nargin <= 6) beta = 0.01; end
    
    % initialization
    [numdocs numwords] = size(wdmat);       % numwords is vocab size (types)
    totalwords = full(sum(sum(wdmat)));     % totalwords is token count
    
    % expand beta to be a matrix (allowing for asymmetry)
    if size(beta) == 1
        beta = beta*ones(numtopics,numwords);
    end
    
    % runs through the burnin period, and then through the EM period
    % total number of iterations required. 
    iters = burnin + samples*samplewait;
    
    % store the first testing alpha and beta. 
    alphas(1) = alpha;
    betas{1} = beta;

    % master lists of word identities, topic assignments, and corresponding
    % document indicator. 
    maswords = zeros(totalwords,1); 
    mastopics = zeros(totalwords,1);
    masdocs = zeros(totalwords,1);
    
    % initializing the topics & creating the master lists
    currword = 0;
    for d=1:numdocs
        currdoc = wdmat(d,:);    % look at each row in the word-doc count matrix
        for w = find(currdoc)    % pull out each non-zero entry
            wc = currdoc(w);     % count for each word-doc pair
            which = currword + (1:wc);
            maswords(which) = w;
            mastopics(which) = ceil(rand(wc,1).*numtopics);   % initial topic structure
            masdocs(which) = d;
            currword = currword + wc;
        end
    end
    
    % make sure the documents are Gibbs sampled in random order
    reorder = randperm(totalwords);
    maswords = maswords(reorder);
    mastopics = mastopics(reorder);
    masdocs = masdocs(reorder);
    
    % one row for each topic and one column for each document or word. By
    % adding alpha and beta here, we don't need to add them when we do
    % collapsed gibbs sampling. 
    tdmat = full(sparse(mastopics,masdocs,1,numtopics,numdocs)) + alpha;  % topic-document count matrix
    twmat = full(sparse(mastopics,maswords,1,numtopics,numwords)) + beta; % topic-word count matrix

    % gibbs sampling starts.....
    for i = 1:iters			      % for each epoch of the Gibbs sampler
        sumctw = sum(twmat,2);    % number of words in each topic
        if mod(i,10) == 1
            sprintf('Used time %g, beginning iteration %d.',toc,i)
            tic
        end
        % loop through each word, and sample new topic. 
        for w = 1:totalwords               
            currword = maswords(w);		 % identity of current word
            currtopic = mastopics(w);    
            currdoc = masdocs(w);

            % removing the current word from the dataset (z_{-i})
            sumctw(currtopic) = sumctw(currtopic) - 1;
            twmat(currtopic,currword) = twmat(currtopic,currword) - 1;
            tdmat(currtopic,currdoc) = tdmat(currtopic,currdoc) - 1;
            
	        % calculate vector of topic probabilites, and select the new
	        % topic from that multinomial distribution. 
            topicprobs = twmat(:,currword) .* tdmat(:,currdoc) ./ sumctw;
            rndom = rand(1);
            bounds = cumsum(topicprobs);
            location = rndom*bounds(end);
            currtopic = find(bounds >= location,1);
                        
            % re-adding the word to the dataset
            mastopics(w) = currtopic;
            sumctw(currtopic) = sumctw(currtopic) + 1;
            twmat(currtopic,currword) = twmat(currtopic,currword) + 1;
            tdmat(currtopic,currdoc) = tdmat(currtopic,currdoc) + 1;
        end 

        % is it time to do an M-step?
        if (i > burnin && mod(i-burnin,samplewait) == 1)
            theta = full(sparse(mastopics,masdocs,1,numtopics,numdocs)) + 1e-6;
            for d = 1:numdocs
                theta(:,d) = theta(:,d) ./ sum(theta(:,d));
            end
            adisc = checkgrad('dcmldalpha', alpha, 1e-10, theta);
            sprintf('Max relative discrepancy %g in alpha gradient.',adisc)
            % update alpha. Here we assume using symmetric alpha. 
            newalpha = minimize(1,'dcmldalpha',10,theta); 
               
            newbeta = beta;
            for k = 1:numtopics
                which = find(mastopics == k);
                phi = full(sparse(maswords(which),masdocs(which),1,numwords,numdocs)) + 1e-6; % word-doc counts
                for d = 1:numdocs
                    phi(:,d) = phi(:,d) ./ sum(phi(:,d));
                end

                b = ones(numwords,1);
                % update beta..
                newb = minimize(b,'dcmldbeta',10,phi); 
                newbeta(k,:) = newb';
            end
                
            tdmat = tdmat - alpha + newalpha;
            twmat = twmat - beta + newbeta;
            alpha = newalpha;
            beta = newbeta;
            
            % store all the new updates. 
            alphas(end+1) = alpha;
            betas{end+1} = beta;
        end % of M-step
    end % for each epoch
    toc  
    %Combining the master list of words, topics, and documents.
    master = [maswords,mastopics,masdocs];
end
            