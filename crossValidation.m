function [trainsets, testsets] = crossValidation(indices, ts_size, n_perm)
% -------------------------------------------------------------------------
% generation of index sets for cross validation 
% 
% INPUT:
%    indices  - 1*n matrix containing indices, e.g. [1 2 3 4 5]
%    ts_size  - size of test set 
%    n_perm  - number of training and testsets to be generated
% OUTPUT:
%    trainsets   - numperm*k matrix containing numperm trainsets 
%    testsets    - numperm*ts_size matrix containing numperm testsets
%
% Author: Ulrich Hoffmann - EPFL, 2004
% Copyright: Ulrich Hoffmann - EPFL
% -------------------------------------------------------------------------

% initialize variables
tr_size = size(indices,2) - ts_size;
trainsets = zeros(n_perm, tr_size);
testsets = zeros(n_perm, ts_size);

% generate train and test sets
for p = 1:n_perm
    perm = randperm(ts_size + tr_size);
    for i = 1:tr_size;
        trainsets(p,i) = indices(perm(i));
    end
    for i = 1:ts_size;
        testsets(p,i) = indices(perm(i+tr_size));
    end
end
    
