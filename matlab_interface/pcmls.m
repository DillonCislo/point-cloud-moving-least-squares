function [Fq, DFq, HFq] = pcmls(Q, P, Fp, varargin)
%PCMLS Evaluate a moving least squares (MLS) interpolant/derivatives of a
%scalar function defined at scattered data points. This is just a wrapper
%to the 'point-cloud-moving-least-squares' Python package.
%
%   INPUT PARAMETERS:
%
%       - Q:    #Q x dim list of query point coordinates
%
%       - P:    #P x dim list of source point coordinates
%
%       - Fp:   #P x 1 vector of source function values
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('PolynomialOrder', m = 2): The maximal degree of the polynomial
%       interpolant
%
%       - ('WeightFunction', weightFunction = 'gaussian'): The type of
%       weight function to use
%
%       - ('WeightMethod', weightMethod = 'knn'): Defines how the weight
%       parameter for each source point is computed
%
%       - ('WeightParam', weightParam = 10): The scalar weight parameter.
%       If weightMethod == 'knn', the weight parameter is the
%       nearest-neighbor index that defines the unique weight parameter for
%       each query point. If weightMethod == 'scalar', then this is the
%       uniform parameter used for all query points
%
%       - ('Vectorized', vectorized = false): Whether to perform a
%       vectorized computation or to compute the MLS interpolant/gradients
%       serially for all query points. Vectorized computations are faster,
%       but very memory hungry
%
%       - ('ComputeGradients', computeGradients = true): Whether to compute
%       MLS gradients
%
%       - ('ComputeHessians', computeHessians = true): Whether to compute
%       MLS Hessians
%
%       - ('Verbose', verbose = false): Whether to produce verbose progress
%       output
%
%   OUTPUT PARAMETERS:
%
%       - Fq:   #Q x 1 query point function values
%
%       - DFq:  #Q x dim query point function gradient values
%
%       - HFq:  dim x dim x #Q query point function Hessian values
%
%   by Dillon Cislo 04/23/2024

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------

validateattributes(Q, {'numeric'}, {'2d', 'finite', 'real'});
numQueries = size(Q,1); dim = size(Q,2);

validateattributes(P, {'numeric'}, {'2d', 'finite', 'real', 'ncols', dim});
numSources = size(P,1);

validateattributes(Fp, {'numeric'}, {'vector', 'finite', 'real'});
if ~(size(Fp,2) == 1), Fp = Fp.'; end
assert(numel(Fp) == numSources, 'Invalid input function size');

% OPTIONAL INPUT PROCESSING -----------------------------------------------

m = 2;
weightFunction = 'gaussian';
weightMethod = 'knn';
weightParam = 10;
vectorized = false;
computeGradients = true;
computeHessians = true;
verbose = false;

allWeightFunctions = {'gaussian', 'wendland'};
allWeightMethods = {'scalar', 'knn'};

for i = 1:numel(varargin)
    
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end
    
    if strcmpi(varargin{i}, 'PolynomialOrder')
        m = varargin{i+1};
        validateattributes(m, {'numeric'}, {'scalar', 'integer', ...
            'finite', 'real', 'positive'});
    end
    
    if strcmpi(varargin{i}, 'WeightFunction')
        weightFunction = lower(varargin{i+1});
        validateattributes(weightFunction, {'char'}, {'vector'});
        assert(ismember(weightFunction, allWeightFunctions), ...
            'Invalid weighting function type');
    end
    
    if strcmpi(varargin{i}, 'WeightMethod')
        weightMethod = lower(varargin{i+1});
        validateattributes(weightMethod, {'char'}, {'vector'});
        assert(ismember(weightMethod, allWeightMethods), ...
            'Invalid weight function parameter estimation method');
    end
    
    if strcmpi(varargin{i}, 'WeightParam')
        weightParam = varargin{i+1};
        validateattributes(weightParam, {'numeric'}, {'scalar', ...
            'finite', 'real', 'positive'});
    end
    
    if strcmpi(varargin{i}, 'Vectorized')
        vectorized = varargin{i+1};
        validateattributes(vectorized, {'logical'}, {'scalar'});
    end
    
    if strcmpi(varargin{i}, 'ComputeGradients')
        computeGradients = varargin{i+1};
        validateattributes(computeGradients, {'logical'}, {'scalar'});
    end
    
    if strcmpi(varargin{i}, 'computeHessians')
        computeHessians = varargin{i+1};
        validateattributes(computeHessians, {'logical'}, {'scalar'});
    end
    
    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'});
    end

end

pcmls_kwargs = struct( 'm', m, 'weight_function', weightFunction, ...
    'weight_method', weightMethod, 'weight_param', weightParam, ...
    'vectorized', vectorized, 'verbose', verbose, ...
    'compute_gradients', computeGradients,...
    'compute_hessians', computeHessians);

%--------------------------------------------------------------------------
% EXECUTE 'pcmls' COMMAND
%--------------------------------------------------------------------------

% Determine if the task is being run in parallel
task = getCurrentTask();
id = get(task, 'ID');

% Set up I/O directory structure
[pcmlsDir, ~, ~] = fileparts(mfilename("fullpath"));
pcmlsFile = fullfile(pcmlsDir, 'run_pcmls_from_matlab.py');
inputFile = fullfile(pcmlsDir, sprintf('input_file_%d.h5', id));
outputFile = fullfile(pcmlsDir, sprintf('output_file_%d.h5', id));
configFile = fullfile(pcmlsDir, sprintf('config_file_%d.json', id));

if exist(inputFile, 'file'), delete(inputFile); end
if exist(outputFile, 'file'), delete(outputFile); end
if exist(configFile, 'file'), delete(configFile); end

% Write input data to file. Notice the weird transpose operations. I guess
% MATLAB and h5py use different read/write conventions
h5create(inputFile, '/q', [dim, numQueries]);
h5create(inputFile, '/p', [dim, numSources]);
h5create(inputFile, '/fp', [numSources, 1]);
h5write(inputFile, '/q', Q.');
h5write(inputFile, '/p', P.');
h5write(inputFile, '/fp', Fp);

% Write configuration metadata to file
config = struct('pcmls_kwargs', pcmls_kwargs);
configJSON = jsonencode(config);
fid = fopen(configFile, 'w');
if (fid == -1), error('Cannot create JSON file'); end
fwrite(fid, configJSON, 'char');
fclose(fid);

pcmlsCommand = ['!python ' pcmlsFile ' ' ...
    inputFile ' ' outputFile ' ' configFile ];
eval(pcmlsCommand)

% Format output
Fq = h5read(outputFile, '/fq');
if computeGradients
    DFq = h5read(outputFile, '/grad_fq').';
else
    DFq = [];
end
if computeHessians
    HFq = h5read(outputFile, '/hess_fq');
else
    HFq = [];
end

% Delete temporary files
delete(inputFile)
delete(outputFile)
delete(configFile)

end

