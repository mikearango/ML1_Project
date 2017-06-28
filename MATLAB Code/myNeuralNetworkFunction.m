function [y1] = myNeuralNetworkFunction(x1)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Generated by Neural Network Toolbox function genFunction, 20-Jun-2017 19:38:06.
%
% [y1] = myNeuralNetworkFunction(x1) takes these arguments:
%   x = 9xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [1.51115;10.73;0;0.29;69.81;0;5.43;0;0];
x1_step1.gain = [87.7963125548726;0.300751879699248;0.44543429844098;0.623052959501558;0.357142857142857;0.322061191626409;0.185873605947955;0.634920634920635;3.92156862745098];
x1_step1.ymin = -1;

% Layer 1
b1 = [1.809142781691417623;-1.4063635089674004774;1.0686583813980536917;-0.79035905556951557838;0.20151187683669147255;-0.35674347555555402378;0.67819555404889930728;0.91221721943004496769;-1.4108500166130757414;-1.862637254465207004];
IW1_1 = [-1.2467148287467630929 0.10062336812859574986 -0.038913296907458788665 -0.48928792651518848711 1.0496281706583707738 -0.12372660982369204641 -0.029107905186454053476 -0.58565926239672294606 -0.079827581231775329718;0.5234379422126788084 0.8114702492188569094 0.76644650023578464459 0.21016401017969596476 0.78846085176267810546 -0.58726224975566831965 0.64655026870987786225 0.56317797604665675859 -0.051165004132387448266;-0.64551469559343843674 -0.28774483075667545151 -1.0293002502167638568 1.0049991220330658503 0.44608945842263397763 -0.42548329292540831847 0.38961147941152118879 1.0686882435743507713 0.19610280616380384999;0.77938286306250736946 0.32986990742269450827 -1.1284329884796922006 0.0067405511374388268203 -0.28281674581302190807 -0.80199459173088283315 -0.47117828209438106235 0.23114507730219827075 -0.61803514125057312789;-0.59365730747003764289 -0.74435374362735540199 0.17995458838418484926 -0.38808066930102913528 0.47620320066991922436 -0.78689003734533968792 0.75451283661082602094 -0.4938984389753532378 0.7256706340694391022;0.061128164424969204382 0.58501472672330634417 -0.59386155850357780217 -1.1205554158508650442 0.14535294314806504468 0.86673774526654367989 0.14784118373346150088 0.16551178120282916684 0.71600231267380431976;0.81455194779264794569 -0.71979486137552850522 0.11523271628819024837 0.18684701134294443015 0.47464531236876633669 -0.7219526734733815454 0.81400907138867240054 -0.58777116139285978669 0.20690888874016369336;0.19990254825544417905 -0.17449592447532713546 0.49241205045351638114 -1.3997413849082571691 0.19157375744625546043 0.87942575478121165489 0.80306592728535375336 -0.54877916180998753681 0.31921575097741522464;-0.48331073466376495151 0.30837879456863054317 -0.087897736641915497069 1.0352480829736796508 0.90429318834101457547 0.68940790025149600861 0.171950252213648902 -0.52533734303916057584 -0.62758709445482097067;-0.25840361810289497191 -0.54797398456741996942 -0.86667573339583392489 -0.089696054645002604166 0.80504189492095623581 -0.55724316396357975734 -0.12694247461271687683 0.61390533681719294812 0.82757548851582241056];

% Layer 2
b2 = [-0.28585915364907860114;0.2535441671907910921];
LW2_1 = [0.11362026492199901739 0.45650130309165931619 1.5742932491296741659 -0.087490009873296539777 -0.94125406280875523191 -0.069817229359881383122 -0.52373774876450029936 -1.1297104854094055515 0.4438783071257919044 -0.28440891269932322416;-0.24827431393728097153 0.48853524087706717838 -0.40792148586383369935 -1.3382610260282046255 -0.58599272304629357322 0.27309197995528539105 -0.62602295870469826244 0.45334150322926630716 -0.8033334093870850845 -0.042264923634496721905];

% ===== SIMULATION ========

% Dimensions
Q = size(x1,2); % samples

% Input 1
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = softmax_apply(repmat(b2,1,Q) + LW2_1*a1);

% Output 1
y1 = a2;
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Competitive Soft Transfer Function
function a = softmax_apply(n,~)
if isa(n,'gpuArray')
    a = iSoftmaxApplyGPU(n);
else
    a = iSoftmaxApplyCPU(n);
end
end
function a = iSoftmaxApplyCPU(n)
nmax = max(n,[],1);
n = bsxfun(@minus,n,nmax);
numerator = exp(n);
denominator = sum(numerator,1);
denominator(denominator == 0) = 1;
a = bsxfun(@rdivide,numerator,denominator);
end
function a = iSoftmaxApplyGPU(n)
nmax = max(n,[],1);
numerator = arrayfun(@iSoftmaxApplyGPUHelper1,n,nmax);
denominator = sum(numerator,1);
a = arrayfun(@iSoftmaxApplyGPUHelper2,numerator,denominator);
end
function numerator = iSoftmaxApplyGPUHelper1(n,nmax)
numerator = exp(n - nmax);
end
function a = iSoftmaxApplyGPUHelper2(numerator,denominator)
if (denominator == 0)
    a = numerator;
else
    a = numerator ./ denominator;
end
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end