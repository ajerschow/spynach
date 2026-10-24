% EasySpin reference data for epreasy's esfit (phase 3): fits of analytic
% models with every deterministic algorithm and target function, with
% autoscaling and baselines, global fits with weights, the confidence
% intervals of a noisy exponential, fits of pepper spectra (one and two
% components, initState), and the helpers rescaledata, basecorr, deriv.
% Syntax:
%
%                       esdump_esfit(version_dir)
%
% version_dir is <fixtures_root>/<easyspin version>. One directory per case
% is written below it with data.h5 and meta.json. The data vectors that are
% fitted (including any noise, generated with a fixed rng seed) are stored
% in data.h5 under /input/, so that the Python side fits exactly the same
% numbers; the model is named in meta.json and rebuilt in Python. Cases
% whose directory already exists are skipped, never overwritten.

function esdump_esfit(version_dir)

dump_file=[mfilename('fullpath') '.m'];
x=linspace(300,400,100);
xg=linspace(0,1000,100);
nofuzz=struct('FuzzLevel',0);

% Analytic models (local functions below), one data set
%-------------------------------------------------------------------------------
y=gaussmodel([350 7],x);
rng(1); noise=randn(size(y))*0.5;
cases={};
cases{end+1}={'ef_gauss_simplex',   y, [340 5], [50 20], [], [], struct('Method','simplex fcn')};
cases{end+1}={'ef_gauss_levmar',    y, [340 5], [50 20], [], [], struct('Method','levmar fcn')};
cases{end+1}={'ef_gauss_grid',      y, [340 5], [50 20], [], [], struct('Method','grid fcn','GridSize',21,'randGrid',false)};
cases{end+1}={'ef_gauss_int',       y, [340 5], [50 20], [], [], struct('Method','simplex int')};
cases{end+1}={'ef_gauss_dint',      y, [340 5], [50 20], [], [], struct('Method','simplex dint')};
cases{end+1}={'ef_gauss_diff',      y, [340 5], [50 20], [], [], struct('Method','simplex diff')};
cases{end+1}={'ef_gauss_fft',       y, [340 5], [50 20], [], [], struct('Method','simplex fft')};
cases{end+1}={'ef_gauss_bounds',    y, [340 5], [], [300 1], [400 30], struct('Method','levmar fcn')};
cases{end+1}={'ef_gauss_scaled',    y, [340 5], [50 20], [], [], struct('Method','simplex fcn','ScaleParams',true)};
cases{end+1}={'ef_gauss_noise',     y+noise, [340 5], [50 20], [], [], struct('Method','simplex fcn')};
cases{end+1}={'ef_gauss_noise_lsq', 3*y+noise, [340 5], [50 20], [], [], struct('Method','levmar fcn','AutoScale','lsq')};
cases{end+1}={'ef_gauss_noise_maxabs', 3*y+noise, [340 5], [50 20], [], [], struct('Method','simplex fcn','AutoScale','maxabs')};
cases{end+1}={'ef_gauss_mask',      y+noise, [340 5], [50 20], [], [], struct('Method','simplex fcn','Mask',x<380)};
baselines={0.1, 0.1+0.01*(x-300), 0.1+0.01*(x-300)+0.0002*(x-340).^2};
autoscales={'none','lsq','maxabs'};
for order=0:2
    for a=1:3
        name=sprintf('ef_gauss_baseline%d_%s',order,autoscales{a});
        cases{end+1}={name, 2*y+noise+baselines{order+1}, [340 5], [50 20], [], [], struct('Method','simplex fcn','AutoScale',autoscales{a},'BaseLine',order)}; %#ok<AGROW>
    end
end
for n=1:numel(cases)
    c=cases{n};
    write_case(version_dir,dump_file,c{1},{c{2}},@(p) gaussmodel(p,x),'gaussmodel',c{3},c{4},c{5},c{6},c{7},struct('x',x));
end

% Global fits of two data sets with weights
%-------------------------------------------------------------------------------
data={gaussian(xg,450,100), gaussian(xg,550,100)};
write_case(version_dir,dump_file,'ef_global_weight1',data,@(p) globalmodel(p,xg),'globalmodel',500,[],100,900,struct('Method','simplex','AutoScale','lsq','weight',[1 0]),struct('x',xg));
write_case(version_dir,dump_file,'ef_global_weight2',data,@(p) globalmodel(p,xg),'globalmodel',500,[],100,900,struct('Method','simplex','AutoScale','lsq','weight',[0 1]),struct('x',xg));
data={gaussian(xg,400,100), gaussian(xg,400,10)};
write_case(version_dir,dump_file,'ef_global_levmar',data,@(p) globalmodel2(p,xg),'globalmodel2',[500 50 50],[],[100 1 1],[900 200 200],struct('Method','levmar'),struct('x',xg));
write_case(version_dir,dump_file,'ef_global_baseline',{data{1}+0.2, data{2}+0.1+1e-4*xg},@(p) globalmodel2(p,xg),'globalmodel2',[500 50 50],[],[100 1 1],[900 200 200],struct('Method','simplex','AutoScale','lsq','BaseLine',[0 1]),struct('x',xg));

% Confidence intervals of a noisy exponential (EasySpin's esfit_std test data)
%-------------------------------------------------------------------------------
std_data=textread(fullfile(fileparts(dump_file),'esfit_std_data.txt')); %#ok<DTXTRD>
t=std_data(:,1); yexp=std_data(:,2);
write_case(version_dir,dump_file,'ef_std',{yexp},@(p) expmodel(p,t),'expmodel',[1.5 0.3 0.5],[],[0.1 0.1 -2],[10 0.5 2],struct('Method','simplex fcn','AutoScale','none'),struct('t',t));
write_case(version_dir,dump_file,'ef_std_levmar',{yexp},@(p) expmodel(p,t),'expmodel',[1.5 0.3 0.5],[],[0.1 0.1 -2],[10 0.5 2],struct('Method','levmar fcn','AutoScale','none'),struct('t',t));

% pepper spectra
%-------------------------------------------------------------------------------
Sys=struct('g',[2 2.1],'lw',1);
Exp=struct('mwFreq',9.5,'Range',[300 360],'nPoints',200);
[~,spc]=pepper(Sys,Exp,nofuzz);
rng(1); spcn=addnoise(spc,50,'n');
write_case(version_dir,dump_file,'ef_pepper_simplex',{spcn},@pepper,'pepper',{Sys,Exp,nofuzz},{struct('g',[0.02 0.02])},[],[],struct('Method','simplex'),struct());
write_case(version_dir,dump_file,'ef_pepper_levmar',{spcn},@pepper,'pepper',{Sys,Exp,nofuzz},{struct('g',[0.02 0.02])},[],[],struct('Method','levmar fcn'),struct());
write_case(version_dir,dump_file,'ef_pepper_lw',{spc},@pepper,'pepper',{struct('g',[2 2.1],'lw',1.3),Exp,nofuzz},{struct('lw',0.5)},[],[],struct('Method','simplex'),struct());
Sys1=struct('g',[2 2.1],'lw',1); Sys2=struct('g',[1.98 2.03],'lw',0.7);
Exp4=struct('mwFreq',9.5,'Range',[300 360],'nPoints',400);
[~,spc2]=pepper({Sys1,Sys2},Exp4,nofuzz);
rng(1); spc2n=addnoise(spc2,80,'u');
write_case(version_dir,dump_file,'ef_pepper_twocomp',{spc2n},@pepper,'pepper',{{Sys1,Sys2},Exp4,nofuzz},{{struct('g',[0.02 0.02]),struct('g',[0.02 0.02])}},[],[],struct('Method','levmar fcn'),struct());
SysT=struct('S',1,'g',2.0023,'D',[800 -200],'lw',5,'initState',{{[0.2 0.5 0.3],'zerofield'}});
ExpT=struct('mwFreq',9.75,'Range',[300 400],'Harmonic',0,'nPoints',200);
[~,spcT]=pepper(SysT,ExpT,nofuzz);
rng(1); spcTn=addnoise(spcT,200,'n');
VaryT=struct('D',[0 100]); VaryT.initState={[0 0 0.2]};
write_case(version_dir,dump_file,'ef_pepper_initstate',{spcTn},@pepper,'pepper',{SysT,ExpT,nofuzz},{VaryT},[],[],struct('Method','simplex'),struct());

% Helpers: rescaledata, basecorr, deriv
%-------------------------------------------------------------------------------
name='helpers3';
case_dir=fullfile(version_dir,name);
if isfolder(case_dir)
    fprintf('  skip %s (exists)\n',name);
else
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    yr=gaussian(x,350,10,1)+0.1*gaussian(x,330,5);
    yref=2.5*gaussian(x,352,12,1);
    oracle_write(h5,'/rescale/y',yr); oracle_write(h5,'/rescale/yref',yref);
    [ys,sc]=rescaledata(yr,'maxabs'); oracle_write(h5,'/rescale/maxabs',ys); oracle_write(h5,'/rescale/maxabs_scale',sc);
    [ys,sc]=rescaledata(yr,yref,'maxabs'); oracle_write(h5,'/rescale/maxabs_ref',ys); oracle_write(h5,'/rescale/maxabs_ref_scale',sc);
    [ys,sc]=rescaledata(yr,yref,'lsq'); oracle_write(h5,'/rescale/lsq',ys); oracle_write(h5,'/rescale/lsq_scale',sc);
    [ys,sc]=rescaledata(yr,'int'); oracle_write(h5,'/rescale/int',ys); oracle_write(h5,'/rescale/int_scale',sc);
    [ys,sc]=rescaledata(yr,'dint'); oracle_write(h5,'/rescale/dint',ys); oracle_write(h5,'/rescale/dint_scale',sc);
    region=x<340|x>360;
    [ys,sc]=rescaledata(yr,yref,'lsq',region); oracle_write(h5,'/rescale/lsq_region',ys); oracle_write(h5,'/rescale/lsq_region_scale',sc);
    yb=yr+0.3+0.002*(x-350)+1e-5*(x-350).^2;
    oracle_write(h5,'/basecorr/y',yb);
    for n=0:3
        [yc,bl]=basecorr(yb,1,n); oracle_write(h5,sprintf('/basecorr/corr%d',n),yc); oracle_write(h5,sprintf('/basecorr/base%d',n),bl);
    end
    [yc,bl]=basecorr(yb,1,2,region); oracle_write(h5,'/basecorr/corr2_region',yc); oracle_write(h5,'/basecorr/base2_region',bl);
    Z=yr.'*yref+0.1*(1:100).'/100*ones(1,100);
    oracle_write(h5,'/basecorr/Z',Z);
    [Zc,Zb]=basecorr(Z,[],[1 2]); oracle_write(h5,'/basecorr/Z2d',Zc); oracle_write(h5,'/basecorr/Z2d_base',Zb);
    [Zc,Zb]=basecorr(Z,2,1); oracle_write(h5,'/basecorr/Zrows',Zc); oracle_write(h5,'/basecorr/Zrows_base',Zb);
    [Zc,Zb]=basecorr(Z,1,3); oracle_write(h5,'/basecorr/Zcols',Zc); oracle_write(h5,'/basecorr/Zcols_base',Zb);
    oracle_write(h5,'/deriv/y',yb); oracle_write(h5,'/deriv/d',deriv(yb)); oracle_write(h5,'/deriv/dx',deriv(x,yb));
    es_meta(case_dir,dump_file,struct('x',x,'region',region));
end

end

% One esfit case: fit, store the data and the results
function write_case(version_dir,dump_file,name,data,fcn,fcn_name,p0,vary,lb,ub,FitOpt,extra)
case_dir=fullfile(version_dir,name);
if isfolder(case_dir), fprintf('  skip %s (exists)\n',name); return; end
fprintf('  case %s\n',name); mkdir(case_dir);
h5=fullfile(case_dir,'data.h5');
FitOpt.Verbosity=0;
tic;  % one call, timed for the project page (a fit is long enough for MATLAB's warm-up not to matter)
if isempty(vary)
    result=esfit(data,fcn,p0,lb,ub,FitOpt);
else
    result=esfit(data,fcn,p0,vary,FitOpt);
end
t=toc;
oracle_write(h5,'/result/seconds',t);
for k=1:numel(data)
    oracle_write(h5,sprintf('/input/data%d',k),data{k});
end
oracle_write(h5,'/result/pfit',result.pfit);
oracle_write(h5,'/result/p_start',result.p_start);
oracle_write(h5,'/result/rmsd',result.rmsd);
oracle_write(h5,'/result/ssr',result.ssr);
oracle_write(h5,'/result/scale',result.scale);
oracle_write(h5,'/result/redchisquare',result.redchisquare);
if ~isempty(result.pstd)
    oracle_write(h5,'/result/pstd',result.pstd);
    oracle_write(h5,'/result/ci95',result.ci95);
    oracle_write(h5,'/result/cov',result.cov);
    oracle_write(h5,'/result/corr',result.corr);
end
if iscell(result.fit)
    for k=1:numel(result.fit)
        oracle_write(h5,sprintf('/result/fit%d',k),result.fit{k});
        oracle_write(h5,sprintf('/result/fitraw%d',k),result.fitraw{k});
        oracle_write(h5,sprintf('/result/baseline%d',k),result.baseline{k});
        oracle_write(h5,sprintf('/result/residuals%d',k),result.residuals{k});
    end
else
    oracle_write(h5,'/result/fit1',result.fit);
    oracle_write(h5,'/result/fitraw1',result.fitraw);
    oracle_write(h5,'/result/baseline1',result.baseline);
    oracle_write(h5,'/result/residuals1',result.residuals);
end
oracle_write(h5,'/result/nbest',numel(result.bestfithistory.rmsd));
inputs=struct('fcn',fcn_name,'p0',{p0},'vary',{vary},'lb',lb,'ub',ub,'FitOpt',FitOpt,'extra',extra,'pnames',{result.pnames},'baselinetype',{result.baselinetype});
es_meta(case_dir,dump_file,inputs);
end

% Analytic models
function y=gaussmodel(p,x)
y=5*exp(-(x-p(1)).^2/(2*p(2)^2));
end

function data=globalmodel(p,x)
data{1}=gaussian(x,p(1),100);
data{2}=gaussian(x,p(1),100);
end

function data=globalmodel2(p,x)
data{1}=gaussian(x,p(1),p(2));
data{2}=gaussian(x,p(1),p(3));
end

function y=expmodel(p,t)
y=p(1)*exp(-p(2)*t)+p(3);
end
