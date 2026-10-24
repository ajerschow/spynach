% EasySpin reference data for epreasy's pulse-EPR simulator saffron (phase 6,
% fast engine): predefined ESEEM/HYSCORE/Mims-ENDOR experiments, custom
% sequences with ideal and real pulses, orientation selection, relaxation
% decays, the product rule, time-domain evolution, crystals, components,
% and the helpers larmorfrq, apowin, fdaxis, exponfit and sf_pathways. Syntax:
%
%                       esdump_saffron(version_dir)
%
% version_dir is <fixtures_root>/<easyspin version>. One directory per case
% is written below it with data.h5 and meta.json; the inputs of each case
% are recorded in meta.json, from which tests/epreasy/test_parity_saffron.py
% rebuilds the case. Cases whose directory already exists are skipped.

function esdump_saffron(version_dir)

dump_file=[mfilename('fullpath') '.m'];

cases={};
cc=@(name,Sys,Exp,Opt) struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt);
n14=struct('Nucs','14N','A',[-1 -1 2]*0.5+0.8,'Q',1);
h1=struct('Nucs','1H','A',[1 1 5]);
h1ax=struct('Nucs','1H','A',[2 7]);
g=struct('GridSize',19);
e2=struct('Field',330,'Sequence','2pESEEM','dt',0.01,'nPoints',200,'tau',0.08);
e3=struct('Field',330,'Sequence','3pESEEM','dt',0.01,'nPoints',200,'tau',0.08);
e4=struct('Field',330,'Sequence','4pESEEM','dt',0.01,'nPoints',200,'tau',0.08);
eh=struct('Field',330,'Sequence','HYSCORE','dt',0.02,'nPoints',48,'tau',0.08);
% predefined experiments
cases{end+1}=cc('pu_2p_14n',n14,e2,g);
cases{end+1}=cc('pu_2p_1h',h1,e2,g);
cases{end+1}=cc('pu_2p_tau0',h1ax,withfield(e2,'tau',0),g);
cases{end+1}=cc('pu_3p_14n',n14,e3,g);
cases{end+1}=cc('pu_3p_1h_T',h1,withfield(e3,'T',0.2),g);
cases{end+1}=cc('pu_4p_14n',n14,e4,g);
cases{end+1}=cc('pu_4p_1h_T',h1ax,withfield(e4,'T',0.3),g);
cases{end+1}=cc('pu_hyscore_14n',n14,eh,g);
cases{end+1}=cc('pu_hyscore_1h_t1t2',h1ax,withfield(withfield(eh,'t1',0.1),'t2',0.05),g);
cases{end+1}=cc('pu_hyscore_expand',n14,eh,withfield(g,'Expand',1));
cases{end+1}=cc('pu_2p_expand',n14,e2,withfield(g,'Expand',2));
cases{end+1}=cc('pu_2p_window',n14,e2,withfield(withfield(g,'Window','han+'),'ZeroFillFactor',4));
cases{end+1}=cc('pu_2p_default_opts',n14,e2,struct());
cases{end+1}=cc('pu_2p_dt_only',n14,rmfield(e2,'nPoints'),g);
% several nuclei, product rule, Opt.Nuclei
two=struct('Nucs','14N,1H','A',[-1 -1 2]*0.5+0.8,'Q',1);
two.A=[two.A; 1 1 4];
two.Q=[1 0; 0 0];
cases{end+1}=cc('pu_3p_two',two,e3,g);
cases{end+1}=cc('pu_3p_two_product',two,e3,withfield(g,'ProductRule',1));
cases{end+1}=cc('pu_hyscore_two_product',two,eh,withfield(g,'ProductRule',1));
cases{end+1}=cc('pu_2p_two_nuclei2',two,e2,withfield(g,'Nuclei',2));
cases{end+1}=cc('pu_3p_fullA',withfield(withfield(struct('Nucs','1H'),'A',[2 5 10]),'AFrame',[20 60 0]*pi/180),withfield(e3,'tau',0.001),withfield(g,'GridSymmetry','Ci'));
Ra=erot([20 60 0]*pi/180);
cases{end+1}=cc('pu_3p_fullA_matrix',struct('Nucs','1H','A',Ra.'*diag([2 5 10])*Ra),withfield(e3,'tau',0.001),withfield(g,'GridSymmetry','Ci'));
cases{end+1}=cc('pu_3p_fullQ',struct('Nucs','14N','A',[1 1 3],'Q',Ra.'*diag([-0.3 -0.7 1])*Ra),e3,g);
cases{end+1}=cc('pu_3p_qframe',struct('Nucs','14N','A',[1 1 3],'Q',[-0.3 -0.7 1],'QFrame',[20 60 0]*pi/180),e3,g);
% time domain
cases{end+1}=cc('pu_2p_td',n14,e2,withfield(g,'TimeDomain',1));
cases{end+1}=cc('pu_3p_td',n14,e3,withfield(g,'TimeDomain',1));
cases{end+1}=cc('pu_hyscore_td',n14,withfield(eh,'nPoints',24),withfield(g,'TimeDomain',1));
cases{end+1}=cc('pu_3p_two_product_td',two,e3,withfield(withfield(g,'ProductRule',1),'TimeDomain',1));
% orientation selection
osys=struct('Nucs','1H','A',[1 6],'g',[2 2.2]);
oexp=struct('Sequence','2pESEEM','mwFreq',9.5,'dt',0.015,'nPoints',128,'ExciteWidth',200,'tau',0.001,'Field',350);
cases{end+1}=cc('pu_orisel_2p',osys,oexp,withfield(g,'GridSize',31));
cases{end+1}=cc('pu_orisel_hstrain',withfield(osys,'HStrain',[50 50 200]),oexp,withfield(g,'GridSize',31));
cases{end+1}=cc('pu_orisel_threshold',osys,oexp,withfield(withfield(g,'GridSize',31),'OriThreshold',0.2));
tr=struct('S',1,'D',[300 20],'Nucs','1H','A',[2 5],'g',2.01);
texp=struct('Sequence','3pESEEM','mwFreq',9.7,'dt',0.01,'nPoints',150,'ExciteWidth',150,'tau',0.1,'Field',340);
cases{end+1}=cc('pu_triplet_3p',tr,texp,g);
cases{end+1}=cc('pu_triplet_transitions',tr,texp,withfield(g,'Transitions',[1 2]));
cases{end+1}=cc('pu_triplet_hyscore',tr,withfield(rmfield(withfield(texp,'Sequence','HYSCORE'),'nPoints'),'nPoints',32),g);
cases{end+1}=cc('pu_triplet_noorisel',rmfield(tr,'g'),rmfield(rmfield(texp,'mwFreq'),'ExciteWidth'),g);
cases{end+1}=cc('pu_s32_2p',struct('S',3/2,'D',200,'Nucs','14N','A',[1 1 3],'Q',0.8),struct('Sequence','2pESEEM','dt',0.01,'nPoints',150,'tau',0.1,'Field',340),g);
cases{end+1}=cc('pu_cu_2p',struct('Nucs','63Cu','A',[50 50 500],'g',[2.05 2.05 2.2],'lwEndor',0.5),struct('Sequence','2pESEEM','dt',0.01,'nPoints',150,'tau',0.1,'Field',340,'mwFreq',9.7,'ExciteWidth',100),g);
% relaxation decays
dsys=struct('S',1/2,'Nucs','1H','A_',[5 2],'T1',2,'T2',1);
cases{end+1}=cc('pu_decay_3p',dsys,withfield(e3,'tau',0.1),g);
cases{end+1}=cc('pu_decay_2p',dsys,withfield(e2,'tau',0.1),g);
cases{end+1}=cc('pu_decay_hyscore',dsys,withfield(eh,'tau',0.1),g);
cases{end+1}=cc('pu_decay_4p',dsys,withfield(e4,'tau',0.1),g);
% crystals
cexp=withfield(withfield(eh,'MolFrame',[pi/3 pi/6 pi/4]),'SampleFrame',[pi/9 pi/5 0]);
cases{end+1}=cc('pu_crystal_hyscore',h1ax,cexp,struct());
cases{end+1}=cc('pu_crystal_2p_multi',h1ax,withfield(withfield(e2,'MolFrame',[0.3 0.5 0.7]),'SampleFrame',[0 0 0; 0 pi/4 0; pi/9 pi/5 1.1]),struct());
cases{end+1}=cc('pu_crystal_3p_two',two,withfield(e3,'SampleFrame',[0 pi/4 0]),struct());
cases{end+1}=cc('pu_crystal_3p_two_product',two,withfield(e3,'SampleFrame',[0 pi/4 0]),struct('ProductRule',1));
cases{end+1}=cc('pu_crystal_triplet',tr,withfield(withfield(texp,'MolFrame',[0.3 0.5 0.7]),'SampleFrame',[0.1 0.9 0.2]),struct());
% Mims ENDOR
mexp=struct('Sequence','MimsENDOR','Field',325,'tau',0.1,'Range',larmorfrq('1H',325)+[-1 1]*10,'nPoints',401);
msys=withfield(h1,'lwEndor',0.1);
cases{end+1}=cc('pu_mims_1h',msys,mexp,g);
cases{end+1}=cc('pu_mims_1h_m1',msys,mexp,withfield(g,'EndorMethod',1));
cases{end+1}=cc('pu_mims_1h_m2',msys,mexp,withfield(g,'EndorMethod',2));
cases{end+1}=cc('pu_mims_1h_T',msys,withfield(mexp,'T',0.3),g);
cases{end+1}=cc('pu_mims_1h_tprf',msys,withfield(mexp,'tprf',5),withfield(g,'EndorMethod',2));
cases{end+1}=cc('pu_mims_14n',struct('Nucs','14N','A',[1 1 3],'Q',0.8,'lwEndor',0.1),withfield(withfield(mexp,'Range',[0 12]),'Field',340),g);
cases{end+1}=cc('pu_mims_14n_m2',struct('Nucs','14N','A',[1 1 3],'Q',0.8,'lwEndor',0.1),withfield(withfield(mexp,'Range',[0 12]),'Field',340),withfield(g,'EndorMethod',2));
cases{end+1}=cc('pu_mims_two',withfield(two,'lwEndor',0.1),withfield(withfield(mexp,'Range',[0 20]),'Field',340),g);
cases{end+1}=cc('pu_mims_two_product',withfield(two,'lwEndor',0.1),withfield(withfield(mexp,'Range',[0 20]),'Field',340),withfield(g,'ProductRule',1));
cases{end+1}=cc('pu_mims_crystal',msys,withfield(mexp,'SampleFrame',[0.2 0.8 0]),struct());
cases{end+1}=cc('pu_mims_orisel',withfield(osys,'lwEndor',0.2),struct('Sequence','MimsENDOR','Field',350,'mwFreq',9.5,'ExciteWidth',200,'tau',0.1,'Range',[5 25],'nPoints',301),withfield(g,'GridSize',31));
% custom sequences
p90=struct('Flip',pi/2); p180=struct('Flip',pi);
cases{end+1}=cc('pu_custom_3p',n14,struct('Field',330,'Sequence',{{p90 0.08 p90 0.0 p90 0.08}},'nPoints',200,'Dim1',{{'d2' 0.01}}),g);
cases{end+1}=cc('pu_custom_2p',n14,struct('Field',330,'Sequence',{{p90 0.08 p180 0.08}},'nPoints',200,'Dim1',{{'d1,d2' 0.01}}),g);
cases{end+1}=cc('pu_custom_4p_2d',n14,struct('Field',330,'Sequence',{{p90 0.08 p90 0 p180 0 p90 0.08}},'nPoints',[40 40],'Dim1',{{'d2' 0.02}},'Dim2',{{'d3' 0.02}}),g);
cases{end+1}=cc('pu_custom_4p_1d',n14,struct('Field',330,'Sequence',{{p90 0.08 p90 0 p180 0 p90 0.08}},'nPoints',200,'Dim1',{{'d2,d3' 0.01}}),g);
cases{end+1}=cc('pu_custom_filter',n14,struct('Field',330,'Sequence',{{p90 0.08 p90 0.0 p90 0.08}},'nPoints',200,'Dim1',{{'d2' 0.01}},'Filter','+a-'),g);
cases{end+1}=cc('pu_custom_pathways',n14,struct('Field',330,'Sequence',{{p90 0.08 p90 0.0 p90 0.08}},'nPoints',200,'Dim1',{{'d2' 0.01}},'Pathways','+b-'),g);
p90y=struct('Flip',pi/2,'Phase',pi/2); p60=struct('Flip',pi/3);
cases{end+1}=cc('pu_custom_phase',n14,struct('Field',330,'Sequence',{{p90 0.08 p90y 0.0 p60 0.08}},'nPoints',200,'Dim1',{{'d2' 0.01}}),g);
cases{end+1}=cc('pu_custom_decrement',n14,struct('Field',330,'Sequence',{{p90 0.08 p90 0.5 p90 0.08}},'nPoints',40,'Dim1',{{'d2' -0.01}}),g);
cases{end+1}=cc('pu_custom_orisel',osys,struct('Field',350,'mwFreq',9.5,'ExciteWidth',200,'Sequence',{{p90 0.001 p180 0.001}},'nPoints',128,'Dim1',{{'d1,d2' 0.015}}),withfield(g,'GridSize',31));
cases{end+1}=cc('pu_custom_td',n14,struct('Field',330,'Sequence',{{p90 0.08 p90 0.0 p90 0.08}},'nPoints',200,'Dim1',{{'d2' 0.01}}),withfield(g,'TimeDomain',1));
cases{end+1}=cc('pu_custom_hyscore_td',n14,struct('Field',330,'Sequence',{{p90 0.08 p90 0 p180 0 p90 0.08}},'nPoints',[16 16],'Dim1',{{'d2' 0.02}},'Dim2',{{'d3' 0.02}}),withfield(g,'TimeDomain',1));
p90r=struct('Flip',pi/2,'tp',0.016); p180r=struct('Flip',pi,'tp',0.032);
cases{end+1}=cc('pu_custom_real_2p',h1ax,struct('Field',330,'Sequence',{{p90r 0.08 p180r 0.08}},'nPoints',100,'Dim1',{{'d1,d2' 0.01}}),withfield(withfield(g,'nOffsets',21),'lwOffset',50));
cases{end+1}=cc('pu_custom_real_3p',h1ax,struct('Field',330,'Sequence',{{p90r 0.08 p90 0.0 p90r 0.08}},'nPoints',100,'Dim1',{{'d2' 0.01}}),withfield(withfield(g,'nOffsets',21),'lwOffset',50));
% components
cases{end+1}=struct('name','pu_twocomp','Sys',{{n14, withfield(h1,'weight',0.5)}},'Exp',e2,'Opt',g);
for n=1:numel(cases)
    c=cases{n};
    case_dir=fullfile(version_dir,c.name);
    if isfolder(case_dir), fprintf('  skip %s (exists)\n',c.name); continue; end
    fprintf('  case %s\n',c.name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    tic; [x,y,info]=saffron(c.Sys,c.Exp,withfield(c.Opt,'Verbosity',0)); t=toc;
    if iscell(x)
        oracle_write(h5,'/spec/x1',x{1}); oracle_write(h5,'/spec/x2',x{2});
    else
        oracle_write(h5,'/spec/x',x);
    end
    oracle_write(h5,'/spec/y',y);
    oracle_write(h5,'/spec/seconds',t);
    if isfield(info,'td') && ~isempty(info.td), oracle_write(h5,'/info/td',info.td); end
    if isfield(info,'fd') && ~isempty(info.fd), oracle_write(h5,'/info/fd',info.fd); end
    if isfield(info,'f'), oracle_write(h5,'/info/f',info.f); end
    if isfield(info,'f1'), oracle_write(h5,'/info/f1',info.f1); oracle_write(h5,'/info/f2',info.f2); end
    es_meta(case_dir,dump_file,struct('fn','saffron','Sys',{c.Sys},'Exp',c.Exp,'Opt',c.Opt));
end

% Helpers: larmorfrq, apowin, fdaxis, exponfit, sf_pathways
%-------------------------------------------------------------------------------
name='helpers5';
case_dir=fullfile(version_dir,name);
if isfolder(case_dir)
    fprintf('  skip %s (exists)\n',name);
else
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    oracle_write(h5,'/larmorfrq/one',larmorfrq('1H',350));
    oracle_write(h5,'/larmorfrq/many',larmorfrq('1H,14N,63Cu,2H',[300 350 1200]));
    wins={'ham','han','bla','bar','con','cos','wel'};
    for k=1:numel(wins)
        oracle_write(h5,['/apowin/' wins{k}],apowin(wins{k},64));
        oracle_write(h5,['/apowin/' wins{k} '_plus'],apowin([wins{k} '+'],65));
        oracle_write(h5,['/apowin/' wins{k} '_minus'],apowin([wins{k} '-'],63));
    end
    oracle_write(h5,'/apowin/kai',apowin('kai',64,3)); oracle_write(h5,'/apowin/gau',apowin('gau',64,0.4)); oracle_write(h5,'/apowin/exp',apowin('exp',65,2));
    oracle_write(h5,'/apowin/exp_plus',apowin('exp+',64,2));
    oracle_write(h5,'/fdaxis/even',fdaxis(0.01,256)); oracle_write(h5,'/fdaxis/odd',fdaxis(0.01,255));
    oracle_write(h5,'/fdaxis/fromaxis',fdaxis((0:99)*0.02));
    t=(0:199)*0.01; y=3+2*exp(-t/0.5); y2=[y; 1-0.5*exp(-t/0.3)].';
    rng(5); yn=y+0.01*randn(size(y));
    [k,c,yfit]=exponfit(t,y,1); oracle_write(h5,'/exponfit/k1',k); oracle_write(h5,'/exponfit/c1',c); oracle_write(h5,'/exponfit/fit1',yfit);
    [k,c,yfit]=exponfit(t,yn,1); oracle_write(h5,'/exponfit/kn',k); oracle_write(h5,'/exponfit/cn',c); oracle_write(h5,'/exponfit/fitn',yfit);
    [k,c,yfit]=exponfit(t,y,1,'noconst'); oracle_write(h5,'/exponfit/knc',k); oracle_write(h5,'/exponfit/cnc',c); oracle_write(h5,'/exponfit/fitnc',yfit);
    yd=3+2*exp(-t/0.5)+exp(-t/0.1);
    [k,c,yfit]=exponfit(t,yd,2); oracle_write(h5,'/exponfit/k2',k); oracle_write(h5,'/exponfit/c2',c); oracle_write(h5,'/exponfit/fit2',yfit);
    [k,c,yfit]=exponfit(t,y2,1); oracle_write(h5,'/exponfit/kcols',k); oracle_write(h5,'/exponfit/ccols',c); oracle_write(h5,'/exponfit/fitcols',yfit);
    oracle_write(h5,'/exponfit/t',t); oracle_write(h5,'/exponfit/y',y); oracle_write(h5,'/exponfit/yn',yn); oracle_write(h5,'/exponfit/yd',yd); oracle_write(h5,'/exponfit/y2',y2);
    % sf_pathways is a private function: callable only from its own folder (addpath does not work for private)
    old_dir=cd(fullfile(fileparts(which('saffron')),'private'));
    oracle_write(h5,'/sf_pathways/p3',sf_pathways([0.08 0 0.08],[0 1 0]));
    oracle_write(h5,'/sf_pathways/p2',sf_pathways([0.08 0.08],[1 1]));
    oracle_write(h5,'/sf_pathways/p4',sf_pathways([0.08 0 0 0.08],[0 1 2 0]));
    oracle_write(h5,'/sf_pathways/p4b',sf_pathways([0.08 0 0 0.08],[0 1 1 0]));
    oracle_write(h5,'/sf_pathways/p5',sf_pathways([0.1 0.2 0.1 0.2 0.1],[0 1 0 -1 0]));
    oracle_write(h5,'/sf_pathways/p1',sf_pathways(0.1,0));
    cd(old_dir);
    es_meta(case_dir,dump_file,struct());
end

end

% A copy of a structure with one field set (setfield() would take a cell value as subscripts)
function s=withfield(s,name,value)
s.(name)=value;
end
