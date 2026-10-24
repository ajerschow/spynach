% EasySpin reference data for epreasy's ENDOR and magnetometry functions
% (phase 4): endorfrq and endorfrq_perturb on crystal orientations, orisel
% weights, salt spectra of powders and crystals, and curry moments and
% susceptibilities. Syntax:
%
%                       esdump_endor(version_dir)
%
% version_dir is <fixtures_root>/<easyspin version>. One directory per case
% is written below it with data.h5 and meta.json; the inputs of each case
% are recorded in meta.json, from which tests/epreasy/test_parity_endor.py
% rebuilds the case. Cases whose directory already exists are skipped.

function esdump_endor(version_dir)

dump_file=[mfilename('fullpath') '.m'];
SampleFrame=[0 0 0; 0.3 1.1 0.7; 2.0 2.5 4.0];

% endorfrq and endorfrq_perturb on three crystal orientations
%-------------------------------------------------------------------------------
cases={};
nitro=struct('g',[2.0088 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'Q',[-0.5 0.2],'HStrain',[10 10 30]);
cu=struct('S',1/2,'g',[2.05 2.05 2.2],'Nucs','63Cu,14N','A',[50 50 500; 40 40 44],'AFrame',[0 0 0; 0.3 0.5 0.7]);
proton=struct('S',3/2,'D',200,'Nucs','1H','A',[3 5]);
xb=struct('Field',350,'mwFreq',9.8,'SampleFrame',SampleFrame);
cases{end+1}={'en_nitroxide',    'endorfrq',nitro,xb,struct()};
cases{end+1}={'en_nitroxide_enh','endorfrq',nitro,xb,struct('Enhancement','on')};
cases{end+1}={'en_nitroxide_exc','endorfrq',nitro,withfield(xb,'ExciteWidth',100),struct()};
cases{end+1}={'en_nitroxide_temp','endorfrq',nitro,withfield(xb,'Temperature',4),struct()};
cases{end+1}={'en_nitroxide_range','endorfrq',nitro,withfield(xb,'Range',[0 30]),struct('Threshold',1e-4)};
cases{end+1}={'en_nitroxide_rf',  'endorfrq',nitro,withfield(xb,'rf',1.2),struct()};
cases{end+1}={'en_cu2n',          'endorfrq',cu,struct('Field',320,'mwFreq',9.5,'SampleFrame',SampleFrame),struct('Nuclei',2)};
cases{end+1}={'en_cu2n_alltrans', 'endorfrq',cu,struct('Field',320,'mwFreq',9.5,'SampleFrame',SampleFrame),struct('Transitions',[1 2; 3 4; 2 3; 5 7])};
cases{end+1}={'en_proton_s32',    'endorfrq',proton,struct('Field',350,'mwFreq',9.5,'SampleFrame',SampleFrame,'ExciteWidth',200),struct()};
cases{end+1}={'en_pt1_nitroxide', 'endorfrq_perturb',nitro,xb,struct('PerturbOrder',1)};
cases{end+1}={'en_pt2_nitroxide', 'endorfrq_perturb',nitro,xb,struct('PerturbOrder',2)};
cases{end+1}={'en_pt2_cu2n',      'endorfrq_perturb',cu,struct('Field',320,'mwFreq',9.5,'SampleFrame',SampleFrame),struct('PerturbOrder',2)};
cases{end+1}={'en_pt2_proton_s32','endorfrq_perturb',proton,struct('Field',350,'mwFreq',9.5,'SampleFrame',SampleFrame,'ExciteWidth',200),struct('PerturbOrder',2)};
cases{end+1}={'en_pt2_exc',       'endorfrq_perturb',nitro,withfield(xb,'ExciteWidth',100),struct('PerturbOrder',2)};
for n=1:numel(cases)
    c=cases{n};
    case_dir=fullfile(version_dir,c{1});
    if isfolder(case_dir), fprintf('  skip %s (exists)\n',c{1}); continue; end
    fprintf('  case %s\n',c{1}); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    if strcmp(c{2},'endorfrq')
        [Pos,Int,Trans,Info]=endorfrq(c{3},c{4},c{5});
        oracle_write(h5,'/res/trans',Trans);
    else
        [Pos,Int,Trans,Info]=endorfrq_perturb(c{3},c{4},c{5});
    end
    oracle_write(h5,'/res/pos',Pos);
    oracle_write(h5,'/res/int',Int);
    oracle_write(h5,'/res/selectivity',Info.Selectivity);
    es_meta(case_dir,dump_file,struct('fn',c{2},'Sys',c{3},'Exp',c{4},'Opt',c{5}));
end

% orisel weights
%-------------------------------------------------------------------------------
name='orisel';
case_dir=fullfile(version_dir,name);
if isfolder(case_dir)
    fprintf('  skip %s (exists)\n',name);
else
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    Sys=struct('g',[2.25 2.25 2],'HStrain',[100 100 100]);
    Exp=struct('Field',326.5,'mwFreq',9.7,'ExciteWidth',50);
    oracle_write(h5,'/orisel/grid',orisel(Sys,Exp,struct('GridSize',10)));
    [W,T]=orisel(Sys,Exp,struct('GridSize',10));
    oracle_write(h5,'/orisel/grid_weights',W); oracle_write(h5,'/orisel/grid_trans',T);
    ori=[0.1 0.5 1.2 2.0; 0.2 0.8 1.1 1.5];
    oracle_write(h5,'/orisel/ori',ori);
    oracle_write(h5,'/orisel/user',orisel(Sys,withfield(Exp,'Orientations',ori)));
    Sys2=struct('S',1,'g',[2.0 2.1 2.2],'D',[300 50],'Nucs','14N','A',[10 10 50],'HStrain',[30 30 30]);
    oracle_write(h5,'/orisel/triplet',orisel(Sys2,withfield(Exp,'Orientations',ori)));
    oracle_write(h5,'/orisel/triplet_nonavg',orisel(Sys2,withfield(Exp,'Orientations',ori),struct('AveragedIntensity',false)));
    es_meta(case_dir,dump_file,struct('Sys',Sys,'Exp',Exp,'Sys2',Sys2,'ori',ori));
end

% salt spectra
%-------------------------------------------------------------------------------
cases={};
sc=@(name,Sys,Exp,Opt) struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt);
s32=struct('S',3/2,'D',200,'Nucs','1H','A',[3 5],'lwEndor',0.2);
e32=struct('Field',350,'mwFreq',9.5,'Range',[0 30]);
cases{end+1}=sc('sa_s32_matrix',s32,e32,struct('Method','matrix'));
cases{end+1}=sc('sa_s32_perturb2',s32,e32,struct('Method','perturb2'));
cases{end+1}=sc('sa_s32_perturb1',s32,e32,struct('Method','perturb1'));
cases{end+1}=sc('sa_s32_nointerp',s32,e32,struct('GridSize',[31 1]));
cases{end+1}=sc('sa_s32_separate',s32,e32,struct('separate','transitions'));
cases{end+1}=sc('sa_s32_harmonic',s32,withfield(e32,'Harmonic',1),struct());
cases{end+1}=sc('sa_s32_lorentz',withfield(s32,'lwEndor',[0 0.3]),e32,struct());
cases{end+1}=sc('sa_s32_autorange',s32,struct('Field',350,'mwFreq',9.5),struct());
n14=struct('Nucs','14N','g',[2.25 2.25 2],'A',[44 49 54],'Q',[1 1 -2]*0.84,'HStrain',[1 1 1]*100,'lwEndor',0);
en14=struct('Range',[16 65],'Field',326.5,'nPoints',2001,'mwFreq',9.7);
cases{end+1}=sc('sa_n14_enh_on',n14,en14,struct('Threshold',1e-4,'GridSize',[31 12],'Intensity','on','Enhancement','on'));
cases{end+1}=sc('sa_n14_enh_off',n14,en14,struct('Threshold',1e-4,'GridSize',[31 12],'Intensity','on','Enhancement','off'));
cases{end+1}=sc('sa_n14_int_off',n14,en14,struct('Threshold',1e-4,'GridSize',[31 12],'Intensity','off','Enhancement','off'));
cases{end+1}=sc('sa_n14_preselect',n14,withfield(en14,'ExciteWidth',50),struct('Threshold',1e-4,'GridSize',[31 12],'OriPreSelect',1));
w14=struct('S',1/2,'Nucs','14N','g',[2.2 2.2 2],'A',[4 4 5],'Q',-0.84*[-1 -1 2],'HStrain',[1 1 1],'lwEndor',0.1);
ew14=struct('Range',[0 35],'Field',3394,'nPoints',2^12,'mwFreq',95);
cases{end+1}=sc('sa_w14_full',w14,ew14,struct('Threshold',1e-4,'GridSize',50));
cases{end+1}=sc('sa_w14_excite_gz',w14,withfield(ew14,'ExciteWidth',3e3),struct('Threshold',1e-4,'GridSize',50));
cases{end+1}=sc('sa_w14_excite_gx',w14,withfield(withfield(ew14,'ExciteWidth',3e3),'mwFreq',104.51),struct('Threshold',1e-4,'GridSize',50));
iso=struct('Nucs','13C','A',5,'lwEndor',0.5);
cases{end+1}=sc('sa_iso_temp5',iso,struct('Field',1150,'Range',[10 20],'Temperature',5),struct());
cases{end+1}=sc('sa_iso_temp80',iso,struct('Field',1150,'Range',[10 20],'Temperature',80),struct());
cases{end+1}=sc('sa_iso',iso,struct('Field',1150,'Range',[5 20]),struct());
cu=struct('S',1/2,'g',[2.05 2.05 2.2],'Nucs','63Cu,14N','A',[50 50 500; 40 40 44],'lwEndor',0.3);
cases{end+1}=sc('sa_cu2n_nuclei',cu,struct('Field',320,'mwFreq',9.5,'Range',[0 40],'ExciteWidth',150),struct('Nuclei',2));
cases{end+1}=sc('sa_crystal',withfield(s32,'lwEndor',0.2),withfield(withfield(e32,'SampleFrame',[0 0 0; 0.3 1.1 0.7]),'MolFrame',[0 0 0]),struct());
cases{end+1}=sc('sa_crystal_separate',withfield(s32,'lwEndor',0.2),withfield(withfield(e32,'SampleFrame',[0 0 0; 0.3 1.1 0.7]),'MolFrame',[0 0 0]),struct('separate','orientations'));
two={struct('Nucs','1H','A',[3 5],'lwEndor',0.2), struct('Nucs','1H','A',[8 10],'lwEndor',0.3,'weight',0.5)};
cases{end+1}=struct('name','sa_twocomp','Sys',{two},'Exp',e32,'Opt',struct());
for n=1:numel(cases)
    c=cases{n};
    case_dir=fullfile(version_dir,c.name);
    if isfolder(case_dir), fprintf('  skip %s (exists)\n',c.name); continue; end
    fprintf('  case %s\n',c.name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    salt(c.Sys,c.Exp,c.Opt);  % warm-up; the second call is timed for the project page
    tic; [x,spec,info]=salt(c.Sys,c.Exp,c.Opt); t=toc;
    oracle_write(h5,'/spec/x',x);
    oracle_write(h5,'/spec/spec',spec);
    oracle_write(h5,'/spec/seconds',t);
    if isfield(info,'Transitions') && ~iscell(info.Transitions) && ~isempty(info.Transitions)
        oracle_write(h5,'/spec/transitions',info.Transitions);
    end
    es_meta(case_dir,dump_file,struct('fn','salt','Sys',{c.Sys},'Exp',c.Exp,'Opt',c.Opt));
end

% curry
%-------------------------------------------------------------------------------
cases={};
cc=@(name,Sys,Exp,Opt) struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt);
dimer=struct('S',[1/2 1/2],'g',[2 2],'ee',-2*-4*30e3);
cases{end+1}=cc('cu_dimer_mag',dimer,struct('Temperature',1,'Field',linspace(0,17,20)*1e3),struct('Output','muBM'));
cases{end+1}=cc('cu_dimer_chi',dimer,struct('Temperature',1:100,'Field',1000),struct('Output','chimol'));
cases{end+1}=cc('cu_mueff_si',struct('g',2,'S',1),struct('Field',10,'Temperature',100),struct('Output','mueff','Units','SI'));
cases{end+1}=cc('cu_mueff_cgs',struct('g',2,'S',1),struct('Field',10,'Temperature',100),struct('Output','mueff','Units','CGS'));
sel=struct('S',[7/2 1/2],'ee',-2*5*30e3);
cases{end+1}=cc('cu_spin1',sel,struct('Temperature',4,'Field',0:100:3000),struct('Output','mu','Spins',1));
cases{end+1}=cc('cu_spin2',sel,struct('Temperature',4,'Field',0:100:3000),struct('Output','mu','Spins',2));
trip=struct('S',1,'g',2,'D',5*30e3);
cases{end+1}=cc('cu_crystal_z',trip,struct('Temperature',0:5:50,'Field',0,'MolFrame',[0 0 0],'SampleFrame',[0 0 0]),struct('Units','CGS','Output','chimol'));
cases{end+1}=cc('cu_crystal_x',trip,struct('Temperature',0:5:50,'Field',0,'MolFrame',[0 0 0],'SampleFrame',[0 -pi/2 0]),struct('Units','CGS','Output','chimol'));
rhombic=struct('S',3/2,'g',[2.1 2.0 1.9],'D',[20e3 3e3]);
cases{end+1}=cc('cu_rhombic_all_si',rhombic,struct('Temperature',[2 20 200],'Field',[0 500 5000]),struct('Output','mu mumol muBM chimol chimolT 1/chimol mueff','Units','SI'));
cases{end+1}=cc('cu_rhombic_all_cgs',rhombic,struct('Temperature',[2 20 200],'Field',[0 500 5000]),struct('Output','mu mumol muBM chimol chimolT 1/chimol mueff','Units','CGS'));
cases{end+1}=cc('cu_rhombic_partition',rhombic,struct('Temperature',[2 20 200],'Field',[0 500 5000]),struct('Output','mu chimolT','Method','partitionfunction'));
cases{end+1}=cc('cu_rhombic_grid',rhombic,struct('Temperature',[2 20 200],'Field',[0 500 5000]),struct('Output','muBM chimol','GridSize',25));
cases{end+1}=cc('cu_rhombic_muvec',rhombic,struct('Temperature',[2 20],'Field',[500 5000],'MolFrame',[0.2 0.4 0.6],'SampleFrame',[0.3 1.1 0.7]),struct('Output','muvec'));
cases{end+1}=cc('cu_tip',struct('S',1/2,'g',2,'TIP',1e-9),struct('Temperature',[5 50],'Field',100),struct('Output','chimol'));
for n=1:numel(cases)
    c=cases{n};
    case_dir=fullfile(version_dir,c.name);
    if isfolder(case_dir), fprintf('  skip %s (exists)\n',c.name); continue; end
    fprintf('  case %s\n',c.name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    keywords=split(c.Opt.Output,' ');
    out=cell(1,numel(keywords));
    [out{:}]=curry(c.Sys,c.Exp,c.Opt);  % warm-up; the second call is timed for the project page
    tic; [out{:}]=curry(c.Sys,c.Exp,c.Opt); t=toc;
    oracle_write(h5,'/curry/seconds',t);
    for k=1:numel(keywords)
        oracle_write(h5,sprintf('/curry/out%d',k),out{k});
    end
    es_meta(case_dir,dump_file,struct('fn','curry','Sys',c.Sys,'Exp',c.Exp,'Opt',c.Opt));
end

end

% A copy of a structure with one field set (setfield() would take a cell value as subscripts)
function s=withfield(s,name,value)
s.(name)=value;
end
