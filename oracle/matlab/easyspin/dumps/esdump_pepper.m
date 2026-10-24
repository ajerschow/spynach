% EasySpin reference data for the epreasy simulators of phase 2: pepper
% (matrix, perturbation, hybrid and eigenfield methods; projection and
% summation paths; axial and rhombic grids; crystals; separate transitions;
% field modulation; dispersion), garlic (Breit-Rabi and perturbation
% methods, equivalent nuclei, fast motion, temperature, modulation,
% frequency sweep), fastmotion, hamsymm, fieldmod and equivcouple. Syntax:
%
%                       esdump_pepper(version_dir)
%
% version_dir is <fixtures_root>/<easyspin version>. One directory per case
% is written below it with data.h5 and meta.json; the inputs of each case
% are recorded in meta.json, from which tests/epreasy/test_parity_pepper.py
% rebuilds the case. Opt.FuzzLevel is 0 in every pepper case. Cases whose
% directory already exists are skipped, never overwritten.

function esdump_pepper(version_dir)

dump_file=[mfilename('fullpath') '.m'];

% pepper and garlic spectra
cases=[pepper_cases() garlic_cases()];
for n=1:numel(cases)
    c=cases{n};
    case_dir=fullfile(version_dir,c.name);
    if isfolder(case_dir), fprintf('  skip %s (exists)\n',c.name); continue; end
    fprintf('  case %s\n',c.name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    % wall clock of a second call (the first warms MATLAB up), for the timing table of the project page
    if strcmp(c.fn,'pepper')
        pepper(c.Sys,c.Exp,c.Opt);
        tic; [x,spec,info]=pepper(c.Sys,c.Exp,c.Opt); t=toc;
        oracle_write(h5,'/spec/x',x);
        oracle_write(h5,'/spec/spec',spec);
        if isfield(info,'Transitions'), oracle_write(h5,'/spec/transitions',info.Transitions); end
    else
        garlic(c.Sys,c.Exp,c.Opt);
        tic; [x,spec,info]=garlic(c.Sys,c.Exp,c.Opt); t=toc;
        oracle_write(h5,'/spec/x',x);
        oracle_write(h5,'/spec/spec',spec);
        oracle_write(h5,'/spec/resfields',info.resfields);
    end
    oracle_write(h5,'/spec/seconds',t);
    es_meta(case_dir,dump_file,struct('fn',c.fn,'Sys',c.Sys,'Exp',c.Exp,'Opt',c.Opt));
end

% fastmotion, hamsymm, fieldmod, equivcouple
name='helpers2';
case_dir=fullfile(version_dir,name);
if isfolder(case_dir)
    fprintf('  skip %s (exists)\n',name);
else
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    A=unitconvert([7.59 5.95 31.76]/10,'mT->MHz');
    fm1=struct('g',[2.0088 2.0064 2.0027],'Nucs','14N','A',A);
    [lw,mI]=fastmotion(fm1,350,1e-10);
    oracle_write(h5,'/fastmotion/lw1',lw); oracle_write(h5,'/fastmotion/mI1',mI);
    fm2=struct('g',[2.0088 2.0064 2.0027],'gFrame',[0.2 0.4 0.6],'Nucs','14N,1H','A',[A; 10 12 20],'AFrame',[0 0 0; 0.3 0.1 0.5],'Q',[-0.5 0.2; 0 0]);
    [lw,mI]=fastmotion(fm2,340,3e-11);
    oracle_write(h5,'/fastmotion/lw2',lw); oracle_write(h5,'/fastmotion/mI2',mI);
    oracle_write(h5,'/fastmotion/lw2freq',fastmotion(fm2,340,3e-11,'freq'));
    groups={'C1','Ci','C2h','D2h','C4h','D4h','S6','D3d','C6h','D6h','Th','Oh','Dinfh','O3'};
    hs=hamsymm_cases();
    for k=1:numel(hs)
        [G,R]=hamsymm(hs{k});
        oracle_write(h5,sprintf('/hamsymm/group_%d',k),find(strcmp(G,groups)));
        oracle_write(h5,sprintf('/hamsymm/R_%d',k),R);
    end
    B=linspace(300,400,1001); spc=lorentzian(B,342,4)+0.5*gaussian(B,360,6);
    oracle_write(h5,'/fieldmod/B',B); oracle_write(h5,'/fieldmod/spc',spc);
    oracle_write(h5,'/fieldmod/h1',fieldmod(B,spc,0.5,1));
    oracle_write(h5,'/fieldmod/h2',fieldmod(B,spc,3,2));
    ec={[1/2 5],[1 3],[3/2 2],[1/2 1]};
    for k=1:numel(ec)
        [F,N]=equivcouple(ec{k}(1),ec{k}(2));
        oracle_write(h5,sprintf('/equivcouple/F_%d',k),F); oracle_write(h5,sprintf('/equivcouple/N_%d',k),N);
    end
    es_meta(case_dir,dump_file,struct('fm1',fm1,'fm2',fm2,'hamsymm',{hs},'equivcouple',{ec}));
end

end

function cases=pepper_cases()
cases={};
nitroxide=struct('g',[2.0088 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'lwpp',0.3);
xband=struct('mwFreq',9.5,'Range',[330 350],'nPoints',1024);
opt0=struct('FuzzLevel',0);
pc=@(name,Sys,Exp,Opt) struct('name',name,'fn','pepper','Sys',Sys,'Exp',Exp,'Opt',Opt);
cases{end+1}=pc('pp_nitroxide_matrix',nitroxide,xband,opt0);
cases{end+1}=pc('pp_nitroxide_perturb2',nitroxide,xband,struct('FuzzLevel',0,'Method','perturb2'));
cases{end+1}=pc('pp_nitroxide_perturb1',nitroxide,xband,struct('FuzzLevel',0,'Method','perturb1'));
cases{end+1}=pc('pp_nitroxide_hybrid',nitroxide,xband,struct('FuzzLevel',0,'Method','hybrid'));
cases{end+1}=pc('pp_nitroxide_eig',nitroxide,withfield(xband,'Harmonic',0),struct('FuzzLevel',0,'Method','eig'));
cases{end+1}=pc('pp_nitroxide_bruteforce',nitroxide,withfield(xband,'Harmonic',0),struct('FuzzLevel',0,'BruteForce',1));
cases{end+1}=pc('pp_nitroxide_nointerp',nitroxide,xband,struct('FuzzLevel',0,'GridSize',[19 1]));
cases{end+1}=pc('pp_nitroxide_grid31x2',nitroxide,xband,struct('FuzzLevel',0,'GridSize',[31 2]));
cases{end+1}=pc('pp_nitroxide_separate',nitroxide,xband,struct('FuzzLevel',0,'separate','transitions'));
cases{end+1}=pc('pp_nitroxide_modamp',nitroxide,withfield(xband,'ModAmp',0.1),opt0);
cases{end+1}=pc('pp_nitroxide_autorange',nitroxide,struct('mwFreq',9.5),opt0);
cases{end+1}=pc('pp_nitroxide_lorentz_phase',withfield(nitroxide,'lwpp',[0 0.3]),withfield(xband,'mwPhase',0.3),opt0);
cases{end+1}=pc('pp_nitroxide_harmonic2',nitroxide,withfield(xband,'Harmonic',2),opt0);
cases{end+1}=pc('pp_nitroxide_temperature',nitroxide,withfield(xband,'Temperature',2),opt0);
cases{end+1}=pc('pp_nitroxide_parallel',nitroxide,withfield(xband,'mwMode','parallel'),opt0);
strained=struct('g',[2.0088 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'HStrain',[20 20 60]);
cases{end+1}=pc('pp_nitroxide_hstrain',strained,xband,opt0);
cases{end+1}=pc('pp_nitroxide_hstrain_lw',withfield(strained,'lwpp',0.2),xband,opt0);
cases{end+1}=pc('pp_nitroxide_gstrain',struct('g',[2.0088 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'gStrain',[0.002 0.001 0.001],'AStrain',[2 2 5]),xband,opt0);
cases{end+1}=pc('pp_axial_g',struct('g',[2.0 2.0 2.2],'lwpp',1),struct('mwFreq',9.5,'Range',[280 360]),opt0);
cases{end+1}=pc('pp_axial_g_hstrain',struct('g',[2.0 2.0 2.2],'HStrain',[30 30 90]),struct('mwFreq',9.5,'Range',[280 360]),opt0);
cases{end+1}=pc('pp_rhombic_g',struct('g',[2.0 2.1 2.2],'lwpp',1),struct('mwFreq',9.5,'Range',[280 360]),opt0);
cases{end+1}=pc('pp_tilted_frames',struct('g',[2.0088 2.0061 2.0027],'gFrame',[0.3 0.6 0.9],'Nucs','14N','A',[16 16 86],'AFrame',[0.2 0.5 1.1],'lwpp',0.3),xband,opt0);
mn=struct('S',5/2,'D',[1000 100],'Nucs','55Mn','A',250,'lwpp',1);
cases{end+1}=pc('pp_mn_powder',mn,struct('mwFreq',9.5,'Range',[50 700],'Harmonic',0),struct('FuzzLevel',0,'Threshold',1e-3));
cases{end+1}=pc('pp_mn_perturb2',mn,struct('mwFreq',9.5,'Range',[50 700],'Harmonic',0),struct('FuzzLevel',0,'Method','perturb2'));
triplet=struct('S',1,'g',[2.003 2.003 2.003],'D',[500 60],'lwpp',1);
cases{end+1}=pc('pp_triplet',triplet,struct('mwFreq',9.5,'Range',[300 380]),opt0);
cases{end+1}=pc('pp_triplet_dstrain',withfield(withfield(triplet,'DStrain',[40 10]),'lwpp',0),struct('mwFreq',9.5,'Range',[300 380]),opt0);
cu=struct('S',1/2,'g',[2.05 2.05 2.2],'Nucs','63Cu,14N,14N,14N,14N','A',[50 50 500; 40 40 40; 40 40 40; 40 40 40; 40 40 40],'lwpp',1);
cases{end+1}=pc('pp_cu4n_perturb2',cu,struct('mwFreq',9.5,'Range',[260 360]),struct('FuzzLevel',0,'Method','perturb2'));
cases{end+1}=pc('pp_cu4n_hybrid',cu,struct('mwFreq',9.5,'Range',[260 360]),struct('FuzzLevel',0,'Method','hybrid','HybridCoreNuclei',1));
cases{end+1}=pc('pp_crystal',nitroxide,withfield(withfield(xband,'SampleFrame',[0 0 0; 0.3 1.1 0.7]),'MolFrame',[0 0 0]),opt0);
cases{end+1}=pc('pp_crystal_summed',nitroxide,withfield(withfield(xband,'SampleFrame',[0 0 0; 0.3 1.1 0.7]),'MolFrame',[0.2 0.4 0.6]),struct('FuzzLevel',0,'separate',''));
cases{end+1}=pc('pp_crystal_eig',nitroxide,withfield(withfield(withfield(xband,'SampleFrame',[0.3 1.1 0.7]),'MolFrame',[0 0 0]),'Harmonic',0),struct('FuzzLevel',0,'Method','eig'));
end

function cases=garlic_cases()
cases={};
gc=@(name,Sys,Exp,Opt) struct('name',name,'fn','garlic','Sys',Sys,'Exp',Exp,'Opt',Opt);
nitro=struct('g',2.006,'Nucs','14N','A',47,'lwpp',0.1);
cases{end+1}=gc('gl_nitroxide',nitro,struct('mwFreq',9.5,'Range',[330 350]),struct());
cases{end+1}=gc('gl_nitroxide_auto',nitro,struct('mwFreq',9.5),struct());
cases{end+1}=gc('gl_nitroxide_perturb2',nitro,struct('mwFreq',9.5,'Range',[330 350]),struct('Method','perturb2'));
cases{end+1}=gc('gl_nitroxide_explicit',withfield(nitro,'lwpp',[0 0.1]),struct('mwFreq',9.5,'Range',[330 350]),struct('AccumMethod','explicit'));
cases{end+1}=gc('gl_nitroxide_gauss_lorentz',withfield(nitro,'lwpp',[0.1 0.05]),struct('mwFreq',9.5,'Range',[330 350],'Harmonic',2),struct());
cases{end+1}=gc('gl_nitroxide_modamp',nitro,struct('mwFreq',9.5,'Range',[330 350],'ModAmp',0.2),struct());
cases{end+1}=gc('gl_nitroxide_temperature',nitro,struct('mwFreq',9.5,'Range',[330 350],'Temperature',1.5),struct());
cases{end+1}=gc('gl_nitroxide_phase',withfield(nitro,'lwpp',[0 0.1]),struct('mwFreq',9.5,'Range',[330 350],'mwPhase',0.4),struct('AccumMethod','explicit'));
cases{end+1}=gc('gl_protons',struct('g',2.0,'Nucs','1H,1H','n',[3 2],'A',[30 12],'lw',[0.2 0]),struct('mwFreq',9.5,'Range',[325 352],'Harmonic',0),struct());
cases{end+1}=gc('gl_cu_n',struct('g',2.1,'Nucs','63Cu,14N','A',[200 40],'lwpp',0.5),struct('mwFreq',9.5,'Range',[290 360]),struct());
A=unitconvert([7.59 5.95 31.76]/10,'mT->MHz');
fm=struct('g',[2.0088 2.0064 2.0027],'Nucs','14N','A',A,'tcorr',1e-10);
cases{end+1}=gc('gl_fastmotion',fm,struct('mwFreq',9.5,'Range',[335 345]),struct());
cases{end+1}=gc('gl_fastmotion_lw',withfield(fm,'lwpp',[0.05 0.02]),struct('mwFreq',9.5,'Range',[335 345]),struct());
nitro_lw=struct('g',2.006,'Nucs','14N','A',47,'lw',0.1);
cases{end+1}=gc('gl_freqsweep',nitro_lw,struct('Field',340,'mwRange',[9.4 9.6]),struct());
cases{end+1}=gc('gl_freqsweep_auto',nitro_lw,struct('Field',340),struct());
end

function hs=hamsymm_cases()
hs={struct('g',2.0), struct('g',[2.0 2.0 2.2]), struct('g',[2.0 2.1 2.2]), ...
    struct('g',[2.0 2.1 2.2],'Nucs','14N','A',[10 10 50],'AFrame',[0.3 0.7 1.1]), ...
    struct('g',[2.0 2.0 2.2],'Nucs','14N','A',[50 10 10]), ...
    struct('g',[2.0 2.0 2.2],'Nucs','14N','A',[10 10 50],'AFrame',[0 pi/2 0]), ...
    struct('g',[2.01 0.002 0; 0.002 2.02 0; 0 0 2.05]), ...
    struct('S',5/2,'B4',[0 0 0 0 10 0 0 0 0]), ...
    struct('S',1,'D',[500 0]), struct('S',1,'D',[500 60],'DFrame',[0.2 0.4 0.6]), ...
    struct('g',[2.0 2.0 2.2],'gFrame',[0.4 0.8 1.2],'Nucs','1H','A',[5 5 9],'AFrame',[0.4 0.8 1.2])};
end

% A copy of a structure with one field set (setfield() would take a cell value as subscripts)
function s=withfield(s,name,value)
s.(name)=value;
end
