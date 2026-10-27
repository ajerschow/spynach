% EasySpin reference data for easyspyn's slow-motion simulator chili (phase 5):
% spectra with the general Liouvillian method (Opt.LiouvMethod='general',
% the method that is ported), the same cases with EasySpin's default fast
% method for the cross-method comparison, and the angular-momentum helpers
% wigner3j, wigner6j, wignerd, tensor_cart2sph and tensor_sph2cart. Syntax:
%
%                       esdump_chili(version_dir)
%
% version_dir is <fixtures_root>/<easyspin version>. One directory per case
% is written below it with data.h5 and meta.json; the inputs of each case
% are recorded in meta.json, from which tests/easyspyn/test_parity_chili.py
% rebuilds the case. Cases whose directory already exists are skipped.

function esdump_chili(version_dir)

dump_file=[mfilename('fullpath') '.m'];
general=struct('LiouvMethod','general');

cases={};
cc=@(name,Sys,Exp,Opt) struct('name',name,'Sys',Sys,'Exp',Exp,'Opt',Opt);
nitro=struct('g',[2.008 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'lw',0.1);
xb=struct('mwFreq',9.8,'Range',[344 354],'nPoints',512);
for tc=[1e-10 1e-9 1e-8 1e-7]
    s=nitro; s.tcorr=tc;
    cases{end+1}=cc(sprintf('ch_nitroxide_%g',tc),s,xb,general); %#ok<AGROW>
end
s=nitro; s.tcorr=1e-9;
cases{end+1}=cc('ch_nitroxide_fast',s,xb,struct());
cases{end+1}=cc('ch_nitroxide_abs',s,withfield(xb,'Harmonic',0),general);
cases{end+1}=cc('ch_nitroxide_small',s,xb,withfield(general,'LLMK',[6 3 2 2]));
cases{end+1}=cc('ch_nitroxide_highfield',s,xb,withfield(general,'highField',true));
cases{end+1}=cc('ch_nitroxide_pimax',s,xb,withfield(general,'pImax',1));
cases{end+1}=cc('ch_nitroxide_mpsymm',s,xb,withfield(general,'MpSymm',true));
cases{end+1}=cc('ch_nitroxide_lorentz',withfield(s,'lw',[0.1 0.1]),xb,general);
cases{end+1}=cc('ch_nitroxide_lwpp',withfield(rmfield(s,'lw'),'lwpp',[0.1 0.05]),xb,general);
cases{end+1}=cc('ch_nitroxide_modamp',s,withfield(xb,'ModAmp',0.2),general);
cases{end+1}=cc('ch_nitroxide_phase',s,withfield(xb,'mwPhase',0.3),general);
cases{end+1}=cc('ch_nitroxide_temp',s,withfield(xb,'Temperature',20),general);
cases{end+1}=cc('ch_nitroxide_harm2',s,withfield(xb,'Harmonic',2),general);
cases{end+1}=cc('ch_nitroxide_autorange',s,struct('mwFreq',9.8),general);
cases{end+1}=cc('ch_nitroxide_nonzi',s,xb,withfield(general,'IncludeNZI',false));
cases{end+1}=cc('ch_nitroxide_norescale',s,xb,withfield(general,'Rescale',false));
cases{end+1}=cc('ch_nitroxide_axialdiff',withfield(rmfield(s,'tcorr'),'Diff',[1e8 1e9]),xb,general);
cases{end+1}=cc('ch_nitroxide_rhombdiff',withfield(rmfield(s,'tcorr'),'Diff',[3e8 1e8 1e9]),xb,general);
cases{end+1}=cc('ch_nitroxide_logdiff',withfield(rmfield(s,'tcorr'),'logDiff',8.5),xb,general);
cases{end+1}=cc('ch_nitroxide_logtcorr',withfield(rmfield(s,'tcorr'),'logtcorr',-8.5),xb,general);
cases{end+1}=cc('ch_nitroxide_gframe',withfield(withfield(s,'gFrame',[0.2 0.4 0.6]),'AFrame',[0.1 0.3 0.5]),xb,general);
cases{end+1}=cc('ch_nitroxide_tilt',withfield(s,'AFrame',[0 0.3 0]),xb,general);
cases{end+1}=cc('ch_nitroxide_director',s,withfield(xb,'SampleFrame',[0 0.7 0]),general);
% solvers and field-sweep methods
gax=struct('g',[2 2 2.1],'tcorr',1e-9);
fs=struct('Field',350,'mwRange',[9 11],'nPoints',100);
for sv={'L','\','E','C','B'}
    nm=sv{1}; if strcmp(nm,'\'), nm='backslash'; end
    cases{end+1}=cc(['ch_solver_' nm],gax,fs,withfield(withfield(general,'LLMK',[8 0 2 0]),'Solver',sv{1})); %#ok<AGROW>
end
cases{end+1}=cc('ch_solver_rtl',gax,fs,withfield(withfield(withfield(general,'LLMK',[8 0 2 0]),'Solver','L'),'Lentz',false));
dbl=struct('g',[2.01 2.003],'lw',0.1,'tcorr',1e-9);
fe=struct('mwFreq',9.5,'Harmonic',0,'nPoints',200,'Range',[337 339]);
cases{end+1}=cc('ch_sweep_approxlin',dbl,fe,withfield(general,'FieldSweepMethod','approxlin'));
cases{end+1}=cc('ch_sweep_approxinv',dbl,fe,withfield(general,'FieldSweepMethod','approxinv'));
cases{end+1}=cc('ch_sweep_explicit',dbl,fe,withfield(general,'FieldSweepMethod','explicit'));
% frequency sweeps, several nuclei, S=1, two electrons
cases{end+1}=cc('ch_freq_nitroxide',withfield(struct('g',[2.01 2.02 2.03],'Nucs','14N','A',[10 20 30],'lw',0.1),'tcorr',1e-8),struct('Field',340,'mwRange',[9.4 9.8],'nPoints',300),general);
cases{end+1}=cc('ch_freq_nucspins',withfield(struct('g',[2 2.001],'Nucs','63Cu','A',[100 200],'lw',5),'tcorr',1e-6),struct('Field',350,'mwRange',[9.4 10.2],'nPoints',600),withfield(general,'LLMK',[30 0 0 0]));
cases{end+1}=cc('ch_twonuclei',withfield(struct('g',[2.08 2.006 2.002],'Nucs','14N,1H','A',[20 20 100; 5 5 8]),'tcorr',0.02e-9),struct('mwFreq',9.5,'Range',[330 338],'nPoints',400),withfield(general,'LLMK',[6 0 2 2]));
cases{end+1}=cc('ch_postconv',withfield(struct('g',[2.08 2.006 2.002],'Nucs','14N,1H','A',[20 20 100; 5 5 8]),'tcorr',0.02e-9),struct('mwFreq',9.5,'Range',[330 338],'nPoints',400),withfield(withfield(general,'LLMK',[6 0 2 2]),'PostConvNucs',2));
cases{end+1}=cc('ch_triplet',withfield(struct('S',1,'g',[2.0 2.0 2.1],'D',[300 50],'lw',0.5),'tcorr',1e-8),struct('mwFreq',9.5,'Range',[300 380],'nPoints',300),withfield(general,'LLMK',[6 3 2 2]));
cases{end+1}=cc('ch_twoelectrons',withfield(struct('S',[1/2 1/2],'g',[2.0 2.0 2.1; 2.0 2.05 2.0],'ee',[-50 -50 100],'lw',0.3),'tcorr',1e-8),struct('mwFreq',9.5,'Range',[320 360],'nPoints',300),withfield(general,'LLMK',[6 3 2 2]));
% potentials and MOMD
pot=struct('g',[2.05 2.00],'logDiff',7);
pe=struct('Field',340,'mwRange',[9.4 9.9],'nPoints',200,'SampleFrame',[0 0 0]);
po=withfield(withfield(general,'LLMK',[4 0 2 2]),'highField',true);
LMKlam=[2 0 0 2; 2 1 0 2; 2 0 1 2; 2 1 1 2; 2 0 2 1; 4 0 0 1];
for p=1:size(LMKlam,1)
    cases{end+1}=cc(sprintf('ch_potential_%d%d%d',LMKlam(p,1),LMKlam(p,2),LMKlam(p,3)),withfield(pot,'Potential',LMKlam(p,:)),pe,po); %#ok<AGROW>
end
cases{end+1}=cc('ch_potential_two',withfield(pot,'Potential',[2 0 0 1.5; 2 0 2 0.5]),pe,po);
cases{end+1}=cc('ch_potential_complex',withfield(pot,'Potential',[2 1 1 1+0.5i]),pe,po);
cases{end+1}=cc('ch_potential_lowfield',withfield(pot,'Potential',[2 0 0 2]),pe,withfield(general,'LLMK',[4 0 2 2]));
momd=struct('g',[2.008 2.006 2.003],'Nucs','14N','A',[20 20 85],'lw',0.3,'tcorr',1e-8,'Potential',[2 0 0 1]);
me=struct('mwFreq',9.5,'CenterSweep',[338 20],'nPoints',300);
cases{end+1}=cc('ch_momd',momd,me,withfield(withfield(general,'GridSize',5),'LLMK',[8 0 2 2]));
cases{end+1}=cc('ch_momd_grid1',momd,me,withfield(withfield(general,'GridSize',1),'LLMK',[8 0 2 2]));
cases{end+1}=cc('ch_momd_ordering',momd,withfield(me,'Ordering',1.5),withfield(withfield(general,'GridSize',5),'LLMK',[8 0 2 2]));
cases{end+1}=cc('ch_momd_director',momd,withfield(me,'SampleFrame',[0 0.5 0]),withfield(general,'LLMK',[8 0 2 2]));
cases{end+1}=cc('ch_momd_fast',momd,me,struct('GridSize',5,'LLMK',[8 0 2 2]));
two={withfield(nitro,'tcorr',1e-9), withfield(withfield(nitro,'tcorr',1e-8),'weight',0.5)};
cases{end+1}=struct('name','ch_twocomp','Sys',{two},'Exp',xb,'Opt',general);
for n=1:numel(cases)
    c=cases{n};
    case_dir=fullfile(version_dir,c.name);
    if isfolder(case_dir), fprintf('  skip %s (exists)\n',c.name); continue; end
    fprintf('  case %s\n',c.name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    tic; [x,spec]=chili(c.Sys,c.Exp,c.Opt); t=toc;
    oracle_write(h5,'/spec/x',x);
    oracle_write(h5,'/spec/spec',spec);
    oracle_write(h5,'/spec/seconds',t);
    es_meta(case_dir,dump_file,struct('fn','chili','Sys',{c.Sys},'Exp',c.Exp,'Opt',c.Opt));
end

% Angular-momentum helpers
%-------------------------------------------------------------------------------
name='helpers4';
case_dir=fullfile(version_dir,name);
if isfolder(case_dir)
    fprintf('  skip %s (exists)\n',name);
else
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    rng(3);
    jm=[];
    for k=1:300
        j1=randi(12)-1; j2=randi(12)-1; j3=randi(12)-1;
        m1=randi(2*j1+1)-j1-1; m2=randi(2*j2+1)-j2-1; m3=-m1-m2;
        if abs(m3)>j3, continue; end
        jm(end+1,:)=[j1 j2 j3 m1 m2 m3]; %#ok<AGROW>
    end
    jm=[jm; 30 28 40 5 -3 -2; 25 25 50 25 -25 0; 1.5 0.5 1 0.5 -0.5 0; 2.5 1.5 2 0.5 0.5 -1];
    vals=zeros(size(jm,1),1);
    for k=1:size(jm,1), vals(k)=wigner3j(jm(k,1),jm(k,2),jm(k,3),jm(k,4),jm(k,5),jm(k,6)); end
    oracle_write(h5,'/wigner3j/jm',jm); oracle_write(h5,'/wigner3j/val',vals);
    jj=[1 1 1 1 1 1; 2 2 2 2 2 2; 3 2 1 2 3 2; 4 3 2 3 2 2; 5 5 4 4 5 3; 6 4 2 3 5 5; 1.5 0.5 1 0.5 1.5 1; 7 6 5 4 3 4];
    v6=zeros(size(jj,1),1);
    for k=1:size(jj,1), v6(k)=wigner6j(jj(k,1),jj(k,2),jj(k,3),jj(k,4),jj(k,5),jj(k,6)); end
    oracle_write(h5,'/wigner6j/jj',jj); oracle_write(h5,'/wigner6j/val',v6);
    ang=[0.3 0.7 1.1];
    for J=[0.5 1 1.5 2 3 4 6]
        tag=strrep(sprintf('%g',J),'.','p');
        oracle_write(h5,['/wignerd/D' tag],wignerd(J,ang(1),ang(2),ang(3)));
        oracle_write(h5,['/wignerd/d' tag],wignerd(J,ang(2)));
        oracle_write(h5,['/wignerd/Dplus' tag],wignerd(J,ang(1),ang(2),ang(3),'+'));
    end
    beta=linspace(0,pi,37);
    el=[2 0 0; 2 1 -1; 4 2 0; 6 -3 1; 9 4 -2; 12 0 0; 3 3 3; 5 -5 1];
    oracle_write(h5,'/wignerd/beta',beta); oracle_write(h5,'/wignerd/el',el);
    for k=1:size(el,1)
        oracle_write(h5,sprintf('/wignerd/el%d',k),wignerd(el(k,:),beta));
        oracle_write(h5,sprintf('/wignerd/elfull%d',k),wignerd(el(k,:),0.3,beta,1.1));
    end
    Tc=[2.0 0.3 0.1; 0.2 2.1 0.4; 0.0 0.5 2.3];
    [T0,T1,T2]=tensor_cart2sph(Tc);
    oracle_write(h5,'/tensor/Tc',Tc); oracle_write(h5,'/tensor/T0',T0); oracle_write(h5,'/tensor/T1',T1); oracle_write(h5,'/tensor/T2',T2);
    oracle_write(h5,'/tensor/back',tensor_sph2cart(T0,T1,T2));
    [T0,T1,T2]=tensor_cart2sph([1 2 3]); oracle_write(h5,'/tensor/vecT0',T0); oracle_write(h5,'/tensor/vecT2',T2);
    [T0,T1,T2]=tensor_cart2sph(2.5); oracle_write(h5,'/tensor/scalT0',T0);
    es_meta(case_dir,dump_file,struct('ang',ang));
end

end

% A copy of a structure with one field set (setfield() would take a cell value as subscripts)
function s=withfield(s,name,value)
s.(name)=value;
end
