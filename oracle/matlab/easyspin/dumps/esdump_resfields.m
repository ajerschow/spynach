% EasySpin reference data for the epreasy resonance-field layer: resfields
% (matrix and hybrid methods, strains, Boltzmann populations, excitation
% modes, photoselection) on crystal orientations, resfields_eig, the line
% shapes (gaussian, lorentzian, lshape, voigtian, convspec), sphgrid, the
% cubicsolve MEX and photoselect. Syntax:
%
%                     esdump_resfields(version_dir)
%
% version_dir is <fixtures_root>/<easyspin version>. One directory per case
% is written below it with data.h5 and meta.json; the inputs of each case
% are recorded in meta.json, from which tests/epreasy/test_parity_resfields.py
% rebuilds the case. Opt.FuzzLevel is 0 everywhere, since EasySpin's
% default adds random noise to the Hamiltonian. Cases whose directory
% already exists are skipped, never overwritten.

function esdump_resfields(version_dir)

dump_file=[mfilename('fullpath') '.m'];
SampleFrame=[0 0 0; 0.3 1.1 0.7; 2.0 2.5 4.0];

% resfields
cases=resfields_cases(SampleFrame);
for n=1:numel(cases)
    name=cases{n}.name;
    case_dir=fullfile(version_dir,name);
    if isfolder(case_dir), fprintf('  skip %s (exists)\n',name); continue; end
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    [Pos,Int,Wid,Trans]=resfields(cases{n}.Sys,cases{n}.Exp,cases{n}.Opt);
    oracle_write(h5,'/resfields/pos',Pos);
    oracle_write(h5,'/resfields/int',Int);
    oracle_write(h5,'/resfields/wid',Wid);
    oracle_write(h5,'/resfields/trans',Trans);
    es_meta(case_dir,dump_file,struct('Sys',cases{n}.Sys,'Exp',cases{n}.Exp,'Opt',cases{n}.Opt));
end

% resfields_eig
eig_cases={'eig_nitroxide',nitroxide(),[330 350]; 'eig_mn',mn(),[50 700]};
for n=1:size(eig_cases,1)
    name=eig_cases{n,1}; Sys=eig_cases{n,2};
    case_dir=fullfile(version_dir,name);
    if isfolder(case_dir), fprintf('  skip %s (exists)\n',name); continue; end
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    Exp=struct('mwFreq',9.5,'Range',eig_cases{n,3},'SampleFrame',SampleFrame);
    Opt=struct('Threshold',0);
    [B,I]=resfields_eig(Sys,Exp,Opt);
    for k=1:numel(B)
        oracle_write(h5,sprintf('/eig/B_%d',k),B{k});
        oracle_write(h5,sprintf('/eig/I_%d',k),I{k});
    end
    Bonly=resfields_eig(Sys,Exp);
    for k=1:numel(Bonly)
        oracle_write(h5,sprintf('/eig/Bonly_%d',k),Bonly{k});
    end
    es_meta(case_dir,dump_file,struct('Sys',Sys,'Exp',Exp,'Opt',Opt));
end

% line shapes
name='lineshapes';
case_dir=fullfile(version_dir,name);
if isfolder(case_dir)
    fprintf('  skip %s (exists)\n',name);
else
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    x=linspace(-20,20,801);
    for d=-1:2
        oracle_write(h5,sprintf('/gaussian/abs_%d',d+1),gaussian(x,1.3,2.0,d));
        if d>=0
            [ya,yd]=gaussian(x,1.3,2.0,d);
            oracle_write(h5,sprintf('/gaussian/disp_%d',d+1),yd);
            oracle_write(h5,sprintf('/gaussian/phase_%d',d+1),gaussian(x,1.3,2.0,d,0.4));
        end
        [ya,yd]=lorentzian(x,1.3,2.0,d);
        oracle_write(h5,sprintf('/lorentzian/abs_%d',d+1),ya);
        oracle_write(h5,sprintf('/lorentzian/disp_%d',d+1),yd);
        oracle_write(h5,sprintf('/lorentzian/phase_%d',d+1),lorentzian(x,1.3,2.0,d,0.4));
        oracle_write(h5,sprintf('/lshape/y_%d',d+1),lshape(x,1.3,[1.5 2.0],d,0.3,0.4));
    end
    for d=0:2
        [ya,yd]=voigtian(x,1.3,[1.5 2.0],d,0.2);
        oracle_write(h5,sprintf('/voigtian/abs_%d',d+1),ya);
        oracle_write(h5,sprintf('/voigtian/disp_%d',d+1),yd);
        oracle_write(h5,sprintf('/voigtian/narrowG_%d',d+1),voigtian(x,1.3,[2.0 1.5],d));
    end
    spec1=zeros(1,601); spec1([150 300 420])=[1 2 0.5];
    oracle_write(h5,'/convspec/spec1',spec1);
    oracle_write(h5,'/convspec/out1',convspec(spec1,0.05,1.2,1,0.4));
    oracle_write(h5,'/convspec/out1b',convspec(spec1,0.05,0.9,0,0.0,0.3));
    spec2=zeros(60,80); spec2(20,30)=1; spec2(45,10)=-2; spec2(5,70)=0.5;
    oracle_write(h5,'/convspec/spec2',spec2);
    oracle_write(h5,'/convspec/out2',convspec(spec2,[0.1 0.2],[1 2],[0 1],[1 0.5]));
    es_meta(case_dir,dump_file,struct('x',x));
end

% spherical grids
name='sphgrid';
case_dir=fullfile(version_dir,name);
if isfolder(case_dir)
    fprintf('  skip %s (exists)\n',name);
else
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    symms={'D2h','Ci','C1','Dinfh','C2h','C4h','D4h','O3','Th','S6'};
    for n=1:numel(symms)
        for gs=[7 19]
            [grid,tri]=sphgrid(symms{n},gs);
            base=sprintf('/%s_%d',symms{n},gs);
            oracle_write(h5,[base '/phi'],grid.phi);
            oracle_write(h5,[base '/theta'],grid.theta);
            oracle_write(h5,[base '/weights'],grid.weights);
            if ~isempty(tri.idx)
                oracle_write(h5,[base '/tri_idx'],double(tri.idx));
                oracle_write(h5,[base '/tri_areas'],tri.areas);
            end
        end
    end
    es_meta(case_dir,dump_file,struct('symms',{symms},'sizes',[7 19]));
end

% photoselect (the cubicsolve MEX is private to EasySpin and cannot be called
% from here; it was checked against the Python port on 1e5 cubics in the spike
% recorded in the plan)
name='photoselect';
case_dir=fullfile(version_dir,name);
if isfolder(case_dir)
    fprintf('  skip %s (exists)\n',name);
else
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    ori2=[0 0; 0.3 1.1; 2.0 2.5; 1.0 0.4]; ori3=[0 0 0; 0.3 1.1 0.7; 2.0 2.5 4.0];
    oracle_write(h5,'/photo/z_y_perp',photoselect('z',ori2,'y',pi/2));
    oracle_write(h5,'/photo/z_y_par',photoselect('z',ori2,'y',0));
    oracle_write(h5,'/photo/z_y_unpol',photoselect('z',ori2,'y',NaN));
    oracle_write(h5,'/photo/ang_vec_chi',photoselect([0.3 0.5],ori3,[0.2 0.1 0.9],0.7));
    oracle_write(h5,'/photo/ang_vec_chi_unpol',photoselect([0.3 0.5],ori3,[0.2 0.1 0.9],NaN));
    es_meta(case_dir,dump_file,struct('ori2',ori2,'ori3',ori3));
end

end

% Spin systems and the resfields cases
function Sys=nitroxide()
Sys=struct('g',[2.0088 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'lwpp',0.3);
end

function Sys=mn()
Sys=struct('S',5/2,'D',[1000 100],'DFrame',[0.2 0.5 0.9],'Nucs','55Mn','A',250);
end

function cases=resfields_cases(SampleFrame)
cases={};
opt0=struct('FuzzLevel',0);
xband=struct('mwFreq',9.5,'Range',[330 350],'SampleFrame',SampleFrame);

Sys=nitroxide();
cases{end+1}=struct('name','rf_nitroxide','Sys',Sys,'Exp',xband,'Opt',opt0);
cases{end+1}=struct('name','rf_nitroxide_parallel','Sys',Sys,'Exp',withfield(xband,'mwMode','parallel'),'Opt',opt0);
cases{end+1}=struct('name','rf_nitroxide_unpol','Sys',Sys,'Exp',withfield(xband,'mwMode',{'y','unpolarized'}),'Opt',opt0);
cases{end+1}=struct('name','rf_nitroxide_circ','Sys',Sys,'Exp',withfield(xband,'mwMode',{[0.3 1.2],'circular+'}),'Opt',opt0);
cases{end+1}=struct('name','rf_nitroxide_alltrans','Sys',Sys,'Exp',xband,'Opt',struct('FuzzLevel',0,'Transitions','all'));
cases{end+1}=struct('name','rf_nitroxide_molframe','Sys',Sys, ...
    'Exp',struct('mwFreq',9.5,'CenterSweep',[340 20],'SampleFrame',SampleFrame,'MolFrame',[0.4 0.8 1.6]),'Opt',opt0);
% strains: H, g and A strain on an untilted nitroxide
Sys=struct('g',[2.0088 2.0061 2.0027],'Nucs','14N','A',[16 16 86],'HStrain',[10 10 30],'gStrain',[0.002 0.001 0.001],'AStrain',[2 2 5]);
cases{end+1}=struct('name','rf_nitroxide_strain','Sys',Sys,'Exp',xband,'Opt',opt0);
% photoselection
Sys=withfield(nitroxide(),'tdm','xz');
cases{end+1}=struct('name','rf_nitroxide_photo','Sys',Sys,'Exp',withfield(xband,'lightBeam','perpendicular'),'Opt',opt0);
% Cu(II) with two 14N: matrix and hybrid methods
Sys=struct('S',1/2,'g',[2.05 2.2],'Nucs','63Cu,14N,14N','A',[50 50 500; 40 40 44; 38 38 42], ...
           'AFrame',[0 0 0; 0.2 0.4 0.6; -0.5 1.3 2.2],'Q',[0 0; -1.2 0.3; 1.0 0.5]);
cu=struct('mwFreq',9.5,'Range',[260 360],'SampleFrame',SampleFrame);
cases{end+1}=struct('name','rf_cu2n_matrix','Sys',Sys,'Exp',cu,'Opt',opt0);
cases{end+1}=struct('name','rf_cu2n_hybrid','Sys',Sys,'Exp',cu,'Opt',struct('FuzzLevel',0,'Method','hybrid','HybridCoreNuclei',1));
% Mn(II) with Boltzmann populations and with D strain
Sys=mn();
cases{end+1}=struct('name','rf_mn_boltzmann','Sys',Sys,'Exp',struct('mwFreq',9.5,'Range',[50 700],'SampleFrame',SampleFrame,'Temperature',4),'Opt',opt0);
Sys=withfield(mn(),'DStrain',[50 10]);
cases{end+1}=struct('name','rf_mn_dstrain','Sys',Sys,'Exp',struct('mwFreq',9.5,'Range',[50 700],'SampleFrame',SampleFrame),'Opt',opt0);
% radical pair with Boltzmann populations at 1.5 K
Sys=struct('S',[1/2 1/2],'g',[2.0023 2.0023 2.0023; 2.00 2.01 2.02],'gFrame',[0 0 0; 0.3 0.6 0.9], ...
           'J',5,'dip',[-10 2],'dvec',[0.3 -0.2 0.1],'eeFrame',[0.1 0.4 0.7],'ee2',2,'Nucs','1H','A',[10 10 12 0 0 0]);
cases{end+1}=struct('name','rf_radical_pair','Sys',Sys,'Exp',withfield(xband,'Temperature',1.5),'Opt',opt0);
end

% A copy of a structure with one field set (setfield() would take a cell value as subscripts)
function s=withfield(s,name,value)
s.(name)=value;
end
