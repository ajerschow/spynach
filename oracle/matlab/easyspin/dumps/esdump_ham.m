% EasySpin reference data for the easyspyn foundation: spin vectors and
% state counts of validated spin systems, spin operators (sop), extended
% Stevens operators (stev), the Hamiltonian components (ham_ez, ham_nz,
% ham_hf, ham_zf, ham_nq, ham_ee, ham_nn), ham() with and without a field,
% levels(), and the helpers erot, eulang, ang2vec, vec2ang, mt2mhz,
% unitconvert and nucdata. Syntax:
%
%                        esdump_ham(version_dir)
%
% version_dir is <fixtures_root>/<easyspin version>. One directory per case
% is written below it with data.h5 and meta.json; the spin system and the
% parameters of each case are recorded in meta.json (inputs), from which
% tests/easyspyn/test_parity.py rebuilds the case. Cases whose directory
% already exists are skipped, never overwritten.

function esdump_ham(version_dir)

dump_file=[mfilename('fullpath') '.m'];
B0=[120 -45 300];               % mT, field vector in the molecular frame
Ori=[0 0; pi/3 pi/4; 1.2 2.0];  % [phi theta] of the field in the molecular frame
Bfields=[0 100 350 1200];       % mT

% Spin systems
cases=spin_system_cases();
for n=1:numel(cases)
    name=cases{n}.name; Sys=cases{n}.Sys;
    case_dir=fullfile(version_dir,name);
    if isfolder(case_dir), fprintf('  skip %s (exists)\n',name); continue; end
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');

    % Validated spin system
    spins=spinvec(Sys);
    oracle_write(h5,'/sys/Spins',spins);
    oracle_write(h5,'/sys/nStates',hsdim(Sys));

    % Hamiltonian components and the field split
    [H0,mux,muy,muz]=ham(Sys);
    oracle_write(h5,'/ham/H0',H0);
    oracle_write(h5,'/ham/mux',mux);
    oracle_write(h5,'/ham/muy',muy);
    oracle_write(h5,'/ham/muz',muz);
    oracle_write(h5,'/ham/H_B',ham(Sys,B0));
    [~,muzL]=ham(Sys,B0);
    oracle_write(h5,'/ham/muzL',muzL);
    oracle_write(h5,'/ham/zf',ham_zf(Sys));
    oracle_write(h5,'/ham/ee',ham_ee(Sys));
    oracle_write(h5,'/ham/hf',ham_hf(Sys));
    oracle_write(h5,'/ham/nq',ham_nq(Sys));
    oracle_write(h5,'/ham/nn',ham_nn(Sys));
    oracle_write(h5,'/ham/ez_B',ham_ez(Sys,B0));
    if ~isempty(Sys.Nucs) && any(nucspin(Sys.Nucs)>0)
        oracle_write(h5,'/ham/nz_B',ham_nz(Sys,B0));
    end

    % Energy levels
    oracle_write(h5,'/levels/E',levels(Sys,Ori,Bfields));
    oracle_write(h5,'/levels/E_z',levels(Sys,'z',350));

    % Spin operators: every component of every spin, a product, a transition-selective one
    for i=1:numel(spins)
        for c=1:5
            oracle_write(h5,sprintf('/sop/s%d_c%d',i,c),sop(spins,[i c]));
        end
    end
    if numel(spins)>=2
        oracle_write(h5,'/sop/s1x_s2z',sop(spins,[1 1; 2 3]));
    end
    if 2*spins(1)+1>=3
        oracle_write(h5,'/sop/tsel',sop(spins,'x(1|3)1'));
    else
        oracle_write(h5,'/sop/tsel',sop(spins,'z(1|2)1'));
    end

    % Extended Stevens operators of the first electron spin, all ranks up to min(2S,6)
    S1=spins(1);
    for k=1:min(2*S1,6)
        for q=-k:k
            oracle_write(h5,sprintf('/stev/k%d_q%d',k,q),stev(spins,[k q 1]));
        end
    end

    inputs=struct('Sys',Sys,'B0',B0,'Ori',Ori,'Bfields',Bfields);
    es_meta(case_dir,dump_file,inputs);
end

% Helpers: rotations, units, nuclear data
name='helpers';
case_dir=fullfile(version_dir,name);
if isfolder(case_dir)
    fprintf('  skip %s (exists)\n',name);
else
    fprintf('  case %s\n',name); mkdir(case_dir);
    h5=fullfile(case_dir,'data.h5');
    angles=[0.4 1.1 2.3; 0 0 0; pi/2 pi/2 0; -0.3 2.9 5.1; 1.0 pi 0.5];
    for n=1:size(angles,1)
        R=erot(angles(n,:));
        oracle_write(h5,sprintf('/rot/erot_%d',n),R);
        oracle_write(h5,sprintf('/rot/eulang_%d',n),eulang(R));
    end
    phi=[0 0.3 1.7 4.0]; theta=[0 0.5 1.2 2.9];
    oracle_write(h5,'/rot/ang2vec',ang2vec(phi,theta));
    [p,t]=vec2ang(ang2vec(phi,theta));
    oracle_write(h5,'/rot/vec2ang',[p; t]);
    fields=[0 100 350 1200];
    oracle_write(h5,'/units/mt2mhz',mt2mhz(fields));
    oracle_write(h5,'/units/mt2mhz_g',mt2mhz(fields,2.1));
    oracle_write(h5,'/units/mhz2mt',mhz2mt([100 9500 35000],2.0023));
    oracle_write(h5,'/units/wn2mhz',unitconvert([1 10 100],'cm^-1->MHz'));
    oracle_write(h5,'/units/wn2mt',unitconvert([1 10 100],'cm^-1->mT',2.05));
    oracle_write(h5,'/units/k2mhz',unitconvert([0.1 1 10],'K->MHz'));
    oracle_write(h5,'/units/ev2wn',unitconvert([0.001 0.01],'eV->cm^-1'));
    nucs='1H,2H,13C,14N,15N,17O,19F,31P,55Mn,63Cu,65Cu,51V,57Fe,195Pt,29Si';
    [I,gn,qm,abund]=nucdata(nucs);
    oracle_write(h5,'/nuc/I',I);
    oracle_write(h5,'/nuc/gn',gn);
    oracle_write(h5,'/nuc/qm',qm);
    oracle_write(h5,'/nuc/abund',abund);
    inputs=struct('angles',angles,'phi',phi,'theta',theta,'fields',fields,'nucs',nucs);
    es_meta(case_dir,dump_file,inputs);
end

end

% The spin systems of the cases
function cases=spin_system_cases()
cases={};

% A nitroxide with tilted g and A frames and a 14N quadrupole (eeqQ/h, eta syntax)
Sys=struct('g',[2.0088 2.0061 2.0027],'gFrame',[10 20 30]*pi/180, ...
           'Nucs','14N','A',[16 16 86],'AFrame',[0.3 0.7 1.1],'Q',[-0.5 0.2],'lwpp',0.3);
cases{end+1}=struct('name','nitroxide_frames','Sys',Sys);

% Cu(II) with two 14N: axial g, hyperfine frames per nucleus, quadrupoles (72 states)
Sys=struct('S',1/2,'g',[2.05 2.2],'Nucs','63Cu,14N,14N', ...
           'A',[50 50 500; 40 40 44; 38 38 42],'AFrame',[0 0 0; 0.2 0.4 0.6; -0.5 1.3 2.2], ...
           'Q',[0 0; -1.2 0.3; 1.0 0.5],'lwpp',1);
cases{end+1}=struct('name','cu_2n','Sys',Sys);

% Mn(II): S=5/2 with D, E, a tilted D frame, a rank-4 Stevens term, and a 55Mn hyperfine
Sys=struct('S',5/2,'D',[1000 100],'DFrame',[0.2 0.5 0.9], ...
           'B4',[1 2 -3 0.5 10 -0.5 3 -2 1],'Nucs','55Mn','A',250);
cases{end+1}=struct('name','mn_s52_stevens','Sys',Sys);

% A radical pair: two S=1/2 with J, dipolar, antisymmetric exchange, biquadratic term, one proton
Sys=struct('S',[1/2 1/2],'g',[2.0023 2.0023 2.0023; 2.00 2.01 2.02],'gFrame',[0 0 0; 0.3 0.6 0.9], ...
           'J',5,'dip',[-10 2],'dvec',[0.3 -0.2 0.1],'eeFrame',[0.1 0.4 0.7],'ee2',2, ...
           'Nucs','1H','A',[10 10 12 0 0 0]);
cases{end+1}=struct('name','radical_pair','Sys',Sys);

% A triplet with full g, D, A and Q matrices, two nuclei with nn coupling, shielding and gnscale
gfull=[2.01 0.002 -0.001; 0.002 2.02 0.003; -0.001 0.003 2.05];
Dfull=[-300 20 -10; 20 -350 15; -10 15 650];
Afull=[12 1 0; 1 13 2; 0 2 30; 5 0.5 0; 0.5 5 0; 0 0 6];
Qfull=[-0.4 0.05 0; 0.05 -0.3 0.02; 0 0.02 0.7; 0 0 0; 0 0 0; 0 0 0];
Sys=struct('S',1,'g',gfull,'D',Dfull,'Nucs','14N,1H','A',Afull,'Q',Qfull, ...
           'nn',[0.05 0.05 -0.1],'nnFrame',[0.2 0.3 0.4],'sigma',[1 1 1; 0.9 1.0 1.1], ...
           'sigmaFrame',[0 0 0; 0.5 0.6 0.7],'gnscale',[1 0.98]);
cases{end+1}=struct('name','triplet_full_tensors','Sys',Sys);

% Not a case: a system listing spin-zero nuclei with one A row per nucleus
% ('12C,13C,16O', A 3x3) errors in EasySpin 6.0.15's ham_hf, which takes
% the row count of the unprocessed A for the full-matrix test while the
% nuclei were already removed. easyspyn handles it; tests/easyspyn covers it.

end
