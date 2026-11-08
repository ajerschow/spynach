% Oracle dump for the state-dependent integrator iserstep: the one-proton
% Hilbert-space case of Spinach's tests/kernel/test_dynamic_iserstep_highorder.m
% (a Hermitian density matrix with coherences under a mildly nonlinear,
% non-commuting generator), one step of each method and the two-half-step
% DP8 reference, written to <out_root>/iserstep_hilb_1spin/data.h5 with a
% meta.json provenance record. Syntax:
%
%                       dump_iserstep(out_root)
%
% Parameters:
%
%    out_root - fixture directory for the current Spinach tag, e.g.
%               /fixtures/<spinach-tag>; the case is skipped if its
%               directory already exists there, never overwritten
%
% Outputs:
%
%    this function writes fixture files, see FIXTURE_REQUESTS.md and
%    oracle/README.md for the layout
%
% Note: oracle/matlab/lib must be on the Matlab path, and Spinach must
%       be on the path with no modifications. The case also writes
%       /probe variables with known content that verify the HDF5 lay-
%       out conventions on the Python side. tests/unit/test_iserstep.py
%       recomputes every step from the same inputs.

function dump_iserstep(out_root)

% Check consistency
grumble(out_root);

% Never overwrite existing fixtures
dump_file=[mfilename('fullpath') '.m'];
case_dir=fullfile(out_root,'iserstep_hilb_1spin');
if isfolder(case_dir)
    fprintf('%s [iserstep_hilb_1spin] exists, skipped\n',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')));
    return;
end
mkdir(case_dir); h5_file=fullfile(case_dir,'data.h5');
progress('started, writing to %s',case_dir);

% Layout convention probes with known content
oracle_write(h5_file,'/probe/dense',reshape(1:24,[2 3 4])+1i*reshape(101:124,[2 3 4]));
oracle_write(h5_file,'/probe/column',[1; 2; 3]);
oracle_write(h5_file,'/probe/sparse',sparse([1 3 2],[2 1 4],[5+1i 7 -2i],3,4));
oracle_write(h5_file,'/probe/cell',{[1 2],sparse(eye(2)); [],7});

% One quiet proton in Hilbert space, as the Spinach test builds it
progress('create and basis');
sys.magnet=0; sys.isotopes={'1H'}; sys.output=fullfile(case_dir,'spinach.log');
sys.disable={'hygiene'}; % existentials() would reject the isworkernode.m shadow
inter.zeeman.scalar={0};
bas.formalism='zeeman-hilb'; bas.approximation='none';
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

% The density matrix and the nonlinear generator of the Spinach test
S=pauli(2); Sx=S.x; Sy=S.y; Sz=S.z;
rho=[0.7,0.2+0.1i;0.2-0.1i,0.3];
H0=0.31*Sx+0.17*Sz;
H1=-0.23*Sy+0.05*Sx;
H2=0.11*Sx-0.07*Sz;
Lfun=@(t,state)H0+sin(4*t)*H1+0.2*real(trace(Sz*state))*H2;
dt=5e-3;
oracle_write(h5_file,'/inputs/rho',rho);
oracle_write(h5_file,'/inputs/H0',H0);
oracle_write(h5_file,'/inputs/H1',H1);
oracle_write(h5_file,'/inputs/H2',H2);
oracle_write(h5_file,'/inputs/dt',dt);

% One step of every method
progress('steps');
methods={'PWCL','LG2','LG4','LG4A','RKMK4','RKMK-DP5','RKMK-DP8','RKMK-RKF45'};
for n=1:numel(methods)
    rho_obs=iserstep(spin_system,{Lfun,0,methods{n}},rho,dt);
    oracle_write(h5_file,['/steps/' strrep(methods{n},'-','_')],full(rho_obs));
end

% The refined DP8 reference of the Spinach test: two half steps
rho_ref=iserstep(spin_system,{Lfun,0,'RKMK-DP8'},rho,dt/2);
rho_ref=iserstep(spin_system,{Lfun,dt/2,'RKMK-DP8'},rho_ref,dt/2);
oracle_write(h5_file,'/steps/reference',full(rho_ref));

% Provenance record
inputs=struct('sys',sys,'inter',inter,'bas',bas,'rho_real',real(rho),'rho_imag',imag(rho),'dt',dt,...  % jsonencode takes no complex values
              'generator','H0+sin(4t)*H1+0.2*real(trace(Sz*rho))*H2',...
              'H0','0.31*Sx+0.17*Sz','H1','-0.23*Sy+0.05*Sx','H2','0.11*Sx-0.07*Sz',...
              'methods',{methods});
oracle_meta(case_dir,dump_file,spin_system,inputs);

% Close the Spinach log
fclose(spin_system.sys.output);
progress('finished');

end

% Timestamped progress line on the console
function progress(varargin)
fprintf('%s [iserstep_hilb_1spin] %s\n',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')),sprintf(varargin{:}));
end

% Consistency enforcement
function grumble(out_root)
if (~ischar(out_root))||(~isfolder(out_root))
    error('out_root must be the name of an existing directory.');
end
end
