% Comparison case: Lie-group product quadratures on a chirped oscillator
% with radiation damping, examples/fundamentals/quadratures/product_quadratures_2.m
% with timed stages. The reference trajectory, the grid sizes and the
% error table are returned as arrays and the two subplots as two figures.
% Stage labels must match product_quadratures_2_spynach.py. Syntax:
%
%     [data,figs]=product_quadratures_2_spinach(timer,options)
%
% Parameters:
%
%     timer   - cmp_timer object
%
%     options - struct, may be empty; np_ref (default 4096, points of
%               the RKMK-DP8 reference trajectory) and grids (default
%               10, grids between 2^8 and 2^11 points); gpu is ignored
%               (no spin system is propagated)
%
% Outputs:
%
%     data    - mu_traj (3 x np_ref), np (grid sizes), bench (grids x
%               7 relative errors: LP, LG-2, LG-4, LG-4A, RKMK4,
%               RKMK-DP5, RKMK-DP8)
%
%     figs    - trajectory, convergence

function [data,figs]=product_quadratures_2_spinach(timer,options)

if nargin<2, options=struct(); end
np_ref=cmp_option(options,'np_ref',2^12);
grids=cmp_option(options,'grids',10);

% Set system parameters
cr=2*pi*400;   % 400 Hz/s chirp rate
r1=10; r2=10;  % 10 Hz relaxation
rrd=40;        % 40 Hz radiation damping

% Bootstrap the object, as interfaces/bootstrap.m does (a ghost spin in
% Liouville space that carries the tolerances; inlined so that the case
% runs on a Spinach checkout without the interfaces directory)
sys.magnet=0; sys.isotopes={'G'}; sys.disable={'hygiene'};
inter.zeeman.matrix=cell(1); inter.coupling.matrix=cell(1);
bas.formalism='sphten-liouv'; bas.approximation='none';
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

% Make Bloch-Maxwell generator
G=@(t,mu)(-1i*[r2 cr*t 0; -cr*t r2 0; 0 0 r1]...
          -1i*rrd*[mu(3) 0 0; 0 mu(3) 0; -mu(1) -mu(2) 0]);

% Set initial magnetisation
mu0=euler2dcm(0,pi*178/180,0)*[0 0 1]';

% Run reference RKMK-DP8 simulation and keep the trajectory
timer.start('reference');
dt_ref=0.5/(np_ref-1); mu_traj=zeros(3,np_ref);
mu_traj(:,1)=mu0;
for n=2:np_ref
    t_curr=(n-2)*dt_ref;
    mu_traj(:,n)=iserstep(spin_system,{G,t_curr,'RKMK-DP8'},mu_traj(:,n-1),dt_ref);
end
mu_ref=mu_traj(:,end);
timer.stop();

% Benchmark arrays
np=ceil(2.^linspace(8,11,grids)); bench=zeros(numel(np),7);
methods={'PWCL','LG2','LG4','LG4A','RKMK4','RKMK-DP5','RKMK-DP8'};

% Benchmarking loop
timer.start('benchmark');
for k=1:numel(np)
    dt=0.5/(np(k)-1);
    for m=1:numel(methods)
        mu=mu0;
        for n=2:np(k)
            t_curr=(n-2)*dt;
            mu=iserstep(spin_system,{G,t_curr,methods{m}},mu,dt);
        end
        bench(k,m)=norm(mu-mu_ref)/norm(mu_ref);
    end
end
timer.stop();

% Plotting, as two figures
timer.start('plot');
time_axis=linspace(0,0.5,np_ref);
figs.trajectory=figure('Visible','off');
plot(time_axis,mu_traj); grid on; axis tight;
xlabel('time, seconds'); ylabel('magnetisation, a.u.');
legend({'x-magnetisation','y-magnetisation','z-magnetisation'},'Location','Best');
figs.convergence=figure('Visible','off');
loglog(np',bench); grid on; xlim tight; ylim([1e-7 1e0]);
xlabel('number of points in the time grid');
ylabel('||difference||/||exact||');
legend({'LP','LG-2','LG-4','LG-4A','RKMK4','RKMK-DP5','RKMK-DP8'},'Location','Best');
timer.stop();

% Outputs
data.mu_traj=mu_traj;
data.np=np;
data.bench=bench;

end
