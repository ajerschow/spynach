% Runs the Spinach side of the comparison cases in comparison/cases:
% each case with timed stages under the MATLAB profiler, and writes
% comparison/runs/<case>[-gpu][-<tag>]/spinach/ with data.mat, timings.json,
% profile.json, meta.json, figure_<name>.png and console.log. Edit the
% settings block below, then run it from MATLAB:
%
%     run('<repo>/comparison/matlab/run_comparison.m')
%
% or from a shell with comparison/run_spinach.sh (MATLAB batch mode).
% Then run the spynach side (comparison/run_spynach.py) and build the
% report (comparison/report.py); see comparison/README.md.

%% Settings
repo_dir='/Users/alexej/Documents/PROG/spynach';             % spynach checkout holding comparison/cases
spinach_dir='/Users/alexej/Documents/PROG/spynach-spinach';  % Spinach checkout
cases={'noesy_strychnine','zulf_pf6_1d','maz_noesy_2','li_exchange_equilibrium','product_quadratures_2'};                 % case directories to run
gpu=false;                                                  % true: sys.enable={'gpu'} in every case, results in runs/<case>-gpu[-<tag>]/
tag='';                                                     % run name suffix: runs/<case>[-gpu]-<tag>/; '' for runs/<case>[-gpu]/
options=struct();                                           % per-case options, e.g. options.noesy_strychnine.npoints=64;
top_n=40;                                                   % profiler functions to keep

%% Paths
comparison_dir=fullfile(repo_dir,'comparison');
if ~isfolder(fullfile(spinach_dir,'kernel'))
    error(['Spinach kernel not found under ' spinach_dir ', check spinach_dir.']);
end
addpath(genpath(fullfile(spinach_dir,'kernel')),'-begin');
addpath(genpath(fullfile(spinach_dir,'etc')),'-begin');
addpath(genpath(fullfile(spinach_dir,'experiments')),'-begin');
addpath(genpath(fullfile(spinach_dir,'interfaces')),'-begin');
addpath(fullfile(comparison_dir,'matlab','lib'),'-begin');
create_path=which('create');
if ~startsWith(create_path,fullfile(spinach_dir,'kernel'))
    error(['create() resolves to ' create_path ', not to the Spinach copy in ' spinach_dir '.']);
end

%% Cases
for n=1:numel(cases)
    case_name=cases{n};
    case_dir=fullfile(comparison_dir,'cases',case_name);
    addpath(case_dir,'-begin');
    run_name=case_name;
    if gpu, run_name=[run_name '-gpu']; end
    if ~isempty(tag), run_name=[run_name '-' tag]; end
    out_dir=fullfile(comparison_dir,'runs',run_name,'spinach');
    if ~isfolder(out_dir), mkdir(out_dir); end
    case_options=cmp_option(options,case_name,struct());
    case_options.gpu=gpu;
    diary(fullfile(out_dir,'console.log')); diary on;
    fprintf('== %s (Spinach %s) -> %s\n',case_name,spinach_dir,out_dir);
    timer=cmp_timer();
    profile off; profile clear; profile on;
    wall=tic;
    try
        [data,figs]=feval([case_name '_spinach'],timer,case_options);
    catch err
        profile off; diary off;
        rethrow(err);
    end
    total_seconds=toc(wall);
    profile off;
    info=profile('info');
    save(fullfile(out_dir,'data.mat'),'-struct','data','-v7');
    cmp_json(fullfile(out_dir,'timings.json'),struct('stages',{timer.records},'total_s',total_seconds));   % cell: a struct array would broadcast
    cmp_profile(fullfile(out_dir,'profile.json'),info,top_n);
    cmp_json(fullfile(out_dir,'meta.json'),cmp_meta(case_name,case_options));
    names=fieldnames(figs);
    for k=1:numel(names)
        exportgraphics(figs.(names{k}),fullfile(out_dir,['figure_' names{k} '.png']),'Resolution',150);
        close(figs.(names{k}));
    end
    fprintf('%-32s %8.1f s\n','total',total_seconds);
    diary off;
end
