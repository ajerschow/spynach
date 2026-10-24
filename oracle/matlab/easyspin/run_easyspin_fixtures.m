% Start script for EasySpin fixture generation (the reference data of the
% epreasy parity tests). Puts EasySpin and the harness on the Matlab path,
% runs every EasySpin dump script, and makes the new fixtures read-only.
% Edit the settings block below, then run it from Matlab with
%
%            run('<repo>/oracle/matlab/easyspin/run_easyspin_fixtures.m')
%
% or from a shell with oracle/run_easyspin_fixtures.sh, which starts Matlab
% in batch mode. Dump scripts skip cases that already exist, so re-running
% after adding a dump script only produces the new cases. Fixtures go to
% <fixtures_root>/<easyspin version>/<case>/, separate from the Spinach
% fixtures. The console output is also written to
% <fixtures_root>/<easyspin version>/run_<UTC time>.log.
%
% EasySpin is only read: the installed release is put on the path, nothing
% is written into it.

%% Settings - edit these
repo_dir='/Users/alexej/Documents/PROG/spynach';                       % spynach checkout holding the dump scripts
easyspin_dir='/Users/alexej/Documents/MATLAB/easyspin-6.0.15/easyspin'; % installed EasySpin release (read-only)
easyspin_version='6.0.15';
fixtures_root='/Users/alexej/Documents/PROG/easyspin-fixtures';
dump_scripts={'esdump_ham','esdump_resfields','esdump_pepper','esdump_esfit','esdump_endor'};                                           % cases that already exist are skipped

%% Nothing below needs editing

% Locate the harness inside the repository
es_dir=fullfile(repo_dir,'oracle','matlab','easyspin');
if ~isfolder(es_dir)
    error(['EasySpin oracle directory not found: ' es_dir ', check repo_dir.']);
end
if ~isfile(fullfile(easyspin_dir,'pepper.m'))
    error(['EasySpin not found under ' easyspin_dir ', check easyspin_dir.']);
end

% Add EasySpin and the harness (the shared HDF5 writer, the meta writer, the dumps)
addpath(easyspin_dir,'-begin');
addpath(fullfile(repo_dir,'oracle','matlab','lib'),'-begin');
addpath(fullfile(es_dir,'lib'),'-begin');
addpath(fullfile(es_dir,'dumps'),'-begin');

% Output directory and run log
version_dir=fullfile(fixtures_root,easyspin_version);
if ~isfolder(version_dir), mkdir(version_dir); end
stamp=char(datetime('now','TimeZone','UTC','Format','yyyyMMdd''T''HHmmss''Z'''));
diary(fullfile(version_dir,['run_' stamp '.log']));
fprintf('EasySpin at %s\nMATLAB %s\nFixtures to %s\n',which('pepper'),version,version_dir);
before=dir(version_dir); before=before([before.isdir]); before={before.name};

% Run the dump scripts
for n=1:numel(dump_scripts)
    fprintf('\n=== %s ===\n',dump_scripts{n});
    try
        feval(dump_scripts{n},version_dir);
    catch err
        fprintf('ERROR in %s: %s\n',dump_scripts{n},err.message);
        diary off;
        rethrow(err);
    end
end

% Make the new fixture directories read-only, as the Spinach harness does
after=dir(version_dir); after=after([after.isdir]); after={after.name};
new_cases=setdiff(after,before);
for n=1:numel(new_cases)
    fileattrib(fullfile(version_dir,new_cases{n}),'-w','','s');
end
fprintf('\n%d new case(s): %s\n',numel(new_cases),strjoin(new_cases,', '));
diary off;
