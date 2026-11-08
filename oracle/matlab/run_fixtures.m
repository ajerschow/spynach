% Start script for oracle fixture generation. Puts Spinach and the oracle
% harness on the Matlab path, runs every dump script, and makes the new
% fixtures read-only. Edit the settings block below, then run it from
% Matlab with
%
%                       run('<repo>/oracle/matlab/run_fixtures.m')
%
% or from a shell with oracle/run_fixtures.sh, which starts Matlab in
% batch mode. Run it from the checkout that holds the dump scripts you
% want (repo_dir below). Dump scripts skip cases that already exist, so
% re-running after adding a dump script only produces the new cases.
% Fixtures go to <fixtures_root>/<spinach tag>/<case>/. The
% console output, including per-stage progress and any error, is also
% written to <fixtures_root>/<spinach tag>/run_<UTC time>.log, and the
% Spinach report output of each case to <case>/spinach.log.

%% Settings - edit these
repo_dir='/Users/alexej/Documents/PROG/spynach';          % spynach checkout holding the dump scripts to run
spinach_dir='/Users/alexej/Documents/PROG/spynach-spinach';
fixtures_root='/Users/alexej/Documents/PROG/spynach-fixtures';
dump_scripts={'dump_hilb_pa','dump_liouv_setup','dump_liouv_pa','dump_aniso','dump_relax','dump_mas','dump_seq','dump_scalar','dump_iserstep'};       % cases that already exist are skipped

%% Nothing below needs editing

% Locate the harness inside the repository
oracle_dir=fullfile(repo_dir,'oracle','matlab');
if ~isfolder(oracle_dir)
    error(['oracle directory not found: ' oracle_dir ', check repo_dir.']);
end
if ~isfolder(fullfile(spinach_dir,'kernel'))
    error(['Spinach kernel not found under ' spinach_dir ', check spinach_dir.']);
end

% Add Spinach directories, as fix_path.m does
addpath(genpath(fullfile(spinach_dir,'etc')),'-begin');
addpath(genpath(fullfile(spinach_dir,'experiments')),'-begin');
addpath(genpath(fullfile(spinach_dir,'interfaces')),'-begin');
addpath(genpath(fullfile(spinach_dir,'kernel')),'-begin');

% Add the harness, with shadow files ahead of Spinach
addpath(fullfile(oracle_dir,'lib'),fullfile(oracle_dir,'dumps'),'-begin');
addpath(genpath(fullfile(oracle_dir,'shadow')),'-begin');

% Make sure no other Spinach copy takes precedence
create_path=which('create');
if ~startsWith(create_path,fullfile(spinach_dir,'kernel'))
    error(['create() resolves to ' create_path ', not to the Spinach copy in ' spinach_dir '.']);
end

% Determine the Spinach tag and warn about local modifications
[status,spinach_tag]=system(['git -C "' spinach_dir '" describe --tags --always']);
if status~=0
    error(['Spinach at ' spinach_dir ' is not a git working tree: ' spinach_tag]);
end
spinach_tag=strtrim(spinach_tag);
[~,porcelain]=system(['git -C "' spinach_dir '" status --porcelain']);
status_lines=splitlines(strtrim(porcelain)); status_lines(cellfun(@isempty,status_lines))=[];
if any((~startsWith(status_lines,'??'))|endsWith(strtrim(status_lines),'.m'))
    warning('Spinach working tree has local modifications, fixtures will be flagged as dirty.');
end

% Create the output directory for this tag
out_root=fullfile(fixtures_root,spinach_tag);
if ~isfolder(out_root), mkdir(out_root); end

% Log all console output to a file
log_file=fullfile(out_root,['run_' char(datetime('now','TimeZone','UTC',...
                  'Format','yyyyMMdd''T''HHmmss''Z''')) '.log']);
diary(log_file); diary on;
fprintf('Logging to %s\n',log_file);
fprintf('Spinach %s, tag %s\nHarness %s\nWriting fixtures to %s\n',...
        spinach_dir,spinach_tag,repo_dir,out_root);

% Run the dump scripts and protect their output
try
    for n=1:numel(dump_scripts)
        fprintf('Running %s...\n',dump_scripts{n});
        existing=dir(out_root); existing={existing([existing.isdir]).name};
        feval(dump_scripts{n},out_root);
        created=dir(out_root); created={created([created.isdir]).name};
        new_dirs=setdiff(created,existing);
        for k=1:numel(new_dirs)
            fileattrib(fullfile(out_root,new_dirs{k}),'-w','a','s');
            fprintf('  wrote %s (read-only)\n',new_dirs{k});
        end
    end
    fprintf('Done. Fixtures are in %s\n',out_root);
catch run_error
    fprintf(2,'FAILED: %s\n',getReport(run_error,'extended','hyperlinks','off'));
    diary off; rethrow(run_error);
end
diary off;

