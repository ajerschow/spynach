% Writes the meta.json provenance record of an EasySpin fixture case:
% the EasySpin installation and version, the Matlab version, SHA-256
% hashes of the dump script and the harness files, and the inputs of the
% case (the spin system and the parameters), which the Python parity
% tests read back to rebuild the case. Syntax:
%
%                 es_meta(case_dir,dump_file,inputs)
%
% Parameters:
%
%    case_dir    - fixture case directory, must already exist and
%                  must not contain a meta.json file
%
%    dump_file   - full path to the .m file of the dump script
%
%    inputs      - structure with the inputs of the case (Sys, fields,
%                  orientations, ...) that jsonencode() can serialise
%
% Outputs:
%
%    this function writes case_dir/meta.json
%
% The EasySpin version is read from the name of the installation folder
% (easyspin-<version>), since the release ships no readable version
% function.

function es_meta(case_dir,dump_file,inputs)

% Check consistency
grumble(case_dir,dump_file,inputs);

% Locate EasySpin and the harness
easyspin_dir=fileparts(which('pepper'));
lib_dir=fileparts(mfilename('fullpath'));
shared_lib=fullfile(fileparts(fileparts(lib_dir)),'lib');
tokens=regexp(easyspin_dir,'easyspin-(\d+\.\d+\.\d+)','tokens','once');
if isempty(tokens)
    easyspin_version='unknown';
else
    easyspin_version=tokens{1};
end

% Assemble the record
meta.schema_version=1;
meta.created_utc=char(datetime('now','TimeZone','UTC',...
                               'Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
meta.easyspin_dir=easyspin_dir;
meta.easyspin_version=easyspin_version;
meta.matlab_version=version();
meta.dump_script=struct('path',dump_file,'sha256',file_sha256(dump_file));
meta.harness=struct('oracle_write',file_sha256(fullfile(shared_lib,'oracle_write.m')),...
                    'es_meta',file_sha256(fullfile(lib_dir,'es_meta.m')));
meta.inputs=inputs;

% Write the record
file_id=fopen(fullfile(case_dir,'meta.json'),'w');
fprintf(file_id,'%s\n',jsonencode(meta,'PrettyPrint',true));
fclose(file_id);

end

% SHA-256 hex digest of a file
function hex_digest=file_sha256(file_path)
file_id=fopen(file_path,'r');
file_bytes=fread(file_id,Inf,'*uint8');
fclose(file_id);
hasher=java.security.MessageDigest.getInstance('SHA-256');
digest=typecast(hasher.digest(file_bytes),'uint8');
hex_digest=lower(reshape(dec2hex(digest,2).',1,[]));
end

% Consistency enforcement
function grumble(case_dir,dump_file,inputs)
if (~ischar(case_dir))||(~isfolder(case_dir))
    error('case_dir must be the name of an existing directory.');
end
if isfile(fullfile(case_dir,'meta.json'))
    error('meta.json already exists in case_dir, fixtures are never overwritten.');
end
if (~ischar(dump_file))||(~isfile(dump_file))
    error('dump_file must be the name of an existing file.');
end
if ~isstruct(inputs)
    error('inputs must be a structure.');
end
end
