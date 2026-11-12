% EasySpin reference data for easyspyn's eprload: Bruker BES3T files written
% by eprsave from a pepper spectrum (1-D real and complex, a nonlinear axis,
% 2-D real and complex, a nonlinear y axis), BES3T files with the standard
% parameter layer keys the scaling options need, hand-written Bruker ESP and
% WinEPR files for every axis rule, flag and scaling key, and any file pair
% found in eprload_files/ next to this script (real instrument files). Each
% file is loaded back with eprload, unscaled and with each scaling given,
% and its abscissa, data and parameters are stored. Syntax:
%
%                       esdump_eprload(version_dir)
%
% version_dir is <fixtures_root>/<easyspin version>. One directory per case
% is written below it with the data files themselves next to data.h5, so
% that the Python side reads the same bytes; the parameter structure that
% eprload returns is stored in meta.json. A scaling that errors in EasySpin
% is recorded as /scaled/<letters>/error = 1. Cases whose directory already
% exists are skipped, never overwritten.

function esdump_eprload(version_dir)

dump_file=[mfilename('fullpath') '.m'];
B=linspace(3300,3400,256);                           % G
Sys=struct('g',[2.0 2.1],'lw',3);
Exp=struct('mwFreq',9.5,'Range',[330 340],'nPoints',256);
[~,spc]=pepper(Sys,Exp,struct('FuzzLevel',0));
spc=spc(:).';
t=linspace(0,5,8);
Z=spc(:)*exp(-t);                                    % 256 x 8, x along the first dimension
spl=sprintf(['EXPT\tCW\nAVGS\t4\nRCAG\t30\nSPTP\t0.04096\nMWPW\t0.002\nSTMP\t300\n' ...
             'XUNI\tG\nXNAM\tField\nB0MA\t1e-4\nRCHM\t1\n']);

% BES3T files written by eprsave
%-------------------------------------------------------------------------------
bes3t(version_dir,dump_file,'el_1d_real',B,spc,'axial powder',9.5,'',{});
bes3t(version_dir,dump_file,'el_1d_complex',B,spc.*(1-0.5i),'',NaN,'',{});
bes3t(version_dir,dump_file,'el_1d_nonlinear',3300+100*linspace(0,1,256).^2,spc,'',NaN,'',{});
bes3t(version_dir,dump_file,'el_2d_real',{B,t},Z,'two dimensions',9.5,'',{});
bes3t(version_dir,dump_file,'el_2d_complex',{B,t},Z*(0.3+1i),'',NaN,'',{});
bes3t(version_dir,dump_file,'el_2d_nonlinear_y',{B,logspace(0,1,8)},Z,'',NaN,'',{});
bes3t(version_dir,dump_file,'el_scaling',B,spc,'scaled',9.5,spl,{'n','P','G','T','c','nPGTc'});
bes3t(version_dir,dump_file,'el_scaling_prescaled',B,spc,'',9.5,[spl sprintf('SctNorm\tTrue\n')],{'P','n'});

% ESP and WinEPR files written by hand
%-------------------------------------------------------------------------------
data=round(spc/max(abs(spc))*1e6);
Zi=round(Z/max(abs(Z(:)))*1e6);
esp(version_dir,dump_file,'el_esp_hcf',{'ANZ','256';'HCF','3350';'HSW','100';'MF','9.5';'JSD','4';'RRG','2e4';'MP','20';'TE','298';'RCT','40.96';'RMA','2';'RHA','1'},data,false,{'n','G','P','T','c','nGPTc'});
esp(version_dir,dump_file,'el_esp_gst',{'ANZ','256';'GST','3300';'GSI','100'},data,false,{});
esp(version_dir,dump_file,'el_esp_both',{'ANZ','256';'HCF','1';'HSW','1';'GST','3300';'GSI','100'},data,false,{});
esp(version_dir,dump_file,'el_esp_endor',{'ANZ','256';'JEX','ENDOR';'HCF','1';'HSW','1';'GST','1';'GSI','20'},data,false,{});
esp(version_dir,dump_file,'el_esp_time',{'ANZ','256';'JEX','Time-Sweep';'RCT','40.96'},data,false,{});
esp(version_dir,dump_file,'el_esp_norange',{'ANZ','256';'HCF','3350'},data,false,{});
esp(version_dir,dump_file,'el_esp_res',{'ANZ','512';'RES','128';'XPLS','256';'HCF','3350';'HSW','100'},data,false,{});
esp(version_dir,dump_file,'el_esp_complex',{'ANZ','512';'JSS','16';'HCF','3350';'HSW','100'},data+1i*round(data/3),false,{});
esp(version_dir,dump_file,'el_esp_2d',{'JSS','4096';'SSX','256';'SSY','8';'ANZ','2048';'HCF','3350';'HSW','100'},Zi,false,{});
esp(version_dir,dump_file,'el_esp_emx2d',{'JSS','4096';'SSX','256';'SSY','8';'ANZ','2048';'XXLB','3300';'XXWI','100';'XYLB','0';'XYWI','21'},Zi,false,{});
esp(version_dir,dump_file,'el_esp_powersweep',{'JSS','4096';'SSX','256';'SSY','8';'ANZ','2048';'JEY','mw-power-sweep';'MP','200';'XXLB','3300';'XXWI','100';'XYLB','0';'XYWI','21'},Zi,false,{'P'});
esp(version_dir,dump_file,'el_esp_slice',{'RES','256';'REY','8';'HCF','3350';'HSW','100'},data,false,{});
esp(version_dir,dump_file,'el_winepr',{'RES','256';'HCF','3350';'HSW','100';'MF','9.78';'RMA','2';'TE','298';'RHA','1'},spc,true,{});

% Real instrument files dropped into eprload_files/ (any BES3T or ESP pair)
%-------------------------------------------------------------------------------
files_dir=fullfile(fileparts(dump_file),'eprload_files');
if isfolder(files_dir)
    listing=[dir(fullfile(files_dir,'*.DSC'));dir(fullfile(files_dir,'*.dsc'));...
             dir(fullfile(files_dir,'*.par'));dir(fullfile(files_dir,'*.PAR'))];
    for k=1:numel(listing)
        [~,stem,ext]=fileparts(listing(k).name);
        user_file(version_dir,dump_file,['el_user_' lower(stem)],files_dir,stem,ext);
    end
end

end

% A BES3T pair from eprsave, with extra descriptor lines appended
function bes3t(version_dir,dump_file,name,x,y,titl,mwfreq,extra,scalings)
case_dir=fullfile(version_dir,name);
if isfolder(case_dir), fprintf('  skip %s (exists)\n',name); return; end
fprintf('  case %s\n',name); mkdir(case_dir);
file=fullfile(case_dir,name);
if isnan(mwfreq), eprsave(file,x,y,titl); else, eprsave(file,x,y,titl,mwfreq); end
if ~isempty(extra)
    fid=fopen([file '.DSC'],'a'); fprintf(fid,'%s',extra); fclose(fid);
end
record(case_dir,dump_file,file,'BES3T',[name '.DSC'],scalings);
end

% An ESP (big-endian int32) or WinEPR (little-endian float32, DOS Format) pair
function esp(version_dir,dump_file,name,keys,data,winepr,scalings)
case_dir=fullfile(version_dir,name);
if isfolder(case_dir), fprintf('  skip %s (exists)\n',name); return; end
fprintf('  case %s\n',name); mkdir(case_dir);
file=fullfile(case_dir,name);
fid=fopen([file '.par'],'w');
if winepr, fprintf(fid,'DOS  Format\n'); end
for k=1:size(keys,1), fprintf(fid,'%s  %s\n',keys{k,1},keys{k,2}); end
fclose(fid);
if winepr
    fid=fopen([file '.spc'],'w','ieee-le'); fmt='float32';
else
    fid=fopen([file '.spc'],'w','ieee-be'); fmt='int32';
end
if isreal(data)
    v=data(:);
else
    v=[real(data(:)) imag(data(:))].'; v=v(:);
end
fwrite(fid,v,fmt); fclose(fid);
record(case_dir,dump_file,file,'ESP',[name '.par'],scalings);
end

% A user's file pair copied into the case directory and loaded with every scaling letter
function user_file(version_dir,dump_file,name,files_dir,stem,ext)
case_dir=fullfile(version_dir,name);
if isfolder(case_dir), fprintf('  skip %s (exists)\n',name); return; end
fprintf('  case %s\n',name); mkdir(case_dir);
copyfile(fullfile(files_dir,[stem '.*']),case_dir);
if any(strcmpi(ext,{'.DSC','.DTA'})), format='BES3T'; else, format='ESP'; end
try
    record(case_dir,dump_file,fullfile(case_dir,stem),format,[stem ext],{'n','P','G','T','c'});
catch err
    fprintf('  %s: eprload failed (%s); the case directory is removed\n',name,err.message);
    rmdir(case_dir,'s');
end
end

% Load the file back and store the abscissa, the data, the scaled data and the parameters
function record(case_dir,dump_file,file,format,file_name,scalings)
h5=fullfile(case_dir,'data.h5');
[x,y,pars]=eprload(file);
if iscell(x)
    for k=1:numel(x), oracle_write(h5,sprintf('/load/x%d',k),x{k}); end
    oracle_write(h5,'/load/naxes',numel(x));
else
    oracle_write(h5,'/load/x',x); oracle_write(h5,'/load/naxes',1);
end
if iscell(y)
    for k=1:numel(y), oracle_write(h5,sprintf('/load/y%d',k),y{k}); end
    oracle_write(h5,'/load/nvalues',numel(y));
else
    oracle_write(h5,'/load/y',y); oracle_write(h5,'/load/nvalues',1);
end
errors={};
for k=1:numel(scalings)
    s=scalings{k};
    try
        [~,ys]=eprload(file,s);
        if iscell(ys)
            for j=1:numel(ys), oracle_write(h5,sprintf('/scaled/%s/y%d',s,j),ys{j}); end
        else
            oracle_write(h5,['/scaled/' s '/y'],ys);
        end
    catch err
        oracle_write(h5,['/scaled/' s '/error'],1);
        errors{end+1}=err.message; %#ok<AGROW>
    end
end
es_meta(case_dir,dump_file,struct('format',format,'file',file_name,'scalings',{scalings},...
                                  'errors',{errors},'params',pars));
end
