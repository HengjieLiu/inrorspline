% Reproduce pTVreg DIR-Lab 4DCT pairwise registration.
%
% Default run:
%   case01, T50 -> T00, published 1mm pTVreg protocol, explicit CPU LCC.
%
% Optional inputs can be supplied by defining these variables before run():
%   case_id, output_root, raw_root, reference_root, protocol_name, metric_name

if ~exist('case_id', 'var') || isempty(case_id)
    case_id = 1;
end
if ~exist('output_root', 'var') || isempty(output_root)
    output_root = getenv('INRORSPLINE_PTVREG_OUTPUT_ROOT');
    if isempty(output_root)
        error('Set output_root or INRORSPLINE_PTVREG_OUTPUT_ROOT before running.');
    end
end
if ~exist('raw_root', 'var') || isempty(raw_root)
    raw_root = getenv('LUNGCT_4DCT_ROOT');
    if isempty(raw_root)
        error('Set raw_root or LUNGCT_4DCT_ROOT before running.');
    end
end
if ~exist('reference_root', 'var') || isempty(reference_root)
    reference_root = getenv('INRORSPLINE_PTVREG_REFERENCE_ROOT');
end
if ~exist('protocol_name', 'var') || isempty(protocol_name)
    protocol_name = 'published_1mm_cp0_fastlcc_cpu';
end
if ~exist('metric_name', 'var') || isempty(metric_name)
    if contains(protocol_name, 'gpu')
        metric_name = 'loc_cc_fftn_gpu';
    else
        metric_name = 'loc_cc_fftn';
    end
end

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
ptv_root = fullfile(repo_root, 'external', 'pTVreg');

addpath(genpath(fullfile(ptv_root, 'mutils', 'My')));
addpath(genpath(fullfile(ptv_root, 'ptv')));

case_label = sprintf('case%02d', case_id);
run_stamp = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmss'));
case_out_root = fullfile(output_root, protocol_name, case_label);
out_dir = fullfile(case_out_root, run_stamp);
if exist(out_dir, 'dir')
    error('Output directory already exists: %s', out_dir);
end
mkdir(out_dir);

datetime_start = datetime('now');
matlab_version = version;

fprintf('pTVreg DIR-Lab 4DCT reproduction\n');
fprintf('case_id: %d\n', case_id);
fprintf('protocol_name: %s\n', protocol_name);
fprintf('output: %s\n', out_dir);

spec = local_case_spec(case_id, raw_root);

volmov = double(local_read_dir_img(spec.mov_img_path, spec.size));
volfix = double(local_read_dir_img(spec.fix_img_path, spec.size));
pts_mov = local_read_points(spec.mov_pts_path);
pts_fix = local_read_points(spec.fix_pts_path);
spc_orig = spec.spacing;

fprintf('raw moving image: %s\n', spec.mov_img_path);
fprintf('raw fixed image: %s\n', spec.fix_img_path);
fprintf('raw size: %s\n', mat2str(spec.size));
fprintf('native spacing: %s\n', mat2str(spc_orig));

volmov = img_thr(volmov, 80, 900, 1);
volfix = img_thr(volfix, 80, 900, 1);

init_size = size(volmov);
min_max1 = [min(pts_mov, [], 1)', max(pts_mov, [], 1)'];
min_max2 = [min(pts_fix, [], 1)', max(pts_fix, [], 1)'];
min_max = [min(min_max1(:, 1), min_max2(:, 1)), ...
           max(min_max1(:, 2), min_max2(:, 2))];
margin_xyz = [10, 10, 5];
crop_v = [max(1, min_max(1, 1) - margin_xyz(1)), min(size(volmov, 1), min_max(1, 2) + margin_xyz(1)); ...
          max(1, min_max(2, 1) - margin_xyz(2)), min(size(volmov, 2), min_max(2, 2) + margin_xyz(2)); ...
          max(1, min_max(3, 1) - margin_xyz(3)), min(size(volmov, 3), min_max(3, 2) + margin_xyz(3))];

fprintf('crop_v: %s\n', mat2str(crop_v));

volmov = crop_data(volmov, crop_v);
volfix = crop_data(volfix, crop_v);

native_crop_size = size(volmov);
target_spacing = [1, 1, 1];
target_size = round(native_crop_size .* spc_orig .* target_spacing);

fprintf('native crop size: %s\n', mat2str(native_crop_size));
fprintf('1mm registration size: %s\n', mat2str(target_size));

volfix = volresize(volfix, target_size, 1);
volmov = volresize(volmov, target_size, 1);

opts = [];
opts.loc_cc_approximate = true;
opts.grid_spacing = [4, 4, 4];
opts.cp_refinements = 0;
opts.display = 'iter';
opts.k_down = 0.7;
opts.interp_type = 0;
opts.metric = metric_name;
opts.metric_param = [2.1, 2.1, 2.1];
opts.scale_metric_param = true;
opts.isoTV = 0.11;
opts.csqrt = 5e-3;
opts.spline_order = 1;
opts.border_mask = 5;
opts.max_iters = 80;
opts.check_gradients = 0;
opts.pix_resolution = [1, 1, 1];

gpu_info = struct();
gpu_info.requested = contains(opts.metric, 'gpu');
gpu_info.available = false;
gpu_info.count = 0;
gpu_info.name = '';
gpu_info.error = '';
if gpu_info.requested
    try
        gpu_info.count = gpuDeviceCount;
        if gpu_info.count > 0
            gpu_dev = gpuDevice;
            gpu_info.available = true;
            gpu_info.name = gpu_dev.Name;
            gpu_info.index = gpu_dev.Index;
            gpu_info.compute_capability = gpu_dev.ComputeCapability;
            gpu_info.total_memory = gpu_dev.TotalMemory;
            gpu_info.available_memory = gpu_dev.AvailableMemory;
            fprintf('GPU device: %s\n', gpu_info.name);
            fprintf('GPU total memory bytes: %.0f\n', gpu_info.total_memory);
            fprintf('GPU available memory bytes: %.0f\n', gpu_info.available_memory);
        else
            error('No available MATLAB GPU device found.');
        end
    catch ME
        gpu_info.error = ME.message;
        error('GPU protocol requested but GPU setup failed: %s', ME.message);
    end
end

fprintf('registration opts:\n');
disp(opts);

timer = tic;
[voldef, Tptv, Kptv, itinfos] = ptv_register(volmov, volfix, opts);
time_used_sec = toc(timer);
datetime_end = datetime('now');

fprintf('registration time_used_sec: %.6f\n', time_used_sec);

Tptv_rsz = cat(4, ...
    volresize(Tptv(:, :, :, 1), native_crop_size), ...
    volresize(Tptv(:, :, :, 2), native_crop_size), ...
    volresize(Tptv(:, :, :, 3), native_crop_size));
voldef_native_crop = volresize(voldef, native_crop_size);

[~, Tptv_rsz] = uncrop_data(voldef_native_crop, Tptv_rsz, crop_v, init_size);
Tptv_rsz = single(Tptv_rsz);

[pt_errs_phys, pts_moved_pix, TRE_phys, TREstd_case, pts_moved_subvoxel] = DIR_movepoints( ...
    pts_mov, pts_fix, Tptv_rsz, spc_orig, []);

TRE_all_phys = pt_errs_phys;
crop_v_saved = crop_v;
spc_orig_saved = spc_orig;

reference = struct();
reference_available = false;
if ~isempty(reference_root)
    reference_mat = fullfile(reference_root, sprintf('%s_T50_to_T00_deformation.mat', case_label));
    reference_available = exist(reference_mat, 'file') == 2;
else
    reference_mat = '';
end
if reference_available
    reference_loaded = load(reference_mat, 'TRE_phys', 'TREstd_case', ...
        'crop_v_saved', 'Tptv_rsz');
    reference.path = reference_mat;
    reference.TRE_phys = reference_loaded.TRE_phys;
    reference.TREstd_case = reference_loaded.TREstd_case;
    reference.crop_v_saved = reference_loaded.crop_v_saved;
    reference.Tptv_rsz_size = size(reference_loaded.Tptv_rsz);
    reference.delta_TRE_phys = TRE_phys - reference.TRE_phys;
    reference.delta_TREstd_case = TREstd_case - reference.TREstd_case;
    reference.crop_matches = isequal(crop_v_saved, reference.crop_v_saved);
    reference.field_size_matches = isequal(size(Tptv_rsz), reference.Tptv_rsz_size);
else
    reference.path = reference_mat;
end

fprintf('TRE_phys_mean_mm: %.10f\n', TRE_phys);
fprintf('TREstd_case_mm: %.10f\n', TREstd_case);
fprintf('Tptv_rsz size: %s\n', mat2str(size(Tptv_rsz)));
if reference_available
    fprintf('reference_TRE_phys_mean_mm: %.10f\n', reference.TRE_phys);
    fprintf('delta_TRE_phys_mean_mm: %.10f\n', reference.delta_TRE_phys);
    fprintf('reference_crop_matches: %d\n', reference.crop_matches);
    fprintf('reference_field_size_matches: %d\n', reference.field_size_matches);
end

result_mat = fullfile(out_dir, sprintf('%s_T50_to_T00_deformation.mat', case_label));
summary_txt = fullfile(out_dir, 'summary.txt');

save(result_mat, ...
    'Tptv_rsz', 'TRE_phys', 'TREstd_case', 'TRE_all_phys', ...
    'pt_errs_phys', 'pts_moved_pix', 'pts_moved_subvoxel', ...
    'pts_mov', 'pts_fix', ...
    'crop_v_saved', 'spc_orig_saved', 'time_used_sec', ...
    'datetime_start', 'datetime_end', 'opts', ...
    'protocol_name', 'case_id', 'case_label', 'matlab_version', ...
    'target_size', 'native_crop_size', 'margin_xyz', 'Kptv', 'itinfos', ...
    'reference', 'reference_available', 'gpu_info', 'metric_name', ...
    '-v7.3');

fid = fopen(summary_txt, 'w');
if fid < 0
    error('Could not open summary file for writing: %s', summary_txt);
end
fprintf(fid, 'case_id: %d\n', case_id);
fprintf(fid, 'case_label: %s\n', case_label);
fprintf(fid, 'protocol_name: %s\n', protocol_name);
fprintf(fid, 'matlab_version: %s\n', matlab_version);
fprintf(fid, 'metric_name: %s\n', metric_name);
fprintf(fid, 'gpu_requested: %d\n', gpu_info.requested);
fprintf(fid, 'gpu_available: %d\n', gpu_info.available);
fprintf(fid, 'gpu_count: %d\n', gpu_info.count);
fprintf(fid, 'gpu_name: %s\n', gpu_info.name);
if isfield(gpu_info, 'total_memory')
    fprintf(fid, 'gpu_total_memory_bytes: %.0f\n', gpu_info.total_memory);
    fprintf(fid, 'gpu_available_memory_bytes_start: %.0f\n', gpu_info.available_memory);
end
if ~isempty(gpu_info.error)
    fprintf(fid, 'gpu_error: %s\n', gpu_info.error);
end
fprintf(fid, 'datetime_start: %s\n', char(datetime_start));
fprintf(fid, 'datetime_end: %s\n', char(datetime_end));
fprintf(fid, 'time_used_sec: %.6f\n', time_used_sec);
fprintf(fid, 'TRE_phys_mean_mm: %.10f\n', TRE_phys);
fprintf(fid, 'TREstd_case_mm: %.10f\n', TREstd_case);
fprintf(fid, 'Tptv_rsz_size: %s\n', mat2str(size(Tptv_rsz)));
fprintf(fid, 'crop_v_saved: %s\n', mat2str(crop_v_saved));
fprintf(fid, 'spc_orig_saved: %s\n', mat2str(spc_orig_saved));
fprintf(fid, 'target_size: %s\n', mat2str(target_size));
fprintf(fid, 'native_crop_size: %s\n', mat2str(native_crop_size));
fprintf(fid, 'reference_mat: %s\n', reference.path);
fprintf(fid, 'reference_available: %d\n', reference_available);
if reference_available
    fprintf(fid, 'reference_TRE_phys_mean_mm: %.10f\n', reference.TRE_phys);
    fprintf(fid, 'reference_TREstd_case_mm: %.10f\n', reference.TREstd_case);
    fprintf(fid, 'delta_TRE_phys_mean_mm: %.10f\n', reference.delta_TRE_phys);
    fprintf(fid, 'delta_TREstd_case_mm: %.10f\n', reference.delta_TREstd_case);
    fprintf(fid, 'reference_crop_v_saved: %s\n', mat2str(reference.crop_v_saved));
    fprintf(fid, 'reference_Tptv_rsz_size: %s\n', mat2str(reference.Tptv_rsz_size));
    fprintf(fid, 'reference_crop_matches: %d\n', reference.crop_matches);
    fprintf(fid, 'reference_field_size_matches: %d\n', reference.field_size_matches);
end
fprintf(fid, 'result_mat: %s\n', result_mat);
fclose(fid);

fprintf('saved result: %s\n', result_mat);
fprintf('saved summary: %s\n', summary_txt);

function spec = local_case_spec(case_id, raw_root)
    sizes = {[256, 256, 94], ...
             [256, 256, 112], ...
             [256, 256, 104], ...
             [256, 256, 99], ...
             [256, 256, 106], ...
             [512, 512, 128], ...
             [512, 512, 136], ...
             [512, 512, 128], ...
             [512, 512, 128], ...
             [512, 512, 120]};
    spacings = {[0.97, 0.97, 2.5], ...
                [1.16, 1.16, 2.5], ...
                [1.15, 1.15, 2.5], ...
                [1.13, 1.13, 2.5], ...
                [1.10, 1.10, 2.5], ...
                [0.97, 0.97, 2.5], ...
                [0.97, 0.97, 2.5], ...
                [0.97, 0.97, 2.5], ...
                [0.97, 0.97, 2.5], ...
                [0.97, 0.97, 2.5]};
    postfixes = {'_s.img', ...
                 '-ssm.img', ...
                 '-ssm.img', ...
                 '-ssm.img', ...
                 '-ssm.img', ...
                 '.img', ...
                 '.img', ...
                 '.img', ...
                 '.img', ...
                 '.img'};
    if case_id < 1 || case_id > 10
        error('case_id must be in 1..10, got %d', case_id);
    end

    case_pack = fullfile(raw_root, sprintf('Case%dPack', case_id));
    if case_id == 8
        case_pack = fullfile(case_pack, 'Case8Deploy');
    end
    image_dir = fullfile(case_pack, 'Images');
    extreme_dir = fullfile(case_pack, 'ExtremePhases');
    if ~exist(extreme_dir, 'dir')
        extreme_dir = fullfile(case_pack, 'extremePhases');
    end

    spec = struct();
    spec.size = sizes{case_id};
    spec.spacing = spacings{case_id};
    spec.mov_img_path = fullfile(image_dir, sprintf('case%d_T50%s', case_id, postfixes{case_id}));
    spec.fix_img_path = fullfile(image_dir, sprintf('case%d_T00%s', case_id, postfixes{case_id}));
    spec.mov_pts_path = fullfile(extreme_dir, sprintf('Case%d_300_T50_xyz.txt', case_id));
    spec.fix_pts_path = fullfile(extreme_dir, sprintf('Case%d_300_T00_xyz.txt', case_id));

    if ~exist(spec.mov_pts_path, 'file')
        spec.mov_pts_path = fullfile(extreme_dir, sprintf('case%d_dirLab300_T50_xyz.txt', case_id));
    end
    if ~exist(spec.fix_pts_path, 'file')
        spec.fix_pts_path = fullfile(extreme_dir, sprintf('case%d_dirLab300_T00_xyz.txt', case_id));
    end

    required = {spec.mov_img_path, spec.fix_img_path, spec.mov_pts_path, spec.fix_pts_path};
    for i = 1:numel(required)
        if ~exist(required{i}, 'file')
            error('Required input is missing: %s', required{i});
        end
    end
end

function vol = local_read_dir_img(fname, sz)
    fid = fopen(fname, 'rb');
    if fid < 0
        error('Could not open image: %s', fname);
    end
    cleaner = onCleanup(@() fclose(fid));
    vol = fread(fid, sz(1) * sz(2) * sz(3), '*int16');
    if numel(vol) ~= prod(sz)
        error('Unexpected voxel count for %s: got %d expected %d', fname, numel(vol), prod(sz));
    end
    vol = reshape(vol, sz);
end

function pts = local_read_points(fname)
    fid = fopen(fname, 'r');
    if fid < 0
        error('Could not open points file: %s', fname);
    end
    cleaner = onCleanup(@() fclose(fid));
    data = textscan(fid, '%f %f %f', 'headerlines', 0);
    pts = [data{1}, data{2}, data{3}];
    if size(pts, 2) ~= 3 || isempty(pts)
        error('Could not parse points from: %s', fname);
    end
end
