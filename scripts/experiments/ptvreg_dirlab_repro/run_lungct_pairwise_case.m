function result = run_lungct_pairwise_case(dataset_name, case_id, protocol_name, backend_name, sweep_root, varargin)
% Run one pTVreg LungCT pairwise case and save a resumable result folder.
%
% Required inputs:
%   dataset_name   '4dct' or 'copd'
%   case_id        integer 1..10
%   protocol_name  supported protocol handle from local_protocol_spec
%   backend_name   'gpu' or 'cpu'
%   sweep_root     output root for this sweep
%
% Optional name/value inputs:
%   'force'        rerun even if success.txt exists, default false
%   'dry_run'      resolve paths/options only, default false
%   'display'      pTVreg optimizer display option, default 'off'

    args = local_parse_args(varargin{:});
    dataset_name = lower(char(dataset_name));
    protocol_name = char(protocol_name);
    backend_name = lower(char(backend_name));
    case_id = double(case_id);

    repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    ptv_root = fullfile(repo_root, 'external', 'pTVreg');
    addpath(genpath(fullfile(ptv_root, 'mutils', 'My')));
    addpath(genpath(fullfile(ptv_root, 'ptv')));

    spec = local_case_spec(dataset_name, case_id, args.data_root);
    protocol = local_protocol_spec(protocol_name);
    metric_name = local_metric_name(backend_name);
    case_label = sprintf('case%02d', case_id);
    out_dir = fullfile(sweep_root, dataset_name, protocol_name, backend_name, case_label);
    summary_txt = fullfile(out_dir, 'summary.txt');
    success_txt = fullfile(out_dir, 'success.txt');
    result_mat = fullfile(out_dir, sprintf('%s_%s_deformation.mat', case_label, spec.direction_tag));

    result = struct();
    result.dataset = dataset_name;
    result.case_id = case_id;
    result.case_label = case_label;
    result.protocol_name = protocol_name;
    result.backend_name = backend_name;
    result.metric_name = metric_name;
    result.out_dir = out_dir;
    result.summary_txt = summary_txt;
    result.result_mat = result_mat;
    result.status = 'initialized';
    result.skipped = false;

    if exist(success_txt, 'file') && ~args.force
        result.status = 'success';
        result.skipped = true;
        fprintf('SKIP existing success: %s %s %s %s\n', dataset_name, case_label, protocol_name, backend_name);
        return;
    end

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    if args.dry_run
        result.status = 'dry_run_ok';
        local_write_summary(summary_txt, result, spec, protocol, struct(), struct(), [], [], version);
        return;
    end

    datetime_start = datetime('now');
    matlab_version = version;
    gpu_info = local_gpu_info(backend_name);

    fprintf('RUN %s %s %s %s\n', dataset_name, case_label, protocol_name, backend_name);
    fprintf('moving image: %s\n', spec.mov_img_path);
    fprintf('fixed image: %s\n', spec.fix_img_path);
    fprintf('out_dir: %s\n', out_dir);

    volmov = double(local_read_dir_img(spec.mov_img_path, spec.size_xyz));
    volfix = double(local_read_dir_img(spec.fix_img_path, spec.size_xyz));
    pts_mov = local_read_points(spec.mov_pts_path);
    pts_fix = local_read_points(spec.fix_pts_path);
    spc_orig = spec.spacing_xyz;

    volmov = img_thr(volmov, 80, 900, 1);
    volfix = img_thr(volfix, 80, 900, 1);

    init_size = size(volmov);
    margin_xyz = [10, 10, 5];
    crop_v = local_landmark_crop(pts_mov, pts_fix, size(volmov), margin_xyz);
    volmov = crop_data(volmov, crop_v);
    volfix = crop_data(volfix, crop_v);

    native_crop_size = size(volmov);
    target_size = round(native_crop_size .* spc_orig .* [1, 1, 1]);
    fprintf('crop_v: %s\n', mat2str(crop_v));
    fprintf('native crop size: %s\n', mat2str(native_crop_size));
    fprintf('1mm registration size: %s\n', mat2str(target_size));

    volfix = volresize(volfix, target_size, 1);
    volmov = volresize(volmov, target_size, 1);

    opts = [];
    opts.loc_cc_approximate = protocol.loc_cc_approximate;
    opts.grid_spacing = protocol.grid_spacing;
    opts.cp_refinements = protocol.cp_refinements;
    opts.display = args.display;
    opts.k_down = 0.7;
    opts.interp_type = protocol.interp_type;
    opts.nlvl = [];
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

    timer = tic;
    [voldef, Tptv, Kptv, itinfos] = ptv_register(volmov, volfix, opts);
    time_used_sec = toc(timer);
    datetime_end = datetime('now');

    Tptv_rsz = cat(4, ...
        volresize(Tptv(:, :, :, 1), native_crop_size), ...
        volresize(Tptv(:, :, :, 2), native_crop_size), ...
        volresize(Tptv(:, :, :, 3), native_crop_size));
    voldef_native_crop = volresize(voldef, native_crop_size);
    [voldef_native_full, Tptv_rsz] = uncrop_data(voldef_native_crop, Tptv_rsz, crop_v, init_size);
    Tptv_rsz = single(Tptv_rsz);
    voldef_native_full = single(voldef_native_full);

    [pt_errs_phys, pts_moved_pix, TRE_phys, TREstd_case, pts_moved_subvoxel] = DIR_movepoints( ...
        pts_mov, pts_fix, Tptv_rsz, spc_orig, []);
    TRE_all_phys = pt_errs_phys;
    crop_v_saved = crop_v;
    spc_orig_saved = spc_orig;

    reference = local_reference(dataset_name, case_id, protocol_name, Tptv_rsz, TRE_phys, TREstd_case, crop_v_saved, args.reference_root);
    reference_available = reference.available;

    result.status = 'success';
    result.skipped = false;
    result.TRE_phys = TRE_phys;
    result.TREstd_case = TREstd_case;
    result.time_used_sec = time_used_sec;
    result.field_size = size(Tptv_rsz);
    result.crop_v_saved = crop_v_saved;

    save(result_mat, ...
        'dataset_name', 'case_id', 'case_label', 'protocol_name', 'backend_name', 'metric_name', ...
        'Tptv_rsz', 'voldef_native_full', 'voldef', 'Tptv', 'Kptv', 'itinfos', ...
        'TRE_phys', 'TREstd_case', 'TRE_all_phys', 'pt_errs_phys', ...
        'pts_moved_pix', 'pts_moved_subvoxel', 'pts_mov', 'pts_fix', ...
        'crop_v_saved', 'spc_orig_saved', 'time_used_sec', ...
        'datetime_start', 'datetime_end', 'matlab_version', ...
        'opts', 'gpu_info', 'protocol', 'spec', 'reference', 'reference_available', ...
        'target_size', 'native_crop_size', 'margin_xyz', ...
        '-v7.3');

    local_write_summary(summary_txt, result, spec, protocol, gpu_info, reference, datetime_start, datetime_end, matlab_version);
    local_touch_success(success_txt);

    fprintf('DONE %s %s %s %s TRE=%.10f std=%.10f time=%.3f sec\n', ...
        dataset_name, case_label, protocol_name, backend_name, TRE_phys, TREstd_case, time_used_sec);
end

function args = local_parse_args(varargin)
    args = struct('force', false, 'dry_run', false, 'display', 'off', ...
        'data_root', getenv('LUNGCT_DIRLAB_ROOT'), ...
        'reference_root', getenv('INRORSPLINE_PTVREG_REFERENCE_ROOT'));
    if mod(numel(varargin), 2) ~= 0
        error('Optional arguments must be name/value pairs.');
    end
    for i = 1:2:numel(varargin)
        key = lower(char(varargin{i}));
        args.(key) = varargin{i + 1};
    end
end

function metric_name = local_metric_name(backend_name)
    switch backend_name
        case 'gpu'
            metric_name = 'loc_cc_fftn_gpu';
        case 'cpu'
            metric_name = 'loc_cc_fftn';
        otherwise
            error('Unsupported backend_name: %s', backend_name);
    end
end

function protocol = local_protocol_spec(protocol_name)
    protocol = struct('name', protocol_name);
    switch protocol_name
        case '4dct_published_1mm_cp0_fastlcc_linear'
            protocol.dataset = '4dct';
            protocol.grid_spacing = [4, 4, 4];
            protocol.cp_refinements = 0;
            protocol.loc_cc_approximate = true;
            protocol.interp_type = 0;
            protocol.published_tre = [0.77, 0.75, 0.93, 1.26, 1.07, 0.83, 0.80, 1.01, 0.91, 0.84];
            protocol.published_mean_tre = 0.92;
            protocol.published_mean_time_sec = 130;
        case 'copd_published_1mm_cp1_exactlcc_cubic'
            protocol.dataset = 'copd';
            protocol.grid_spacing = [8, 8, 8];
            protocol.cp_refinements = 1;
            protocol.loc_cc_approximate = false;
            protocol.interp_type = 1;
            protocol.published_tre = [0.71, 1.91, 0.77, 0.67, 0.71, 0.66, 0.75, 0.78, 0.64, 0.85];
            protocol.published_mean_tre = 0.8461;
            protocol.published_mean_time_sec = 442;
        case 'copd_published_1mm_cp0_fastlcc_linear'
            protocol.dataset = 'copd';
            protocol.grid_spacing = [4, 4, 4];
            protocol.cp_refinements = 0;
            protocol.loc_cc_approximate = true;
            protocol.interp_type = 0;
            protocol.published_tre = [0.79, 3.81, 0.83, 0.78, 0.98, 0.90, 0.82, 1.02, 0.72, 1.07];
            protocol.published_mean_tre = 1.17;
            protocol.published_mean_time_sec = 186;
        otherwise
            error('Unsupported protocol_name: %s', protocol_name);
    end
end

function spec = local_case_spec(dataset_name, case_id, data_root)
    if nargin < 3 || isempty(data_root)
        error('Set data_root or LUNGCT_DIRLAB_ROOT before resolving inputs.');
    end
    switch dataset_name
        case '4dct'
            spec = local_4dct_case_spec(case_id, fullfile(data_root, '4DCT'));
        case 'copd'
            spec = local_copd_case_spec(case_id, fullfile(data_root, 'COPD'));
        otherwise
            error('Unsupported dataset_name: %s', dataset_name);
    end
end

function spec = local_4dct_case_spec(case_id, raw_root)
    sizes = {[256, 256, 94], [256, 256, 112], [256, 256, 104], [256, 256, 99], [256, 256, 106], ...
             [512, 512, 128], [512, 512, 136], [512, 512, 128], [512, 512, 128], [512, 512, 120]};
    spacings = {[0.97, 0.97, 2.5], [1.16, 1.16, 2.5], [1.15, 1.15, 2.5], [1.13, 1.13, 2.5], [1.10, 1.10, 2.5], ...
                [0.97, 0.97, 2.5], [0.97, 0.97, 2.5], [0.97, 0.97, 2.5], [0.97, 0.97, 2.5], [0.97, 0.97, 2.5]};
    postfixes = {'_s.img', '-ssm.img', '-ssm.img', '-ssm.img', '-ssm.img', '.img', '.img', '.img', '.img', '.img'};
    local_validate_case_id(case_id);

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
    spec.dataset = '4dct';
    spec.direction_tag = 'T50_to_T00';
    spec.moving_phase = 'T50';
    spec.fixed_phase = 'T00';
    spec.size_xyz = sizes{case_id};
    spec.spacing_xyz = spacings{case_id};
    spec.mov_img_path = fullfile(image_dir, sprintf('case%d_T50%s', case_id, postfixes{case_id}));
    spec.fix_img_path = fullfile(image_dir, sprintf('case%d_T00%s', case_id, postfixes{case_id}));
    spec.mov_pts_path = local_first_existing({ ...
        fullfile(extreme_dir, sprintf('Case%d_300_T50_xyz.txt', case_id)), ...
        fullfile(extreme_dir, sprintf('case%d_dirLab300_T50_xyz.txt', case_id))});
    spec.fix_pts_path = local_first_existing({ ...
        fullfile(extreme_dir, sprintf('Case%d_300_T00_xyz.txt', case_id)), ...
        fullfile(extreme_dir, sprintf('case%d_dirLab300_T00_xyz.txt', case_id))});
    local_validate_required(spec);
end

function spec = local_copd_case_spec(case_id, raw_root)
    sizes = {[512, 512, 121], [512, 512, 102], [512, 512, 126], [512, 512, 126], [512, 512, 131], ...
             [512, 512, 119], [512, 512, 112], [512, 512, 115], [512, 512, 116], [512, 512, 135]};
    spacings = {[0.625, 0.625, 2.5], [0.645, 0.645, 2.5], [0.652, 0.652, 2.5], [0.590, 0.590, 2.5], [0.647, 0.647, 2.5], ...
                [0.633, 0.633, 2.5], [0.625, 0.625, 2.5], [0.586, 0.586, 2.5], [0.664, 0.664, 2.5], [0.742, 0.742, 2.5]};
    local_validate_case_id(case_id);

    case_root = fullfile(raw_root, sprintf('copd%d', case_id), sprintf('copd%d', case_id));
    spec = struct();
    spec.dataset = 'copd';
    spec.direction_tag = 'eBH_to_iBH';
    spec.moving_phase = 'eBHCT';
    spec.fixed_phase = 'iBHCT';
    spec.size_xyz = sizes{case_id};
    spec.spacing_xyz = spacings{case_id};
    spec.mov_img_path = fullfile(case_root, sprintf('copd%d_eBHCT.img', case_id));
    spec.fix_img_path = fullfile(case_root, sprintf('copd%d_iBHCT.img', case_id));
    spec.mov_pts_path = fullfile(case_root, sprintf('copd%d_300_eBH_xyz_r1.txt', case_id));
    spec.fix_pts_path = fullfile(case_root, sprintf('copd%d_300_iBH_xyz_r1.txt', case_id));
    local_validate_required(spec);
end

function local_validate_case_id(case_id)
    if case_id < 1 || case_id > 10 || case_id ~= round(case_id)
        error('case_id must be an integer in 1..10, got %g', case_id);
    end
end

function path = local_first_existing(paths)
    path = paths{1};
    for i = 1:numel(paths)
        if exist(paths{i}, 'file')
            path = paths{i};
            return;
        end
    end
end

function local_validate_required(spec)
    required = {spec.mov_img_path, spec.fix_img_path, spec.mov_pts_path, spec.fix_pts_path};
    for i = 1:numel(required)
        if ~exist(required{i}, 'file')
            error('Required input is missing: %s', required{i});
        end
    end
end

function crop_v = local_landmark_crop(pts_mov, pts_fix, vol_size, margin_xyz)
    min_max1 = [min(pts_mov, [], 1)', max(pts_mov, [], 1)'];
    min_max2 = [min(pts_fix, [], 1)', max(pts_fix, [], 1)'];
    min_max = [min(min_max1(:, 1), min_max2(:, 1)), max(min_max1(:, 2), min_max2(:, 2))];
    crop_v = [max(1, min_max(1, 1) - margin_xyz(1)), min(vol_size(1), min_max(1, 2) + margin_xyz(1)); ...
              max(1, min_max(2, 1) - margin_xyz(2)), min(vol_size(2), min_max(2, 2) + margin_xyz(2)); ...
              max(1, min_max(3, 1) - margin_xyz(3)), min(vol_size(3), min_max(3, 2) + margin_xyz(3))];
end

function gpu_info = local_gpu_info(backend_name)
    gpu_info = struct('requested', strcmp(backend_name, 'gpu'), 'available', false, ...
        'count', 0, 'name', '', 'index', NaN, 'compute_capability', '', ...
        'total_memory', NaN, 'available_memory', NaN, 'error', '');
    if ~gpu_info.requested
        return;
    end
    try
        gpu_info.count = gpuDeviceCount;
        if gpu_info.count < 1
            error('No MATLAB GPU device is available.');
        end
        gpu_dev = gpuDevice;
        gpu_info.available = true;
        gpu_info.name = gpu_dev.Name;
        gpu_info.index = gpu_dev.Index;
        gpu_info.compute_capability = gpu_dev.ComputeCapability;
        gpu_info.total_memory = gpu_dev.TotalMemory;
        gpu_info.available_memory = gpu_dev.AvailableMemory;
    catch ME
        gpu_info.error = ME.message;
        error('GPU backend requested but GPU setup failed: %s', ME.message);
    end
end

function reference = local_reference(dataset_name, case_id, protocol_name, Tptv_rsz, TRE_phys, TREstd_case, crop_v_saved, reference_root)
    reference = struct('available', false, 'path', '', 'published_tre', NaN, ...
        'published_mean_tre', NaN, 'delta_published_tre', NaN);
    protocol = local_protocol_spec(protocol_name);
    if isfield(protocol, 'published_tre')
        reference.published_tre = protocol.published_tre(case_id);
        reference.published_mean_tre = protocol.published_mean_tre;
        reference.delta_published_tre = TRE_phys - reference.published_tre;
    end
    if strcmp(dataset_name, '4dct') && ~isempty(reference_root)
        reference.path = fullfile(reference_root, sprintf('case%02d_T50_to_T00_deformation.mat', case_id));
        if exist(reference.path, 'file')
            loaded = load(reference.path, 'TRE_phys', 'TREstd_case', 'crop_v_saved', 'Tptv_rsz');
            reference.available = true;
            reference.local_TRE_phys = loaded.TRE_phys;
            reference.local_TREstd_case = loaded.TREstd_case;
            reference.local_crop_v_saved = loaded.crop_v_saved;
            reference.local_Tptv_rsz_size = size(loaded.Tptv_rsz);
            reference.delta_local_TRE_phys = TRE_phys - loaded.TRE_phys;
            reference.delta_local_TREstd_case = TREstd_case - loaded.TREstd_case;
            reference.local_crop_matches = isequal(crop_v_saved, loaded.crop_v_saved);
            reference.local_field_size_matches = isequal(size(Tptv_rsz), reference.local_Tptv_rsz_size);
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

function local_write_summary(summary_txt, result, spec, protocol, gpu_info, reference, datetime_start, datetime_end, matlab_version)
    if nargin < 7
        datetime_start = NaT;
        datetime_end = NaT;
        matlab_version = version;
    end
    fid = fopen(summary_txt, 'w');
    if fid < 0
        error('Could not open summary file: %s', summary_txt);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, 'status: %s\n', result.status);
    fprintf(fid, 'dataset: %s\n', result.dataset);
    fprintf(fid, 'case_id: %d\n', result.case_id);
    fprintf(fid, 'case_label: %s\n', result.case_label);
    fprintf(fid, 'protocol_name: %s\n', result.protocol_name);
    fprintf(fid, 'backend_name: %s\n', result.backend_name);
    fprintf(fid, 'metric_name: %s\n', result.metric_name);
    fprintf(fid, 'matlab_version: %s\n', matlab_version);
    fprintf(fid, 'datetime_start: %s\n', char(datetime_start));
    fprintf(fid, 'datetime_end: %s\n', char(datetime_end));
    fprintf(fid, 'moving_phase: %s\n', spec.moving_phase);
    fprintf(fid, 'fixed_phase: %s\n', spec.fixed_phase);
    fprintf(fid, 'moving_image: %s\n', spec.mov_img_path);
    fprintf(fid, 'fixed_image: %s\n', spec.fix_img_path);
    fprintf(fid, 'moving_landmarks: %s\n', spec.mov_pts_path);
    fprintf(fid, 'fixed_landmarks: %s\n', spec.fix_pts_path);
    fprintf(fid, 'size_xyz: %s\n', mat2str(spec.size_xyz));
    fprintf(fid, 'spacing_xyz: %s\n', mat2str(spec.spacing_xyz));
    fprintf(fid, 'grid_spacing: %s\n', mat2str(protocol.grid_spacing));
    fprintf(fid, 'cp_refinements: %d\n', protocol.cp_refinements);
    fprintf(fid, 'loc_cc_approximate: %d\n', protocol.loc_cc_approximate);
    fprintf(fid, 'interp_type: %d\n', protocol.interp_type);
    if isfield(result, 'time_used_sec')
        fprintf(fid, 'time_used_sec: %.6f\n', result.time_used_sec);
        fprintf(fid, 'TRE_phys_mean_mm: %.10f\n', result.TRE_phys);
        fprintf(fid, 'TREstd_case_mm: %.10f\n', result.TREstd_case);
        fprintf(fid, 'Tptv_rsz_size: %s\n', mat2str(result.field_size));
        fprintf(fid, 'crop_v_saved: %s\n', mat2str(result.crop_v_saved));
        fprintf(fid, 'result_mat: %s\n', result.result_mat);
    end
    if ~isempty(fieldnames(gpu_info))
        fprintf(fid, 'gpu_requested: %d\n', gpu_info.requested);
        fprintf(fid, 'gpu_available: %d\n', gpu_info.available);
        fprintf(fid, 'gpu_count: %d\n', gpu_info.count);
        fprintf(fid, 'gpu_name: %s\n', gpu_info.name);
        fprintf(fid, 'gpu_total_memory_bytes: %.0f\n', gpu_info.total_memory);
        fprintf(fid, 'gpu_available_memory_bytes_start: %.0f\n', gpu_info.available_memory);
    end
    if ~isempty(fieldnames(reference))
        fprintf(fid, 'published_TRE_phys_mm: %.10f\n', reference.published_tre);
        fprintf(fid, 'delta_published_TRE_phys_mm: %.10f\n', reference.delta_published_tre);
        fprintf(fid, 'reference_mat: %s\n', reference.path);
        fprintf(fid, 'reference_available: %d\n', reference.available);
        if isfield(reference, 'local_TRE_phys')
            fprintf(fid, 'reference_TRE_phys_mean_mm: %.10f\n', reference.local_TRE_phys);
            fprintf(fid, 'delta_reference_TRE_phys_mean_mm: %.10f\n', reference.delta_local_TRE_phys);
            fprintf(fid, 'reference_crop_matches: %d\n', reference.local_crop_matches);
            fprintf(fid, 'reference_field_size_matches: %d\n', reference.local_field_size_matches);
        end
    end
end

function local_touch_success(success_txt)
    fid = fopen(success_txt, 'w');
    if fid < 0
        error('Could not write success marker: %s', success_txt);
    end
    fprintf(fid, 'success\n');
    fclose(fid);
end
