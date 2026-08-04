% Run the full pTVreg LungCT CPU/GPU sweep sequentially.
%
% Optional variables before run():
%   sweep_root       existing or new output root
%   force            rerun successful cases, default false
%   display_mode     pTVreg display mode, default 'off'
%   smoke_only       run only COPD case01 GPU smoke tests, default false
%   data_root        DIR-Lab raw root containing 4DCT/ and COPD/
%   reference_root   optional root containing reference deformation files

if ~exist('force', 'var') || isempty(force)
    force = false;
end
if ~exist('display_mode', 'var') || isempty(display_mode)
    display_mode = 'off';
end
if ~exist('smoke_only', 'var') || isempty(smoke_only)
    smoke_only = false;
end
if ~exist('data_root', 'var') || isempty(data_root)
    data_root = getenv('LUNGCT_DIRLAB_ROOT');
    if isempty(data_root)
        error('Set data_root or LUNGCT_DIRLAB_ROOT before running.');
    end
end
if ~exist('reference_root', 'var') || isempty(reference_root)
    reference_root = getenv('INRORSPLINE_PTVREG_REFERENCE_ROOT');
end
if ~exist('sweep_root', 'var') || isempty(sweep_root)
    run_stamp = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmss'));
    sweep_parent = getenv('INRORSPLINE_PTVREG_SWEEP_ROOT');
    if isempty(sweep_parent)
        error('Set sweep_root or INRORSPLINE_PTVREG_SWEEP_ROOT before running.');
    end
    sweep_root = fullfile(sweep_parent, run_stamp);
end
if ~exist(sweep_root, 'dir')
    mkdir(sweep_root);
end

fprintf('pTVreg full LungCT sweep root: %s\n', sweep_root);
local_preflight(sweep_root, data_root, reference_root);

if smoke_only
    tasks = local_smoke_tasks();
else
    tasks = local_full_tasks();
end

failures = {};
for i = 1:numel(tasks)
    task = tasks(i);
    fprintf('\n=== Task %d/%d: %s case%02d %s %s ===\n', ...
        i, numel(tasks), task.dataset, task.case_id, task.protocol, task.backend);
    try
        run_lungct_pairwise_case(task.dataset, task.case_id, task.protocol, task.backend, sweep_root, ...
            'force', force, 'display', display_mode, 'data_root', data_root, ...
            'reference_root', reference_root);
    catch ME
        failures{end + 1, 1} = sprintf('%s case%02d %s %s: %s', ...
            task.dataset, task.case_id, task.protocol, task.backend, ME.message); %#ok<SAGROW>
        local_write_error(sweep_root, task, ME);
        fprintf(2, 'FAILED: %s\n', failures{end});
    end
    local_generate_reports(sweep_root, false);
end

local_generate_reports(sweep_root, true);

if ~isempty(failures)
    fprintf(2, '\nSweep completed with %d failed task(s):\n', numel(failures));
    for i = 1:numel(failures)
        fprintf(2, '  %s\n', failures{i});
    end
    error('pTVreg full sweep had failures. See error.txt files under %s', sweep_root);
end

fprintf('\nSweep completed successfully.\n');
fprintf('Results: %s\n', sweep_root);

function tasks = local_full_tasks()
    blocks = {
        '4dct', '4dct_published_1mm_cp0_fastlcc_linear', 'gpu';
        'copd', 'copd_published_1mm_cp1_exactlcc_cubic', 'gpu';
        'copd', 'copd_published_1mm_cp0_fastlcc_linear', 'gpu';
        '4dct', '4dct_published_1mm_cp0_fastlcc_linear', 'cpu';
        'copd', 'copd_published_1mm_cp1_exactlcc_cubic', 'cpu';
        'copd', 'copd_published_1mm_cp0_fastlcc_linear', 'cpu';
        };
    tasks = struct('dataset', {}, 'protocol', {}, 'backend', {}, 'case_id', {});
    for b = 1:size(blocks, 1)
        for case_id = 1:10
            tasks(end + 1) = struct('dataset', blocks{b, 1}, 'protocol', blocks{b, 2}, ...
                'backend', blocks{b, 3}, 'case_id', case_id); %#ok<SAGROW>
        end
    end
end

function tasks = local_smoke_tasks()
    tasks = [
        struct('dataset', 'copd', 'protocol', 'copd_published_1mm_cp1_exactlcc_cubic', 'backend', 'gpu', 'case_id', 1), ...
        struct('dataset', 'copd', 'protocol', 'copd_published_1mm_cp0_fastlcc_linear', 'backend', 'gpu', 'case_id', 1)];
end

function local_preflight(sweep_root, data_root, reference_root)
    fprintf('Preflight: MATLAB %s\n', version);
    gcount = gpuDeviceCount;
    fprintf('Preflight GPU count: %d\n', gcount);
    if gcount < 1
        error('No MATLAB GPU is visible.');
    end
    g = gpuDevice;
    fprintf('Preflight default GPU: %s, available_memory=%.0f\n', g.Name, g.AvailableMemory);
    local_validate_all_inputs(sweep_root, data_root, reference_root);
end

function local_validate_all_inputs(sweep_root, data_root, reference_root)
    probe_root = fullfile(sweep_root, '_preflight_path_probe');
    probes = {
        '4dct', '4dct_published_1mm_cp0_fastlcc_linear';
        'copd', 'copd_published_1mm_cp1_exactlcc_cubic';
        'copd', 'copd_published_1mm_cp0_fastlcc_linear';
        };
    for p = 1:size(probes, 1)
        for case_id = 1:10
            run_lungct_pairwise_case(probes{p, 1}, case_id, probes{p, 2}, 'cpu', probe_root, ...
                'dry_run', true, 'display', 'off', 'force', true, ...
                'data_root', data_root, 'reference_root', reference_root);
        end
    end
    fprintf('Preflight input resolution passed for all 4DCT/COPD cases.\n');
end

function local_write_error(sweep_root, task, ME)
    out_dir = fullfile(sweep_root, task.dataset, task.protocol, task.backend, sprintf('case%02d', task.case_id));
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    fid = fopen(fullfile(out_dir, 'error.txt'), 'w');
    if fid >= 0
        fprintf(fid, 'dataset: %s\n', task.dataset);
        fprintf(fid, 'case_id: %d\n', task.case_id);
        fprintf(fid, 'protocol_name: %s\n', task.protocol);
        fprintf(fid, 'backend_name: %s\n', task.backend);
        fprintf(fid, 'message: %s\n', ME.message);
        fprintf(fid, 'identifier: %s\n', ME.identifier);
        for k = 1:numel(ME.stack)
            fprintf(fid, 'stack_%d: %s:%d %s\n', k, ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
        end
        fclose(fid);
    end
end

function local_generate_reports(sweep_root, compute_field_diffs)
    if nargin < 2
        compute_field_diffs = true;
    end
    rows = local_collect_rows(sweep_root);
    if isempty(rows)
        return;
    end
    T = struct2table(rows);
    T = local_normalize_text_columns(T);
    T = sortrows(T, {'dataset', 'protocol_name', 'backend_name', 'case_id'});
    writetable(T, fullfile(sweep_root, 'all_results.csv'));
    C = local_backend_comparison(T, compute_field_diffs);
    if ~isempty(C)
        writetable(C, fullfile(sweep_root, 'backend_comparison.csv'));
    end
    local_write_summary_md(sweep_root, T, C);
end

function rows = local_collect_rows(sweep_root)
    listing = dir(fullfile(sweep_root, '**', 'summary.txt'));
    rows = struct('dataset', {}, 'case_id', {}, 'case_label', {}, 'protocol_name', {}, ...
        'backend_name', {}, 'metric_name', {}, 'status', {}, 'TRE_phys_mean_mm', {}, ...
        'TREstd_case_mm', {}, 'time_used_sec', {}, 'crop_v_saved', {}, 'Tptv_rsz_size', {}, ...
        'published_TRE_phys_mm', {}, 'delta_published_TRE_phys_mm', {}, ...
        'reference_available', {}, 'delta_reference_TRE_phys_mean_mm', {}, 'result_mat', {});
    for i = 1:numel(listing)
        path = fullfile(listing(i).folder, listing(i).name);
        if contains(path, [filesep, '_preflight_path_probe', filesep])
            continue;
        end
        kv = local_read_kv(path);
        if ~isfield(kv, 'status') || ~strcmp(kv.status, 'success')
            continue;
        end
        row = struct();
        row.dataset = local_get(kv, 'dataset', '');
        row.case_id = str2double(local_get(kv, 'case_id', 'NaN'));
        row.case_label = local_get(kv, 'case_label', '');
        row.protocol_name = local_get(kv, 'protocol_name', '');
        row.backend_name = local_get(kv, 'backend_name', '');
        row.metric_name = local_get(kv, 'metric_name', '');
        row.status = local_get(kv, 'status', '');
        row.TRE_phys_mean_mm = str2double(local_get(kv, 'TRE_phys_mean_mm', 'NaN'));
        row.TREstd_case_mm = str2double(local_get(kv, 'TREstd_case_mm', 'NaN'));
        row.time_used_sec = str2double(local_get(kv, 'time_used_sec', 'NaN'));
        row.crop_v_saved = local_get(kv, 'crop_v_saved', '');
        row.Tptv_rsz_size = local_get(kv, 'Tptv_rsz_size', '');
        row.published_TRE_phys_mm = str2double(local_get(kv, 'published_TRE_phys_mm', 'NaN'));
        row.delta_published_TRE_phys_mm = str2double(local_get(kv, 'delta_published_TRE_phys_mm', 'NaN'));
        row.reference_available = str2double(local_get(kv, 'reference_available', '0'));
        row.delta_reference_TRE_phys_mean_mm = str2double(local_get(kv, 'delta_reference_TRE_phys_mean_mm', 'NaN'));
        row.result_mat = local_get(kv, 'result_mat', '');
        rows(end + 1) = row; %#ok<AGROW>
    end
end

function T = local_normalize_text_columns(T)
    text_cols = {'dataset', 'case_label', 'protocol_name', 'backend_name', ...
        'metric_name', 'status', 'crop_v_saved', 'Tptv_rsz_size', 'result_mat'};
    for i = 1:numel(text_cols)
        col = text_cols{i};
        if ~ismember(col, T.Properties.VariableNames)
            continue;
        end
        if ischar(T.(col))
            T.(col) = cellstr(T.(col));
        elseif isstring(T.(col))
            T.(col) = cellstr(T.(col));
        end
    end
end

function C = local_backend_comparison(T, compute_field_diffs)
    keys = unique(T(:, {'dataset', 'protocol_name', 'case_id'}), 'rows');
    rows = struct('dataset', {}, 'protocol_name', {}, 'case_id', {}, ...
        'gpu_time_sec', {}, 'cpu_time_sec', {}, 'speedup_cpu_over_gpu', {}, ...
        'gpu_TRE_phys_mean_mm', {}, 'cpu_TRE_phys_mean_mm', {}, 'delta_gpu_minus_cpu_TRE_mm', {}, ...
        'gpu_TREstd_case_mm', {}, 'cpu_TREstd_case_mm', {}, 'delta_gpu_minus_cpu_TREstd_mm', {}, ...
        'field_max_abs_diff', {}, 'field_mean_abs_diff', {}, 'field_rms_diff', {});
    for i = 1:height(keys)
        mask = strcmp(T.dataset, keys.dataset{i}) & strcmp(T.protocol_name, keys.protocol_name{i}) & T.case_id == keys.case_id(i);
        gpu = T(mask & strcmp(T.backend_name, 'gpu'), :);
        cpu = T(mask & strcmp(T.backend_name, 'cpu'), :);
        if isempty(gpu) || isempty(cpu)
            continue;
        end
        row = struct();
        row.dataset = keys.dataset{i};
        row.protocol_name = keys.protocol_name{i};
        row.case_id = keys.case_id(i);
        row.gpu_time_sec = gpu.time_used_sec(1);
        row.cpu_time_sec = cpu.time_used_sec(1);
        row.speedup_cpu_over_gpu = cpu.time_used_sec(1) / gpu.time_used_sec(1);
        row.gpu_TRE_phys_mean_mm = gpu.TRE_phys_mean_mm(1);
        row.cpu_TRE_phys_mean_mm = cpu.TRE_phys_mean_mm(1);
        row.delta_gpu_minus_cpu_TRE_mm = gpu.TRE_phys_mean_mm(1) - cpu.TRE_phys_mean_mm(1);
        row.gpu_TREstd_case_mm = gpu.TREstd_case_mm(1);
        row.cpu_TREstd_case_mm = cpu.TREstd_case_mm(1);
        row.delta_gpu_minus_cpu_TREstd_mm = gpu.TREstd_case_mm(1) - cpu.TREstd_case_mm(1);
        if compute_field_diffs
            [row.field_max_abs_diff, row.field_mean_abs_diff, row.field_rms_diff] = ...
                local_field_diff(gpu.result_mat{1}, cpu.result_mat{1});
        else
            row.field_max_abs_diff = NaN;
            row.field_mean_abs_diff = NaN;
            row.field_rms_diff = NaN;
        end
        rows(end + 1) = row; %#ok<AGROW>
    end
    if isempty(rows)
        C = table();
    else
        C = struct2table(rows);
        C = local_normalize_text_columns(C);
    end
end

function [max_abs_diff, mean_abs_diff, rms_diff] = local_field_diff(gpu_mat, cpu_mat)
    max_abs_diff = NaN;
    mean_abs_diff = NaN;
    rms_diff = NaN;
    if ~exist(gpu_mat, 'file') || ~exist(cpu_mat, 'file')
        return;
    end
    try
        g = load(gpu_mat, 'Tptv_rsz');
        c = load(cpu_mat, 'Tptv_rsz');
        d = double(g.Tptv_rsz) - double(c.Tptv_rsz);
        max_abs_diff = max(abs(d(:)));
        mean_abs_diff = mean(abs(d(:)));
        rms_diff = sqrt(mean(d(:) .^ 2));
    catch ME
        fprintf(2, 'Field comparison failed for %s vs %s: %s\n', gpu_mat, cpu_mat, ME.message);
    end
end

function local_write_summary_md(sweep_root, T, C)
    fid = fopen(fullfile(sweep_root, 'summary.md'), 'w');
    if fid < 0
        error('Could not write summary.md');
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '# pTVreg LungCT Host Reproduction Sweep\n\n');
    fprintf(fid, 'Sweep root: `%s`\n\n', sweep_root);
    fprintf(fid, 'Completed runs: %d\n\n', height(T));
    protocols = unique(T.protocol_name, 'stable');
    for p = 1:numel(protocols)
        sub = T(strcmp(T.protocol_name, protocols{p}), :);
        fprintf(fid, '## %s\n\n', protocols{p});
        fprintf(fid, '| Backend | Cases | Mean TRE (mm) | Mean TRE std (mm) | Mean runtime (sec) | Total runtime (hr) |\n');
        fprintf(fid, '|---|---:|---:|---:|---:|---:|\n');
        backends = unique(sub.backend_name, 'stable');
        for b = 1:numel(backends)
            sb = sub(strcmp(sub.backend_name, backends{b}), :);
            fprintf(fid, '| %s | %d | %.6f | %.6f | %.3f | %.3f |\n', ...
                backends{b}, height(sb), mean(sb.TRE_phys_mean_mm, 'omitnan'), ...
                mean(sb.TREstd_case_mm, 'omitnan'), mean(sb.time_used_sec, 'omitnan'), ...
                sum(sb.time_used_sec, 'omitnan') / 3600);
        end
        fprintf(fid, '\n');
        fprintf(fid, '| Case | Backend | TRE (mm) | TRE std (mm) | Runtime (sec) | Published TRE | Delta published |\n');
        fprintf(fid, '|---:|---|---:|---:|---:|---:|---:|\n');
        sub = sortrows(sub, {'case_id', 'backend_name'});
        for i = 1:height(sub)
            fprintf(fid, '| %02d | %s | %.10f | %.10f | %.3f | %.3f | %.6f |\n', ...
                sub.case_id(i), sub.backend_name{i}, sub.TRE_phys_mean_mm(i), ...
                sub.TREstd_case_mm(i), sub.time_used_sec(i), ...
                sub.published_TRE_phys_mm(i), sub.delta_published_TRE_phys_mm(i));
        end
        fprintf(fid, '\n');
    end
    if ~isempty(C)
        fprintf(fid, '## CPU vs GPU\n\n');
        fprintf(fid, '| Dataset | Protocol | Case | CPU/GPU speedup | Delta TRE GPU-CPU | Field mean abs diff | Field RMS diff |\n');
        fprintf(fid, '|---|---|---:|---:|---:|---:|---:|\n');
        for i = 1:height(C)
            fprintf(fid, '| %s | %s | %02d | %.3f | %.10f | %.10g | %.10g |\n', ...
                C.dataset{i}, C.protocol_name{i}, C.case_id(i), C.speedup_cpu_over_gpu(i), ...
                C.delta_gpu_minus_cpu_TRE_mm(i), C.field_mean_abs_diff(i), C.field_rms_diff(i));
        end
        fprintf(fid, '\n');
    end
end

function kv = local_read_kv(path)
    kv = struct();
    text = fileread(path);
    lines = regexp(text, '\r?\n', 'split');
    for i = 1:numel(lines)
        line = lines{i};
        idx = strfind(line, ':');
        if isempty(idx)
            continue;
        end
        key = strtrim(line(1:idx(1) - 1));
        val = strtrim(line(idx(1) + 1:end));
        key = regexprep(key, '[^A-Za-z0-9_]', '_');
        if isempty(key)
            continue;
        end
        kv.(key) = val;
    end
end

function val = local_get(kv, key, default_val)
    if isfield(kv, key)
        val = kv.(key);
    else
        val = default_val;
    end
end
