# Technical log

Known limitations are recorded before launch.

- This campaign compares optimization paths, not independent registration
  representations: all six variants use DirectCP cubic B-spline controls.
- 4DCT accelerated schedules are a documented four-stage projection of the
  five-stage L2B schedules because the established 4DCT anchor has four
  control-grid stages.
- Cold/warm timing is process-temperature timing. Warm cases do not reuse
  controls, optimizer moments, or registration state.
- Timing ends at saved dense-DVF output and is not a measurement of downstream
  TRE/regularity evaluation or launcher queue time.
- TRE reporting must preserve native-crop snap-to-voxel as the headline metric
  and retain reg-1mm snap/subvoxel values as diagnostics.

## Launch record

The full campaign completed on GPUs 0-3 with 12 persistent dataset/variant
workers and 120 completed registrations. Automatic aggregate validation
reported zero errors. Review tables and the report are generated in this
directory; large run outputs remain in the external run root.

The saved-DVF timing boundary is recorded immediately after both dense 1-mm
DVF arrays are written and before TRE/regularity evaluation. Thus
`loaded_images_to_dvf_file_sec` excludes downstream evaluation while including
input loading, optimization, final materialization, and DVF writing.
