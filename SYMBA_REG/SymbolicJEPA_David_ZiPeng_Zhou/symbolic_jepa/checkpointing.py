"""Local-first, atomic checkpointing with best-effort Google Drive backup.

Colab's mounted Drive is a FUSE filesystem: writes can fail, stall, or land
truncated, and a failure there must never destroy a run.  The invariant this
module enforces is::

    training
       -> local Colab filesystem (/content/...)
       -> atomic save (temp file -> fsync -> load back -> validate)
       -> authoritative local checkpoint
       -> best-effort backup to Google Drive

A Drive error after a successful local save is a warning, not an exception.

Scope: checkpoint / manifest / backup plumbing only.  Nothing here knows about
models, datasets, or the training loop.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path

import torch

__all__ = [
    'CheckpointError',
    'REQUIRED_FIELDS',
    'is_temp_name',
    'validate_checkpoint',
    'save_checkpoint_atomic',
    'save_json_atomic',
    'load_checkpoint',
    'load_local_or_restore',
    'backup_file_to_drive',
    'backup_checkpoint_to_drive',
    'check_best_latest_consistency',
    'write_manifest',
    'sync_local_runs_to_drive',
]


class CheckpointError(RuntimeError):
    """A checkpoint is missing, malformed, or belongs to a different run."""


# Fields every checkpoint of a given kind must carry.  Callers add run-specific
# ones through `extra_required` (e.g. optimizer/scheduler/history for resume).
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    'latest': ('model', 'epoch', 'seed', 'run_config'),
    'best': ('model', 'epoch', 'val', 'val_acc', 'seed', 'run_config'),
    'pretrain': ('model', 'epoch', 'seed', 'run_config'),
}

# Substrings that mark a file as in-flight or quarantined.  Nothing carrying
# one of these is ever treated as a resumable checkpoint or synced to Drive.
TEMP_MARKERS = ('.tmp-', '.uploading-', '.corrupt-', '.restore-')

# What `sync_local_runs_to_drive` is willing to copy.
DEFAULT_SYNC_NAMES = (
    'latest.pt', 'best.pt', 'metrics.json', 'manifest.json', 'history.json',
)


def is_temp_name(name: str) -> bool:
    """True if *name* is an in-flight or quarantined file, not a checkpoint."""
    return any(marker in name for marker in TEMP_MARKERS)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


def _tmp_path(final_path: Path, marker: str) -> Path:
    """A sibling temp path on the same filesystem as *final_path*."""
    return final_path.with_name(
        f'{final_path.name}{marker}{os.getpid()}-{uuid.uuid4().hex[:8]}'
    )


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync so a rename survives a hard crash."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _run_config_diff(saved: Mapping, expected: Mapping) -> list[str]:
    lines = [
        f'  {k}: stored={saved.get(k, "<missing>")!r}  expected={v!r}'
        for k, v in sorted(expected.items())
        if saved.get(k, '<missing>') != v
    ]
    lines += [
        f'  {k}: stored={saved[k]!r}  expected=<absent>'
        for k in sorted(saved) if k not in expected
    ]
    return lines


def validate_checkpoint(
    checkpoint,
    expected_seed: int,
    expected_run_config: Mapping,
    kind: str = 'latest',
    extra_required: tuple[str, ...] = (),
    max_epoch: int | None = None,
    path=None,
) -> None:
    """Raise CheckpointError unless *checkpoint* belongs to the expected run.

    Checks structure (required fields, non-empty model state, sensible epoch)
    and provenance (seed, experiment version, exact run configuration).  A
    checkpoint that passes has been produced by this experiment version, this
    seed, and this configuration — nothing else can validate.

    Args:
        checkpoint: The loaded checkpoint mapping.
        expected_seed: Seed the current run is training under.
        expected_run_config: `make_run_config(...)` for the current run.
            Must contain an ``experiment_version`` key.
        kind: ``'latest'``, ``'best'``, or ``'pretrain'`` — selects the
            required-field set.
        extra_required: Additional fields the caller's resume logic needs.
        max_epoch: Upper bound for a sane epoch number, if known.
        path: File the checkpoint came from, for error messages.
    """
    where = f'\n  {path}' if path is not None else ''

    if kind not in REQUIRED_FIELDS:
        raise ValueError(f'unknown checkpoint kind {kind!r}')
    if not isinstance(checkpoint, Mapping):
        raise CheckpointError(
            f'{kind} checkpoint is a {type(checkpoint).__name__}, '
            f'not a dict.{where}')

    missing = [f for f in REQUIRED_FIELDS[kind] + tuple(extra_required)
               if f not in checkpoint]
    if missing:
        raise CheckpointError(
            f'{kind} checkpoint is missing required field(s): '
            f'{", ".join(missing)}.{where}')

    model = checkpoint['model']
    if not isinstance(model, Mapping) or len(model) == 0:
        raise CheckpointError(
            f'{kind} checkpoint has an empty or non-dict model state.{where}')

    epoch = checkpoint['epoch']
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise CheckpointError(
            f'{kind} checkpoint has a nonsensical epoch {epoch!r}.{where}')
    if max_epoch is not None and epoch > max_epoch:
        raise CheckpointError(
            f'{kind} checkpoint epoch {epoch} exceeds the configured maximum '
            f'{max_epoch}; it was not produced by this configuration.{where}')

    if checkpoint['seed'] != expected_seed:
        raise CheckpointError(
            f'{kind} checkpoint seed {checkpoint["seed"]!r} != run seed '
            f'{expected_seed!r}.{where}')

    saved_cfg = checkpoint['run_config']
    if not isinstance(saved_cfg, Mapping) or not saved_cfg:
        raise CheckpointError(
            f'{kind} checkpoint carries no run configuration; its provenance '
            f'cannot be established.{where}')

    # Checked on its own so a v1 (or any foreign-version) file gets a message
    # that names the actual problem instead of a 20-line config diff.
    saved_ver = saved_cfg.get('experiment_version', '<missing>')
    want_ver = expected_run_config['experiment_version']
    if saved_ver != want_ver:
        raise CheckpointError(
            f'{kind} checkpoint belongs to experiment version {saved_ver!r}, '
            f'not {want_ver!r}. Checkpoints are never migrated across '
            f'versions.{where}')

    diff = _run_config_diff(saved_cfg, expected_run_config)
    if diff:
        raise CheckpointError(
            f'{kind} checkpoint run configuration does not match this '
            f'run:{where}\n' + '\n'.join(diff))

    if kind == 'best':
        val, val_acc = checkpoint['val'], checkpoint['val_acc']
        if not isinstance(val, (int, float)) or val != val:  # NaN
            raise CheckpointError(
                f'best checkpoint has a non-numeric val {val!r}.{where}')
        if not isinstance(val_acc, (int, float)) or not 0.0 <= val_acc <= 1.0:
            raise CheckpointError(
                f'best checkpoint val_acc {val_acc!r} is not a fraction in '
                f'[0, 1].{where}')


def check_best_latest_consistency(
    best: Mapping, latest: Mapping, rtol: float = 1e-6, atol: float = 1e-8,
    path=None,
) -> None:
    """Raise CheckpointError if best.pt and latest.pt describe different runs.

    ``latest['best_val']`` is written in the same epoch that produced
    ``best['val']``, so the two are the *same float* — they are not
    independently recomputed.  The tolerance therefore only absorbs
    float32/float64 round-tripping through torch.save, not genuine numerical
    drift; anything larger means the two files come from different sessions.
    """
    where = f'\n  {path}' if path is not None else ''

    best_epoch, latest_epoch = best.get('epoch'), latest.get('epoch')
    if best_epoch is not None and latest_epoch is not None:
        if best_epoch > latest_epoch:
            raise CheckpointError(
                f'CHECKPOINT INCONSISTENCY: best.pt epoch {best_epoch} is '
                f'newer than latest.pt epoch {latest_epoch}. best.pt is '
                f'written before latest.pt within an epoch, so a single run '
                f'always has latest >= best; these came from different '
                f'sessions.{where}')

    bv, lbv = best.get('val'), latest.get('best_val')
    if bv is None or lbv is None:
        return
    if abs(bv - lbv) > atol + rtol * abs(lbv):
        raise CheckpointError(
            f'CHECKPOINT INCONSISTENCY\n'
            f'  best.pt   val      = {bv!r}\n'
            f'  latest.pt best_val = {lbv!r}\n'
            f'best.pt does not belong to this run.{where}')


# ---------------------------------------------------------------------------
# Atomic local writes
# ---------------------------------------------------------------------------

def save_checkpoint_atomic(
    checkpoint: Mapping,
    final_path,
    validate=None,
    keep_prev: bool = True,
    label: str | None = None,
    verbose: bool = True,
) -> Path:
    """Write *checkpoint* to *final_path* atomically, verifying it first.

    Sequence: create the directory, write a temp file on the same filesystem,
    flush + fsync, load the temp file back with ``map_location='cpu'``,
    run *validate* on what was actually read from disk, and only then
    ``os.replace()`` it into place.  ``final_path`` is never written directly,
    so a crash or a bad serialization can never damage the previous good file.

    Args:
        checkpoint: Mapping to save.
        final_path: Authoritative destination (local filesystem).
        validate: Optional callable applied to the reloaded checkpoint; it
            should raise on any problem.
        keep_prev: Hard-link the outgoing file to ``<stem>.prev<suffix>``
            before installing the new one, keeping one previous generation.
        label: Name used in log lines; defaults to the file name.
        verbose: Print save/verify lines.

    Returns:
        The final path.

    Raises:
        CheckpointError: If the reloaded checkpoint fails *validate*.
    """
    final_path = Path(final_path)
    label = label or final_path.name
    final_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = _tmp_path(final_path, '.tmp-')
    try:
        with open(tmp, 'wb') as f:
            torch.save(dict(checkpoint), f)
            f.flush()
            os.fsync(f.fileno())

        reloaded = torch.load(tmp, map_location='cpu', weights_only=False)
        if validate is not None:
            validate(reloaded)
        del reloaded
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if keep_prev and final_path.exists():
        # Hard-link rather than copy: instant, and the final file stays present
        # the whole time.  Not available on every filesystem, hence best-effort.
        prev = final_path.with_name(f'{final_path.stem}.prev{final_path.suffix}')
        link_tmp = _tmp_path(prev, '.tmp-')
        try:
            os.link(final_path, link_tmp)
            os.replace(link_tmp, prev)
        except OSError:
            Path(link_tmp).unlink(missing_ok=True)

    os.replace(tmp, final_path)
    _fsync_dir(final_path.parent)

    if verbose:
        epoch = checkpoint.get('epoch')
        suffix = f' (epoch {epoch})' if epoch is not None else ''
        print(f'[checkpoint] saved local {label}{suffix}')
        print(f'[checkpoint] verified local {label}')
    return final_path


def save_json_atomic(obj, final_path, verbose: bool = False) -> Path:
    """Write *obj* as JSON through the same temp -> verify -> replace path."""
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = _tmp_path(final_path, '.tmp-')
    try:
        with open(tmp, 'w') as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp) as f:
            json.load(f)          # parses back, or we never install it
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    os.replace(tmp, final_path)
    _fsync_dir(final_path.parent)
    if verbose:
        print(f'[checkpoint] saved local {final_path.name}')
    return final_path


def write_manifest(manifest_path, updates: Mapping, verbose: bool = False) -> dict:
    """Merge *updates* into the run manifest and rewrite it atomically.

    Call only after a local checkpoint save has succeeded — the manifest
    describes what is on the local disk, so a Drive failure must never
    invalidate it.
    """
    manifest_path = Path(manifest_path)
    manifest: dict = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception as e:
            print(f'[checkpoint WARNING] manifest unreadable ({type(e).__name__}); '
                  f'rewriting it: {manifest_path}')
            manifest = {}

    manifest.update(updates)
    manifest['last_local_save_time'] = _now()
    save_json_atomic(manifest, manifest_path, verbose=verbose)
    return manifest


# ---------------------------------------------------------------------------
# Loading / resume
# ---------------------------------------------------------------------------

def load_checkpoint(path, validate=None, map_location='cpu'):
    """Load and optionally validate a checkpoint. Raises on either failure."""
    path = Path(path)
    if is_temp_name(path.name):
        raise CheckpointError(
            f'{path.name} is an in-flight or quarantined file and is never '
            f'resumable.\n  {path}')
    if not path.exists():
        raise CheckpointError(f'no such checkpoint\n  {path}')
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if validate is not None:
        validate(ck)
    return ck


def _quarantine(path: Path) -> Path:
    """Rename a bad checkpoint aside. Non-destructive: nothing is deleted."""
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    dest = path.with_name(f'{path.name}.corrupt-{stamp}')
    try:
        os.replace(path, dest)
        print(f'[checkpoint] quarantined -> {dest.name} (not deleted)')
    except OSError as e:
        print(f'[checkpoint WARNING] could not quarantine {path.name}: {e}')
    return dest


def load_local_or_restore(
    local_path, drive_path, validate=None, verbose: bool = True,
) -> tuple[object | None, str]:
    """Resolve a resumable checkpoint, preferring the local filesystem.

    ``local -> (restore drive to local) -> fresh``.  Training always runs from
    the local copy; a Drive backup is only ever used by copying it down,
    validating it, and installing it locally first.

    Returns:
        ``(checkpoint, source)`` where source is ``'local'``, ``'drive'``, or
        ``'none'``.  A corrupt or foreign local file is reported loudly and
        quarantined (never deleted, never silently trusted) before Drive is
        tried.
    """
    local_path = Path(local_path)

    if local_path.exists():
        try:
            ck = load_checkpoint(local_path, validate=validate)
            if verbose:
                print(f'[checkpoint] resuming from local {local_path.name} '
                      f'(epoch {ck.get("epoch")})')
            return ck, 'local'
        except Exception as e:
            print(f'[checkpoint ERROR] local {local_path.name} is unusable: '
                  f'{type(e).__name__}: {e}')
            _quarantine(local_path)

    if drive_path is not None:
        drive_path = Path(drive_path)
        try:
            available = drive_path.exists()
        except OSError as e:
            print(f'[backup WARNING] cannot reach Drive ({type(e).__name__}: {e}); '
                  f'starting fresh unless a local checkpoint exists.')
            available = False

        if available:
            tmp = _tmp_path(local_path, '.restore-')
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(drive_path, tmp)
                ck = torch.load(tmp, map_location='cpu', weights_only=False)
                if validate is not None:
                    validate(ck)
            except Exception as e:
                Path(tmp).unlink(missing_ok=True)
                print(f'[checkpoint ERROR] Drive backup {drive_path.name} could '
                      f'not be restored: {type(e).__name__}: {e}')
            else:
                os.replace(tmp, local_path)
                _fsync_dir(local_path.parent)
                if verbose:
                    print(f'[checkpoint] restored Drive -> local '
                          f'{local_path.name} (epoch {ck.get("epoch")}); '
                          f'training continues from the local copy')
                return ck, 'drive'

    if verbose:
        print(f'[checkpoint] no valid checkpoint for {local_path.parent.name}'
              f'/{local_path.name}; starting fresh')
    return None, 'none'


# ---------------------------------------------------------------------------
# Best-effort Drive backup
# ---------------------------------------------------------------------------

def backup_file_to_drive(
    local_path, drive_path, verify_size: bool = True,
    verify_load: bool = False, verbose: bool = True,
) -> bool:
    """Copy an already-validated local file to Drive. Never raises.

    Copies to a temporary Drive name first and only replaces the previous
    backup once the copy is complete, so an interrupted upload cannot leave a
    truncated file where a good backup used to be.

    Returns:
        True on success.  On any failure a ``[backup WARNING]`` is printed —
        visually distinct from the ``[checkpoint ERROR]`` used for local
        problems — and False is returned so the caller keeps training.
    """
    local_path, tmp = Path(local_path), None
    if drive_path is None:
        return False
    drive_path = Path(drive_path)

    try:
        if not local_path.exists():
            raise FileNotFoundError(f'nothing to back up at {local_path}')

        drive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_path(drive_path, '.uploading-')
        shutil.copy2(local_path, tmp)

        if verify_size:
            src, dst = local_path.stat().st_size, Path(tmp).stat().st_size
            if src != dst:
                raise OSError(f'size mismatch after copy: {src} -> {dst} bytes')
        if verify_load and drive_path.suffix == '.pt':
            torch.load(tmp, map_location='cpu', weights_only=False)

        os.replace(tmp, drive_path)
        if verbose:
            print(f'[backup] {local_path.name} -> Drive ok')
        return True
    except Exception as e:
        print(f'[backup WARNING] Google Drive copy failed: '
              f'{type(e).__name__}: {e}')
        print('Local checkpoint remains valid; continuing training.')
        if tmp is not None:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass
        return False


# Same behaviour, named for the checkpoint call sites.
backup_checkpoint_to_drive = backup_file_to_drive


def sync_local_runs_to_drive(
    local_root, drive_root, names: tuple[str, ...] = DEFAULT_SYNC_NAMES,
    verbose: bool = True,
) -> dict:
    """Copy completed local run artifacts to Drive. Never deletes anything.

    Intended to be run by hand before deliberately shutting a Colab runtime
    down.  Only files named in *names* are considered, and in-flight or
    quarantined files are skipped outright.
    """
    local_root = Path(local_root)
    if drive_root is None:
        print('[backup] no Drive root configured; nothing to sync.')
        return {'ok': 0, 'failed': 0, 'skipped': 0}
    drive_root = Path(drive_root)

    if not local_root.exists():
        print(f'[backup] no local run directory yet: {local_root}')
        return {'ok': 0, 'failed': 0, 'skipped': 0}

    ok = failed = skipped = 0
    for path in sorted(local_root.rglob('*')):
        if not path.is_file():
            continue
        if is_temp_name(path.name) or path.name not in names:
            skipped += 1
            continue
        dest = drive_root / path.relative_to(local_root)
        if backup_file_to_drive(path, dest, verbose=False):
            ok += 1
            if verbose:
                print(f'  ok    {path.relative_to(local_root)}')
        else:
            failed += 1
            if verbose:
                print(f'  FAIL  {path.relative_to(local_root)}')

    print(f'\n[backup] sync complete: {ok} copied, {failed} failed, '
          f'{skipped} skipped (temp/unlisted).')
    print(f'  local: {local_root}\n  drive: {drive_root}')
    print('  Nothing was deleted locally or on Drive.')
    return {'ok': ok, 'failed': failed, 'skipped': skipped}
