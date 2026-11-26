import { useEffect, useRef, useState } from 'react';
import { DEFAULT_PARALLEL_TASKS, MAX_PARALLEL_TASKS } from '../lib/parallelTasks';

// internal design notes (E1): the room's base
// number of tasks that run side by side. Beyond it the Hub starts more only
// while the room's memory has room to spare, and never more than four.
export function ParallelTasksDialog({ room, onSave, onClose }) {
  const initial = room.parallel_tasks_base ?? DEFAULT_PARALLEL_TASKS;
  const [value, setValue] = useState(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const dialog = useRef(null);

  useEffect(() => {
    const element = dialog.current;
    element.showModal();
    return () => element.close();
  }, []);

  const submit = async event => {
    event.preventDefault();
    if (saving) return;
    const n = Number(value);
    const base = Number.isInteger(n) && n >= 1 && n <= MAX_PARALLEL_TASKS ? n : initial;
    setError('');
    setSaving(true);
    try {
      await onSave({ parallel_tasks_base: base });
      onClose();
    } catch (err) {
      setError(err.message);
      setSaving(false);
    }
  };

  return (
    <dialog
      ref={dialog}
      aria-label="Parallel tasks"
      className="w-[calc(100%-2rem)] max-w-sm rounded-2xl p-5 backdrop:bg-black/60"
      style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
      onCancel={event => {
        event.preventDefault();
        if (!saving) onClose();
      }}
    >
      <form onSubmit={submit} className="space-y-4">
        <h3 className="text-sm font-semibold">Parallel tasks</h3>
        <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
          Always run up to
          <input
            type="number"
            min={1}
            max={MAX_PARALLEL_TASKS}
            step={1}
            required
            disabled={saving}
            value={value}
            onChange={event => setValue(event.target.value)}
            className="input-field w-16 rounded-lg px-2 py-1"
          />
          tasks at once
        </label>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          All tasks of a room share its memory. More tasks than this start only while the
          room has memory to spare, and never more than {MAX_PARALLEL_TASKS}. The rest wait
          and start by themselves when a running task finishes.
        </p>
        {error && <p role="alert" className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-ghost text-xs" onClick={onClose} disabled={saving}>Cancel</button>
          <button type="submit" className="btn-primary text-xs" disabled={saving}>{saving ? 'Saving…' : 'Save'}</button>
        </div>
      </form>
    </dialog>
  );
}
