import { useEffect, useRef, useState } from 'react';
import { autoRetrySettings, DEFAULT_AUTO_RETRY_MAX } from '../lib/autoRetry';

export function AutoRetryFields({ enabled, attempts, onEnabledChange, onAttemptsChange, disabled }) {
  return (
    <fieldset className="space-y-2" disabled={disabled}>
      <label className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-primary)' }}>
        <input
          type="checkbox"
          role="switch"
          className="accent-[var(--accent)]"
          checked={enabled}
          onChange={event => onEnabledChange(event.target.checked)}
        />
        Auto-retry on failure
      </label>
      <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
        Attempts
        <input
          type="number"
          min={1}
          max={10}
          step={1}
          required
          disabled={!enabled}
          value={attempts}
          onChange={event => onAttemptsChange(event.target.value)}
          className="input-field w-20 rounded-lg px-2 py-1 disabled:opacity-50"
        />
      </label>
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
        Automatic retries are billed like regular work.
      </p>
    </fieldset>
  );
}

export function AutoRetrySettingsDialog({ room, onSave, onClose }) {
  const initialMax = room.auto_retry_max ?? DEFAULT_AUTO_RETRY_MAX;
  const [enabled, setEnabled] = useState(room.auto_retry_enabled ?? true);
  const [attempts, setAttempts] = useState(initialMax);
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
    setError('');
    setSaving(true);
    try {
      await onSave(autoRetrySettings(enabled, attempts, initialMax));
      onClose();
    } catch (err) {
      setError(err.message);
      setSaving(false);
    }
  };

  return (
    <dialog
      ref={dialog}
      aria-label="Room auto-retry settings"
      className="w-[calc(100%-2rem)] max-w-sm rounded-2xl p-5 backdrop:bg-black/60"
      style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
      onCancel={event => {
        event.preventDefault();
        if (!saving) onClose();
      }}
    >
      <form onSubmit={submit} className="space-y-4">
        <h3 className="text-sm font-semibold">Room auto-retry settings</h3>
        <AutoRetryFields
          enabled={enabled}
          attempts={attempts}
          onEnabledChange={setEnabled}
          onAttemptsChange={setAttempts}
          disabled={saving}
        />
        {error && <p role="alert" className="text-xs" style={{ color: '#f87171' }}>{error}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-ghost text-xs" onClick={onClose} disabled={saving}>Cancel</button>
          <button type="submit" className="btn-primary text-xs" disabled={saving}>{saving ? 'Saving…' : 'Save'}</button>
        </div>
      </form>
    </dialog>
  );
}
