import { useEffect, useState } from 'react';
import { HardDrive } from 'lucide-react';
import { LEVEL_COLORS, fetchUsage, formatMb } from '../lib/storageQuota';

// The "Storage" bar in the sidebar (internal design notes §5.2, T3): files
// of all the owner's rooms against room_storage_quota_mb. Without a quota (self-hosted Hub)
// it is not shown.
const REFRESH_MS = 5 * 60 * 1000;

export default function StorageBar({ token }) {
  const [usage, setUsage] = useState(null);

  useEffect(() => {
    if (!token) return undefined;
    let cancelled = false;
    const load = () => fetchUsage(token).then(u => { if (!cancelled) setUsage(u); }).catch(() => {});
    load();
    const timer = setInterval(load, REFRESH_MS);
    return () => { cancelled = true; clearInterval(timer); };
  }, [token]);

  const quota = Number(usage?.storage_quota_mb);
  if (!usage || !Number.isFinite(quota) || quota <= 0) return null;
  const percent = Math.min(100, Number(usage.storage_percent || 0));
  const color = LEVEL_COLORS[usage.storage_level] || LEVEL_COLORS.ok;
  return (
    <div className="px-4 py-3 flex-shrink-0 border-t" style={{ borderColor: 'var(--border)' }} title="Room files — Settings → Plan and limits">
      <div className="flex items-center justify-between gap-2 text-[11px]">
        <span className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <HardDrive size={12} /> Storage
        </span>
        <span className="font-mono" style={{ color: 'var(--text-primary)' }}>
          {formatMb(usage.storage_used_mb)} / {formatMb(quota)}
        </span>
      </div>
      <div className="h-1.5 rounded-full mt-1.5 overflow-hidden" style={{ background: 'var(--bg-hover)' }}>
        <div className="h-full rounded-full" style={{ width: `${percent}%`, background: color }} />
      </div>
    </div>
  );
}
