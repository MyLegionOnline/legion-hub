import { Sparkles } from 'lucide-react';
import { useOnboarding } from '../lib/onboardingContext';

// The first-run guide's two pieces (lib/onboarding.jsx decides which step is
// current): `Spot` rings an element and may hang a bubble off it; the banner
// is an inline card for content areas, where a floating bubble would be
// clipped by the cards around it.

function StepLabel() {
  const { stepNumber, total } = useOnboarding();
  return (
    <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'var(--accent-light)' }}>
      Step {stepNumber} of {total}
    </span>
  );
}

function SkipLink() {
  const { skip } = useOnboarding();
  return (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); skip(); }}
      className="text-[10px] underline-offset-2 hover:underline"
      style={{ color: 'var(--text-muted)' }}
    >
      Skip tour
    </button>
  );
}

const PLACEMENT = {
  right: { left: '100%', top: '50%', transform: 'translateY(-50%)', marginLeft: 14 },
  bottom: { top: '100%', left: '50%', transform: 'translateX(-50%)', marginTop: 12 },
  top: { bottom: '100%', left: '50%', transform: 'translateX(-50%)', marginBottom: 12 },
  'top-end': { bottom: '100%', right: 0, marginBottom: 12 },
};

const ARROW = {
  right: { left: -5, top: '50%', marginTop: -5 },
  bottom: { top: -5, left: '50%', marginLeft: -5 },
  top: { bottom: -5, left: '50%', marginLeft: -5 },
  'top-end': { bottom: -5, right: 24 },
};

function Bubble({ placement, children }) {
  return (
    <div
      role="status"
      className="absolute z-[60] rounded-xl px-3 py-2.5 fade-in text-left"
      style={{
        ...PLACEMENT[placement],
        width: 'max-content',
        maxWidth: 240,
        background: 'var(--bg-card)',
        border: '1px solid var(--accent)',
        boxShadow: '0 8px 24px rgba(0,0,0,0.35)',
        cursor: 'default',
      }}
      onClick={(e) => e.stopPropagation()}
    >
      <span
        aria-hidden="true"
        className="absolute"
        style={{
          ...ARROW[placement],
          width: 10, height: 10, transform: 'rotate(45deg)',
          background: 'var(--bg-card)',
          borderLeft: '1px solid var(--accent)', borderBottom: '1px solid var(--accent)',
        }}
      />
      <div className="flex items-center justify-between gap-3 mb-1 relative">
        <StepLabel />
        <SkipLink />
      </div>
      <p className="text-xs leading-snug relative" style={{ color: 'var(--text-primary)', fontWeight: 400 }}>{children}</p>
    </div>
  );
}

/**
 * Wraps `children` in a ring that pulses while `active`; with `hint` it also
 * shows a bubble on the given side.  Inactive, it only adds a positioned div.
 */
export function Spot({ active, hint, placement = 'right', className = '', radius = 12, children }) {
  return (
    <div
      className={`relative ${active ? 'onb-ring' : ''} ${className}`}
      style={{ borderRadius: radius }}
    >
      {children}
      {active && hint && <Bubble placement={placement}>{hint}</Bubble>}
    </div>
  );
}

/** Inline card shown at the top of a page while the guide is on `step`. */
export function OnboardingBanner({ step, children, actions }) {
  const { step: current } = useOnboarding();
  if (current !== step) return null;
  return (
    <div
      className="rounded-2xl p-4 space-y-2 fade-in"
      style={{ background: 'var(--accent-dim)', border: '1px solid var(--accent)' }}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-1.5">
          <Sparkles size={13} style={{ color: 'var(--accent-light)' }} />
          <StepLabel />
        </div>
        <SkipLink />
      </div>
      <div className="text-sm leading-snug" style={{ color: 'var(--text-primary)' }}>{children}</div>
      {actions}
    </div>
  );
}
