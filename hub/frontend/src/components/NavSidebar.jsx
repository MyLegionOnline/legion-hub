
import { MessageSquare, Bot, Sparkles, Settings, LogOut, Sun, Moon, Store, Link2, UserCircle } from 'lucide-react';
import { useOnboarding } from '../lib/onboardingContext';
import { Spot } from './OnboardingHint';

const NAV_ITEMS = [
  { id: 'chats',    icon: MessageSquare, label: 'Rooms'    },
  { id: 'agents',   icon: Bot,           label: 'Agents'   },
  { id: 'skills',   icon: Sparkles,      label: 'Skills'   },
  { id: 'connections', icon: Link2,      label: 'Connections' },
  { id: 'settings', icon: Settings,      label: 'Settings' },
];

// The first-run guide points at one nav item per step, until the user is there.
const GUIDE_HINTS = {
  secret:   { tab: 'settings', text: 'Add your first API key for your AI provider: Settings → Secrets.' },
  provider: { tab: 'settings', text: 'Set up your first provider: Settings → Providers.' },
  model:    { tab: 'settings', text: 'Choose the default model for your agents: Settings → Providers.' },
  agents:   { tab: 'agents',   text: 'Your agents now use the default model. Open one and press Configure to change it.' },
};

export function NavSidebar({ tab, onTab, onMarketplace, accountUrl, legalUrls, user, onLogout, theme, onThemeToggle }) {
  const isDark = theme === 'dark';
  const { step } = useOnboarding();
  const guide = step ? GUIDE_HINTS[step] : null;
  const guided = (id) => !!guide && guide.tab === id && tab !== id;

  return (
    <>
      {/* Desktop: vertical left sidebar */}
      <aside
        className="hidden md:flex flex-col py-4 gap-1 flex-shrink-0 border-r"
        style={{ width: 200, background: 'var(--bg-panel)', borderColor: 'var(--border)' }}
      >
        {/* Logo block */}
        <div className="px-4 mb-5 flex flex-col items-center select-none">
          <img
            src="/logo_small.svg"
            alt="Legion"
            style={{ width: '100%', height: 'auto', display: 'block' }}
          />
          <p
            className="text-[10px] font-semibold uppercase mt-2 whitespace-nowrap"
            style={{ color: 'var(--text-muted)', letterSpacing: '0.12em' }}
          >
            My Legion online
          </p>
        </div>

        {/* Nav items */}
        <div className="flex flex-col gap-0.5 flex-1 px-2">
          {onMarketplace && (
            <button className="nav-item" onClick={onMarketplace} title="Open Legion Market">
              <Store size={17} /><span>Marketplace</span>
            </button>
          )}
          {NAV_ITEMS.map(({ id, icon: Icon, label }) => (
            <Spot key={id} active={guided(id)} hint={guide?.text} placement="right">
              <button
                className={`nav-item ${tab === id ? 'active' : ''}`}
                onClick={() => onTab(id)}
              >
                <Icon size={17} />
                <span>{label}</span>
              </button>
            </Spot>
          ))}
        </div>

        {/* Bottom controls */}
        <div className="px-2 flex flex-col gap-0.5">
          {/* Theme toggle */}
          <button
            className="nav-item"
            title={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
            onClick={onThemeToggle}
          >
            {isDark ? <Sun size={17} /> : <Moon size={17} />}
            <span>{isDark ? 'Light mode' : 'Dark mode'}</span>
          </button>

          {/* User */}
          <button
            className="nav-item"
            title={user ? `${user.full_name} (${user.email})` : 'Profile'}
            onClick={() => onTab('settings')}
          >
            <div
              className="avatar w-6 h-6 text-[10px] flex-shrink-0"
              style={{ background: 'var(--accent-dim)', color: 'var(--accent-light)', border: '1.5px solid var(--accent)' }}
            >
              {(user?.full_name || 'U').slice(0, 2).toUpperCase()}
            </div>
            <span className="truncate text-xs">{user?.full_name || 'Profile'}</span>
          </button>

          {/* The Legion account profile lives in Market (internal design notes A6) */}
          {accountUrl && (
            <a className="nav-item" href={accountUrl} target="_blank" rel="noopener noreferrer" title="Your Legion account">
              <UserCircle size={17} />
              <span>Legion account</span>
            </a>
          )}

          {/* Logout */}
          <button className="nav-item" title="Sign out" onClick={onLogout}>
            <LogOut size={17} />
            <span>Sign out</span>
          </button>

          {/* Legion's documents, while an account of Legion is attached (legal plan L4) */}
          {legalUrls && (
            <p className="px-3 pt-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
              <a href={legalUrls.terms} target="_blank" rel="noopener noreferrer" className="hover:underline">Terms</a>
              {' · '}
              <a href={legalUrls.privacy} target="_blank" rel="noopener noreferrer" className="hover:underline">Privacy</a>
            </p>
          )}
        </div>
      </aside>

      {/* Mobile: bottom nav bar */}
      <nav
        className="md:hidden fixed bottom-0 left-0 right-0 z-50 flex items-center justify-around px-4 py-2 border-t"
        style={{ background: 'var(--bg-panel)', borderColor: 'var(--border)', height: 60 }}
      >
        {NAV_ITEMS.map(({ id, icon: Icon, label }) => (
          <Spot key={id} active={guided(id)} hint={guide?.text} placement="top-end">
            <button
              onClick={() => onTab(id)}
              className="flex flex-col items-center gap-0.5 px-4 py-1 rounded-xl transition-all"
              style={{ color: tab === id ? 'var(--accent-light)' : 'var(--text-secondary)' }}
            >
              <Icon size={22} />
              <span className="text-[10px] font-medium">{label}</span>
            </button>
          </Spot>
        ))}
        {onMarketplace && (
          <button onClick={onMarketplace} className="flex flex-col items-center gap-0.5 px-4 py-1 rounded-xl transition-all" style={{ color: 'var(--text-secondary)' }}><Store size={22} /><span className="text-[10px] font-medium">Market</span></button>
        )}
      </nav>
    </>
  );
}
