import { useEffect, useState } from 'react';

// Matches Tailwind's `md` breakpoint: below it the Hub uses the phone layout
// (bottom nav bar, one pane at a time, the room panel behind the ⋮ menu).
const QUERY = '(max-width: 767px)';

export function useIsMobile() {
  const [isMobile, setIsMobile] = useState(() => window.matchMedia(QUERY).matches);
  useEffect(() => {
    const mq = window.matchMedia(QUERY);
    const onChange = (e) => setIsMobile(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return isMobile;
}
