// Room agent membership can change after the room is created (Members panel).
// Every view that caches GET /rooms/{id}/members listens for this event and
// refetches: the sidebar fires it after its own add/remove, ChatWindow fires
// it on the ROOM_AGENTS_CHANGED room event (a change made in another tab).
export const ROOM_AGENTS_CHANGED = 'legion:room-agents-changed';

export function notifyRoomAgentsChanged(roomId) {
  window.dispatchEvent(new CustomEvent(ROOM_AGENTS_CHANGED, { detail: { roomId: String(roomId) } }));
}

// Subscribe to membership changes of one room; returns the unsubscribe fn.
export function onRoomAgentsChanged(roomId, handler) {
  const listener = (e) => {
    if (!roomId || e.detail?.roomId === String(roomId)) handler();
  };
  window.addEventListener(ROOM_AGENTS_CHANGED, listener);
  return () => window.removeEventListener(ROOM_AGENTS_CHANGED, listener);
}

export const NOT_IN_ROOM_LABEL = 'not in room';
