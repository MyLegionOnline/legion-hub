// Room MCP servers (assignment and per-agent scope) can change from the
// sidebar, the "All MCP servers" window or another tab. Every view that caches
// GET /rooms/{id}/mcp-servers listens for this event and refetches: the hook
// fires it after its own change, ChatWindow fires it on the
// ROOM_MCP_SCOPE_CHANGED room event.
export const ROOM_MCP_CHANGED = 'legion:room-mcp-changed';

export function notifyRoomMcpChanged(roomId) {
  window.dispatchEvent(new CustomEvent(ROOM_MCP_CHANGED, { detail: { roomId: String(roomId) } }));
}

// Subscribe to MCP changes of one room; returns the unsubscribe fn.
export function onRoomMcpChanged(roomId, handler) {
  const listener = (e) => {
    if (!roomId || e.detail?.roomId === String(roomId)) handler();
  };
  window.addEventListener(ROOM_MCP_CHANGED, listener);
  return () => window.removeEventListener(ROOM_MCP_CHANGED, listener);
}
