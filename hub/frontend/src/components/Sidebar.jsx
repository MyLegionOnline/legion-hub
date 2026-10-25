import { Hash, MessageSquare, Bot, Settings, LogOut } from 'lucide-react';

export function Sidebar({ onRoomSelect, selectedRoomId, onSettingsClick, activeView }) {
  const rooms = [
    { id: '1', name: 'general', type: 'group' },
    { id: '2', name: 'engineering', type: 'group' },
  ];

  const agents = [
    { id: 'a1', nickname: 'engineer', agent_uid: 'agent-123', status: 'online' },
    { id: 'a2', nickname: 'researcher', agent_uid: 'agent-456', status: 'offline' },
  ];

  return (
    <div className="w-64 bg-gray-800 flex flex-col border-r border-gray-700">
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <MessageSquare className="text-blue-500" />
          Legion
        </h1>
      </div>
      
      <div className="flex-1 overflow-y-auto p-2 space-y-4">
        <div>
          <h2 className="px-2 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
            Rooms
          </h2>
          <div className="space-y-1">
            {rooms.map((room) => (
              <button
                key={room.id}
                onClick={() => onRoomSelect(room)}
                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-sm transition-colors ${
                  activeView === 'chat' && selectedRoomId === room.id ? 'bg-blue-600 text-white' : 'text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                }`}
              >
                <Hash size={16} />
                {room.name}
              </button>
            ))}
          </div>
        </div>

        <div>
          <h2 className="px-2 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
            Agents
          </h2>
          <div className="space-y-1">
            {agents.map((agent) => (
              <div
                key={agent.id}
                className="flex items-center justify-between px-2 py-1.5 rounded text-sm text-gray-400"
              >
                <div className="flex items-center gap-2">
                  <Bot size={16} className={agent.status === 'online' ? 'text-green-500' : 'text-gray-600'} />
                  <span>{agent.nickname}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="p-2 border-t border-gray-700 space-y-1">
        <button 
            onClick={onSettingsClick}
            className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-sm transition-colors ${
                activeView === 'settings' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-700'
            }`}
        >
          <Settings size={16} />
          Settings
        </button>
        <button className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-sm text-red-400 hover:bg-red-900/20 transition-colors">
          <LogOut size={16} />
          Logout
        </button>
      </div>
    </div>
  );
}
