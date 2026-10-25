/*
 * Which rendered preview (if any) a workspace file gets. `content` is what
 * `/workspace/fs/file` returns — plain text, or a `data:{mime};base64,...`
 * URI for binaries.
 */

const MARKDOWN_EXT = new Set(['md', 'markdown']);
const MERMAID_EXT = new Set(['mmd', 'mermaid']);

function extOf(path) {
  return String(path || '').split('.').pop()?.toLowerCase() || '';
}

/** 'pdf' | 'markdown' | 'mermaid' | null — null means "show as code". */
export function previewKind(path, content) {
  const ext = extOf(path);
  const isDataUri = typeof content === 'string' && content.startsWith('data:');
  if (ext === 'pdf') return isDataUri ? 'pdf' : null;
  if (isDataUri) return null;
  if (MARKDOWN_EXT.has(ext)) return 'markdown';
  if (MERMAID_EXT.has(ext)) return 'mermaid';
  return null;
}
