import { useEffect, useId, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { previewKind } from '../lib/filePreviewKind';

/*
 * Rendered preview for workspace files the browser can show without any
 * server-side tooling: PDF (the browser's own viewer), Markdown and Mermaid.
 */

/* ── PDF: data URI → Blob URL → browser's built-in viewer ── */
function PdfPreview({ content, path }) {
  const [url, setUrl] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let objectUrl = null;
    let cancelled = false;
    setUrl(null);
    setError(false);
    // The stored mime may be application/octet-stream; the viewer needs the
    // blob typed as PDF, so re-wrap the bytes.
    fetch(content)
      .then(r => r.arrayBuffer())
      .then(buf => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(new Blob([buf], { type: 'application/pdf' }));
        setUrl(objectUrl);
      })
      .catch(() => { if (!cancelled) setError(true); });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [content]);

  if (error) return <PreviewMessage text="Could not open the PDF" />;
  if (!url) return <PreviewMessage text="Loading…" />;
  return (
    <iframe
      src={url}
      title={path}
      className="flex-1 w-full border-0"
      style={{ minHeight: 300, background: '#525659' }}
    />
  );
}

/* ── Mermaid: lazy-loaded on first use, kept out of the main bundle ── */
let mermaidPromise = null;
function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then(m => m.default);
  }
  return mermaidPromise;
}

function currentMermaidTheme() {
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'default' : 'dark';
}

export function MermaidDiagram({ code }) {
  const ref = useRef(null);
  const reactId = useId();
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    const id = `mermaid-${reactId.replace(/[^a-zA-Z0-9_-]/g, '')}`;
    loadMermaid()
      .then(async (mermaid) => {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'strict',
          suppressErrorRendering: true,
          theme: currentMermaidTheme(),
        });
        const { svg } = await mermaid.render(id, code);
        if (!cancelled && ref.current) ref.current.innerHTML = svg;
      })
      .catch((err) => {
        if (!cancelled) setError(String(err?.message || err || 'Mermaid error'));
      });
    return () => { cancelled = true; };
  }, [code, reactId]);

  if (error) {
    return (
      <div className="my-2">
        <div className="text-[11px] mb-1" style={{ color: '#f87171' }}>Mermaid: {error}</div>
        <pre className="text-xs p-3 rounded-lg overflow-auto" style={{ background: 'rgba(0,0,0,.3)' }}>{code}</pre>
      </div>
    );
  }
  return <div ref={ref} className="my-2 flex justify-center overflow-auto [&>svg]:max-w-full" />;
}

/* ── Markdown with ```mermaid fences rendered as diagrams ── */
const markdownComponents = {
  pre({ node, children, ...rest }) {
    const code = node?.children?.[0];
    const classes = code?.properties?.className || [];
    if (code?.tagName === 'code' && classes.includes('language-mermaid')) {
      const text = (code.children || []).map(c => c.value || '').join('');
      return <MermaidDiagram code={text} />;
    }
    return <pre {...rest}>{children}</pre>;
  },
  a({ node, ...props }) {
    return <a {...props} target="_blank" rel="noopener noreferrer" />;
  },
};

function MarkdownPreview({ content }) {
  return (
    <div className="flex-1 overflow-auto p-5 markdown-body text-sm" style={{ color: 'var(--text-primary)', minHeight: 300 }}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {content || ''}
      </ReactMarkdown>
    </div>
  );
}

function PreviewMessage({ text }) {
  return (
    <div className="flex-1 flex items-center justify-center p-8 text-xs" style={{ color: 'var(--text-muted)', minHeight: 300 }}>
      {text}
    </div>
  );
}

export default function FilePreview({ path, content, kind = previewKind(path, content) }) {
  if (kind === 'pdf') return <PdfPreview content={content} path={path} />;
  if (kind === 'markdown') return <MarkdownPreview content={content} />;
  if (kind === 'mermaid') {
    return (
      <div className="flex-1 overflow-auto p-5" style={{ minHeight: 300 }}>
        <MermaidDiagram code={content || ''} />
      </div>
    );
  }
  return null;
}
