import { useState, useEffect, useRef } from "react";

// ── Shared data-fetching hook ──
export function useAPI(fetchFn, transformFn, deps = [], enabled = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState(null);
  const fetchedRef = useRef(false);
  useEffect(() => {
    if (!enabled || fetchedRef.current) return;
    fetchedRef.current = true;
    let cancelled = false;
    setLoading(true);
    fetchFn()
      .then(raw => { if (!cancelled) setData(transformFn ? transformFn(raw) : raw); })
      .catch(err => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ...deps]);
  return { data, loading, error };
}
